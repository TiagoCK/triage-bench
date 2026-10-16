"""GitHub access for the service: GitHub App installation tokens, with a fallback to the
read-only personal token.

An App authenticates as itself with a short-lived JWT signed by its private key, then
trades that JWT for an installation token scoped to the repos it is installed on. Those
tokens last an hour; they are cached and refreshed 5 minutes before they expire.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

import httpx
import jwt
from psycopg import Connection

from triage.github_client import GitHubClient
from triage.service.settings import Settings

log = logging.getLogger(__name__)

REFRESH_MARGIN_S = 300
MAX_SLEEP_IN_JOB_S = 30.0


class RateLimited(RuntimeError):
    """GitHub asked us to wait longer than a job should block; the job retries later."""


def job_sleep(seconds: float) -> None:
    """A sleep for GitHubClient inside queue jobs: short waits are fine, long ones raise so
    the queue's backoff handles them instead of a blocked worker."""
    if seconds > MAX_SLEEP_IN_JOB_S:
        raise RateLimited(f"GitHub asked to wait {seconds:.0f}s")
    time.sleep(seconds)


class AppAuth:
    def __init__(
        self,
        app_id: int,
        private_key: str,
        api_url: str = "https://api.github.com",
        http: httpx.Client | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.app_id = app_id
        self._key = private_key
        self._api = api_url.rstrip("/")
        self._http = http or httpx.Client(timeout=30)
        self._clock = clock
        self._tokens: dict[int, tuple[str, float]] = {}

    def app_jwt(self) -> str:
        now = int(self._clock())
        # iat is backdated a minute for clock drift; GitHub caps exp at 10 minutes.
        claims = {"iat": now - 60, "exp": now + 9 * 60, "iss": str(self.app_id)}
        return jwt.encode(claims, self._key, algorithm="RS256")

    def installation_token(self, installation_id: int) -> str:
        cached = self._tokens.get(installation_id)
        if cached and cached[1] - self._clock() > REFRESH_MARGIN_S:
            return cached[0]
        resp = self._http.post(
            f"{self._api}/app/installations/{installation_id}/access_tokens",
            headers={
                "Authorization": f"Bearer {self.app_jwt()}",
                "Accept": "application/vnd.github+json",
                "User-Agent": "triage-bench",
            },
        )
        resp.raise_for_status()
        body = resp.json()
        expires = datetime.fromisoformat(body["expires_at"].replace("Z", "+00:00")).timestamp()
        self._tokens[installation_id] = (body["token"], expires)
        log.info("new installation token for %s, expires %s", installation_id, body["expires_at"])
        return body["token"]


class TokenProvider:
    """The token to read a repo with: the App's installation token if the App covers the
    repo, otherwise the read-only personal token."""

    def __init__(self, settings: Settings, app: AppAuth | None = None) -> None:
        self.settings = settings
        if app is None and settings.github_app_id and settings.github_app_private_key:
            app = AppAuth(
                settings.github_app_id,
                settings.github_app_private_key.get_secret_value(),
                settings.github_api_url,
            )
        self.app = app

    def token_for(self, conn: Connection, repo: str) -> str:
        if self.app is not None:
            row = conn.execute(
                "SELECT installation_id FROM installations WHERE repo = %s", (repo,)
            ).fetchone()
            if row:
                return self.app.installation_token(row["installation_id"])
        if self.settings.github_token is None:
            raise RuntimeError(
                f"no GitHub credentials for {repo}: set TRIAGE_GITHUB_TOKEN (or GITHUB_TOKEN),"
                " or install the GitHub App on it"
            )
        return self.settings.github_token.get_secret_value()

    def client_for(self, conn: Connection, repo: str) -> GitHubClient:
        # No page cache: live polling must always see fresh data.
        return GitHubClient(
            self.token_for(conn, repo), cache_dir=None, sleep=job_sleep, tolerate_not_found=True
        )


def record_installations(conn: Connection, event: str, payload: dict[str, Any]) -> None:
    """Keep the repo -> installation map current from webhook deliveries."""
    installation = (payload.get("installation") or {}).get("id")
    if not installation:
        return
    action = payload.get("action")
    if event == "installation" and action == "deleted":
        conn.execute("DELETE FROM installations WHERE installation_id = %s", (installation,))
        return
    if event == "installation_repositories":
        for r in payload.get("repositories_removed") or []:
            conn.execute("DELETE FROM installations WHERE repo = %s", (r["full_name"],))
        repos = [r["full_name"] for r in payload.get("repositories_added") or []]
    elif event == "installation":
        repos = [r["full_name"] for r in payload.get("repositories") or []]
    else:
        repo = (payload.get("repository") or {}).get("full_name")
        repos = [repo] if repo else []
    for repo in repos:
        conn.execute(
            "INSERT INTO installations (repo, installation_id) VALUES (%s, %s)"
            " ON CONFLICT (repo) DO UPDATE SET installation_id = EXCLUDED.installation_id,"
            " updated_at = now()",
            (repo, installation),
        )
