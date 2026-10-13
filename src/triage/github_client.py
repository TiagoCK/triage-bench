"""Minimal GitHub GraphQL client: retries, rate-limit awareness, and an on-disk page cache.

The cache makes collection resumable and reruns deterministic: a page is keyed by the query
and its variables, so re-running `triage collect` replays cached pages instead of hitting
the API. Delete `.cache/github` to force a fresh pull.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

GRAPHQL_URL = "https://api.github.com/graphql"
RETRY_STATUSES = {502, 503, 504}
MAX_RETRIES = 6
# Pause when the remaining GraphQL point budget drops below this.
RATE_LIMIT_FLOOR = 50


class GitHubError(RuntimeError):
    pass


class GitHubClient:
    def __init__(
        self,
        token: str,
        cache_dir: Path | None = Path(".cache/github"),
        transport: httpx.BaseTransport | None = None,
        sleep=time.sleep,
    ) -> None:
        if not token:
            raise GitHubError("GITHUB_TOKEN is empty. GraphQL requires a token; see .env.example.")
        self._http = httpx.Client(
            headers={
                "Authorization": f"Bearer {token}",
                "User-Agent": "triage-bench",
            },
            timeout=60,
            transport=transport,
        )
        self._cache_dir = cache_dir
        self._sleep = sleep
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def query(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        cache_path = self._cache_path(query, variables)
        if cache_path and cache_path.exists():
            return json.loads(cache_path.read_text())

        data = self._post_with_retries(query, variables)
        if cache_path:
            cache_path.write_text(json.dumps(data))
        self._respect_rate_limit(data)
        return data

    def _post_with_retries(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        delay = 2.0
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self._http.post(GRAPHQL_URL, json={"query": query, "variables": variables})
            except httpx.TransportError as exc:
                log.warning("transport error (attempt %d): %s", attempt, exc)
                self._sleep(delay)
                delay *= 2
                continue

            if resp.status_code in RETRY_STATUSES:
                log.warning("HTTP %d (attempt %d), retrying", resp.status_code, attempt)
                self._sleep(delay)
                delay *= 2
                continue
            if resp.status_code in (403, 429) and _is_secondary_rate_limit(resp):
                wait = float(resp.headers.get("retry-after", 60))
                log.warning("secondary rate limit, sleeping %.0fs", wait)
                self._sleep(wait)
                continue
            if resp.status_code != 200:
                raise GitHubError(f"HTTP {resp.status_code}: {resp.text[:500]}")

            payload = resp.json()
            if payload.get("errors"):
                # Large pages occasionally time out server-side; those come back as errors
                # with partial data. Retry them; anything else is a real query bug.
                messages = "; ".join(e.get("message", "") for e in payload["errors"])
                if "timeout" in messages.lower() or "something went wrong" in messages.lower():
                    log.warning("GraphQL transient error (attempt %d): %s", attempt, messages)
                    self._sleep(delay)
                    delay *= 2
                    continue
                raise GitHubError(f"GraphQL errors: {messages}")
            return payload["data"]

        raise GitHubError(f"giving up after {MAX_RETRIES} attempts")

    def _respect_rate_limit(self, data: dict[str, Any]) -> None:
        rl = data.get("rateLimit")
        if not rl or rl.get("remaining", RATE_LIMIT_FLOOR + 1) > RATE_LIMIT_FLOOR:
            return
        reset = datetime.fromisoformat(rl["resetAt"].replace("Z", "+00:00"))
        wait = max(0.0, (reset - datetime.now(UTC)).total_seconds()) + 5
        log.warning("GraphQL budget low (%s left), sleeping %.0fs", rl["remaining"], wait)
        self._sleep(wait)

    def _cache_path(self, query: str, variables: dict[str, Any]) -> Path | None:
        if not self._cache_dir:
            return None
        key = json.dumps({"q": query, "v": variables}, sort_keys=True)
        return self._cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()}.json"


def _is_secondary_rate_limit(resp: httpx.Response) -> bool:
    return "retry-after" in resp.headers or "secondary rate limit" in resp.text.lower()
