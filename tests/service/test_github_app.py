import pytest

pytest.importorskip("jwt", reason="needs the service extra")
pytest.importorskip("psycopg", reason="needs the service extra")

import json  # noqa: E402

import httpx  # noqa: E402
import jwt  # noqa: E402
from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import rsa  # noqa: E402

from triage.service.github_app import (  # noqa: E402
    AppAuth,
    RateLimited,
    TokenProvider,
    job_sleep,
    record_installations,
)
from triage.service.settings import Settings  # noqa: E402


@pytest.fixture(scope="module")
def keypair():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    return pem, key.public_key()


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def token_server(calls):
    def handler(request):
        calls.append(request)
        n = len(calls)
        return httpx.Response(
            201, json={"token": f"inst-token-{n}", "expires_at": "2027-01-15T08:00:00Z"}
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_app_jwt_claims(keypair):
    pem, public = keypair
    clock = Clock()
    token = AppAuth(12345, pem, clock=clock).app_jwt()
    claims = jwt.decode(
        token,
        public,
        algorithms=[
            "RS256"
        ],  # The fake clock is not "now", so only the claims are checked, not their timing.
        options={"verify_exp": False, "verify_iat": False},
    )
    assert claims["iss"] == "12345"
    assert claims["iat"] == int(clock.t) - 60
    assert claims["exp"] - int(clock.t) <= 600  # GitHub's 10-minute cap


def test_installation_token_cached_until_near_expiry(keypair):
    pem, _ = keypair
    calls = []
    from datetime import datetime

    expires = datetime.fromisoformat("2027-01-15T08:00:00+00:00").timestamp()
    clock = Clock(expires - 3600)
    auth = AppAuth(1, pem, http=token_server(calls), clock=clock)
    assert auth.installation_token(77) == "inst-token-1"
    assert auth.installation_token(77) == "inst-token-1"  # cached
    assert len(calls) == 1
    assert calls[0].url.path == "/app/installations/77/access_tokens"
    assert calls[0].headers["Authorization"].startswith("Bearer ")
    clock.t = expires - 200  # inside the 5-minute refresh margin
    assert auth.installation_token(77) == "inst-token-2"


def test_job_sleep_raises_on_long_waits():
    job_sleep(0)
    with pytest.raises(RateLimited):
        job_sleep(3600)


def test_token_provider_prefers_installation_then_falls_back(conn, keypair):
    pem, _ = keypair
    calls = []
    settings = Settings(github_token="read-only-pat")
    app = AppAuth(1, pem, http=token_server(calls), clock=Clock(1_700_000_000))
    provider = TokenProvider(settings, app)
    assert provider.token_for(conn, "acme/widgets") == "read-only-pat"  # not installed
    record_installations(
        conn,
        "issues",
        {"installation": {"id": 9}, "repository": {"full_name": "acme/widgets"}},
    )
    assert provider.token_for(conn, "acme/widgets") == "inst-token-1"

    with pytest.raises(RuntimeError, match="no GitHub credentials"):
        TokenProvider(Settings(github_token=None), None).token_for(conn, "acme/other")


def test_installation_events_update_the_map(conn):
    def repos():
        return {
            r["repo"]: r["installation_id"]
            for r in conn.execute("SELECT repo, installation_id FROM installations")
        }

    record_installations(
        conn,
        "installation",
        {
            "action": "created",
            "installation": {"id": 5},
            "repositories": [{"full_name": "a/one"}, {"full_name": "a/two"}],
        },
    )
    assert repos() == {"a/one": 5, "a/two": 5}
    record_installations(
        conn,
        "installation_repositories",
        {
            "action": "removed",
            "installation": {"id": 5},
            "repositories_removed": [{"full_name": "a/two"}],
            "repositories_added": [],
        },
    )
    assert repos() == {"a/one": 5}
    record_installations(conn, "installation", {"action": "deleted", "installation": {"id": 5}})
    assert repos() == {}


def test_webhook_receiver_records_installation(settings, pool, conn):
    from fastapi.testclient import TestClient

    from triage.service.app import create_app

    from .test_api import post
    from .test_webhooks_unit import issue_payload

    payload = {**issue_payload(), "installation": {"id": 42}}
    with TestClient(create_app(settings, pool)) as client:
        assert post(client, payload).status_code == 202
    row = conn.execute("SELECT installation_id FROM installations").fetchone()
    assert row["installation_id"] == 42
    assert json.dumps(payload)  # payload stays JSON-serializable
