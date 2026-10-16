import json

import pytest

pytest.importorskip("fastapi", reason="needs the service extra")
pytest.importorskip("psycopg", reason="needs the service extra")

from fastapi.testclient import TestClient  # noqa: E402

from triage.service.app import create_app  # noqa: E402
from triage.service.settings import Settings  # noqa: E402

from .test_webhooks_unit import issue_payload, sign  # noqa: E402


@pytest.fixture
def client(settings, pool, conn):
    with TestClient(create_app(settings, pool)) as client:
        yield client


def post(client, payload, event="issues", delivery="d-1", secret="test-secret"):
    body = json.dumps(payload).encode()
    return client.post(
        "/webhooks/github",
        content=body,
        headers={
            "X-Hub-Signature-256": sign(body, secret),
            "X-GitHub-Event": event,
            "X-GitHub-Delivery": delivery,
            "Content-Type": "application/json",
        },
    )


def test_valid_webhook_is_queued(client, conn):
    resp = post(client, issue_payload())
    assert resp.status_code == 202
    job = conn.execute("SELECT kind, payload, dedupe_key FROM jobs").fetchone()
    assert job["kind"] == "github.issue"
    assert job["payload"]["source"] == "webhook"
    assert job["dedupe_key"] == "d-1"


def test_repeat_delivery_is_processed_once(client, conn):
    assert post(client, issue_payload()).status_code == 202
    again = post(client, issue_payload())
    assert again.status_code == 200 and again.json() == {"duplicate": True}
    assert conn.execute("SELECT count(*) AS n FROM jobs").fetchone()["n"] == 1


def test_bad_signature_is_rejected_and_not_recorded(client, conn):
    assert post(client, issue_payload(), secret="wrong").status_code == 401
    assert conn.execute("SELECT count(*) AS n FROM webhook_deliveries").fetchone()["n"] == 0


def test_ignored_events_are_acknowledged_but_not_queued(client, conn):
    resp = post(client, issue_payload(pull_request={"url": "x"}), delivery="d-pr")
    assert resp.status_code == 200 and resp.json() == {"ignored": True}
    assert conn.execute("SELECT count(*) AS n FROM jobs").fetchone()["n"] == 0
    ping = post(client, {"zen": "hi"}, event="ping", delivery="d-ping")
    assert ping.status_code == 200


def test_missing_secret_refuses_everything(database_url, pool, conn):
    app = create_app(Settings(database_url=database_url, webhook_secret=None), pool)
    with TestClient(app) as client:
        assert post(client, issue_payload()).status_code == 503


def test_missing_headers_and_bad_json(client):
    body = b"not json"
    headers = {"X-Hub-Signature-256": sign(body, "test-secret"), "X-GitHub-Event": "issues"}
    assert client.post("/webhooks/github", content=body, headers=headers).status_code == 400
    headers["X-GitHub-Delivery"] = "d-x"
    assert client.post("/webhooks/github", content=body, headers=headers).status_code == 400


def test_healthz_reports_queue(client):
    post(client, issue_payload())
    health = client.get("/healthz").json()
    assert health["status"] == "ok" and health["mode"] == "shadow"
    assert health["jobs"] == {"queued": 1}
