"""Webhook signature checks and event routing. Pure functions: no database, no extras."""

import hashlib
import hmac

from triage.service.webhooks import job_for_event, verify_signature

SECRET = "s3cret"
BODY = b'{"action": "opened"}'


def sign(body: bytes, secret: str = SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def test_valid_signature():
    assert verify_signature(SECRET, BODY, sign(BODY))


def test_rejects_wrong_secret_tampered_body_and_bad_headers():
    assert not verify_signature(SECRET, BODY, sign(BODY, "other"))
    assert not verify_signature(SECRET, BODY + b" ", sign(BODY))
    assert not verify_signature(SECRET, BODY, None)
    assert not verify_signature(SECRET, BODY, sign(BODY).replace("sha256=", "sha1="))


def issue_payload(**overrides):
    issue = {
        "number": 7,
        "title": "Crash on start",
        "body": None,
        "state": "open",
        "labels": [{"name": "bug"}],
        "user": {"login": "alice"},
        "html_url": "https://github.com/acme/widgets/issues/7",
        "created_at": "2026-10-01T00:00:00Z",
        "updated_at": "2026-10-01T00:00:00Z",
    }
    issue.update(overrides)
    return {"action": "opened", "issue": issue, "repository": {"full_name": "acme/widgets"}}


def test_opened_issue_becomes_a_job_with_only_needed_fields():
    kind, payload = job_for_event("issues", issue_payload())
    assert kind == "github.issue"
    assert payload["repo"] == "acme/widgets"
    assert payload["issue"]["labels"] == ["bug"]
    assert payload["issue"]["body"] == ""  # null bodies normalized
    assert "user" not in payload["issue"]


def test_pull_requests_and_unhandled_events_are_ignored():
    assert job_for_event("issues", issue_payload(pull_request={"url": "x"})) is None
    assert job_for_event("issues", {**issue_payload(), "action": "pinned"}) is None
    assert job_for_event("star", issue_payload()) is None
    assert job_for_event("issues", {"action": "opened"}) is None


def test_comment_events_carry_the_comment():
    payload = {
        **issue_payload(),
        "action": "created",
        "comment": {"id": 1, "body": "Duplicate of #3", "user": {"login": "maint"}},
    }
    kind, job = job_for_event("issue_comment", payload)
    assert job["action"] == "comment_created"
    assert job["comment"]["body"] == "Duplicate of #3"
