import pytest

pytest.importorskip("fastapi", reason="needs the service extra")
pytest.importorskip("psycopg", reason="needs the service extra")

from triage.service import queue  # noqa: E402
from triage.service.webhooks import job_for_event  # noqa: E402
from triage.service.worker import HANDLERS, run_once  # noqa: E402

from .test_webhooks_unit import issue_payload  # noqa: E402


def enqueue_event(conn, settings, **issue_overrides):
    kind, payload = job_for_event("issues", issue_payload(**issue_overrides))
    payload["source"] = "webhook"
    queue.enqueue(conn, kind, payload, settings.job_max_attempts)
    conn.commit()


def test_issue_job_stores_the_issue(conn, settings):
    enqueue_event(conn, settings)
    job = run_once(conn, settings, "w")
    assert job.kind == "github.issue"
    row = conn.execute("SELECT repo, number, title, labels, source FROM issues").fetchone()
    assert row == {
        "repo": "acme/widgets",
        "number": 7,
        "title": "Crash on start",
        "labels": ["bug"],
        "source": "webhook",
    }
    assert queue.counts(conn) == {"done": 1}
    assert run_once(conn, settings, "w") is None  # queue now empty


def test_older_event_never_overwrites_newer_data(conn, settings):
    enqueue_event(conn, settings, title="New title", updated_at="2026-10-05T00:00:00Z")
    enqueue_event(conn, settings, title="Old title", updated_at="2026-10-02T00:00:00Z")
    run_once(conn, settings, "w")
    run_once(conn, settings, "w")
    assert conn.execute("SELECT title FROM issues").fetchone()["title"] == "New title"


def test_failing_handler_retries_then_parks_as_dead(conn, settings):
    calls = []

    def flaky(c, payload):
        calls.append(1)
        c.execute(
            "INSERT INTO issues (repo, number, title, state, created_at, updated_at,"
            " source) VALUES ('x/y', 1, 't', 'open', now(), now(), 'test')"
        )
        raise RuntimeError("downstream unavailable")

    queue.enqueue(conn, "flaky", {}, settings.job_max_attempts)
    conn.commit()
    handlers = {**HANDLERS, "flaky": flaky}
    for _ in range(settings.job_max_attempts):
        run_once(conn, settings, "w", handlers)
    assert len(calls) == 3
    row = conn.execute("SELECT status, attempts, last_error FROM jobs").fetchone()
    assert row["status"] == "dead" and row["attempts"] == 3
    assert "downstream unavailable" in row["last_error"]
    # The failed handler's writes were rolled back each time.
    assert conn.execute("SELECT count(*) AS n FROM issues").fetchone()["n"] == 0


def test_unknown_job_kind_is_not_silently_dropped(conn, settings):
    queue.enqueue(conn, "mystery", {}, max_attempts=1)
    conn.commit()
    run_once(conn, settings, "w")
    row = conn.execute("SELECT status, last_error FROM jobs").fetchone()
    assert row["status"] == "dead" and "no handler" in row["last_error"]
