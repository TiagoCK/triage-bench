import pytest

pytest.importorskip("fastapi", reason="needs the service extra")
pytest.importorskip("psycopg", reason="needs the service extra")

from concurrent.futures import ThreadPoolExecutor  # noqa: E402

from triage.service import queue  # noqa: E402


def test_enqueue_dedupes_on_key(conn):
    first = queue.enqueue(conn, "noop", {}, max_attempts=3, dedupe_key="d1")
    again = queue.enqueue(conn, "noop", {}, max_attempts=3, dedupe_key="d1")
    conn.commit()
    assert first is not None and again is None
    assert queue.counts(conn) == {"queued": 1}


def test_claim_takes_oldest_and_marks_running(conn):
    ids = [queue.enqueue(conn, "noop", {"i": i}, max_attempts=3) for i in range(3)]
    conn.commit()
    job = queue.claim(conn, "w1")
    assert job.id == ids[0] and job.attempts == 1 and job.payload == {"i": 0}
    assert queue.counts(conn) == {"queued": 2, "running": 1}


def test_concurrent_workers_never_share_a_job(pool, conn):
    for i in range(40):
        queue.enqueue(conn, "noop", {"i": i}, max_attempts=3)
    conn.commit()

    def drain(worker):
        got = []
        with pool.connection() as c:
            while (job := queue.claim(c, worker)) is not None:
                got.append(job.id)
        return got

    with ThreadPoolExecutor(4) as ex:
        claimed = [j for batch in ex.map(drain, ["a", "b", "c", "d"]) for j in batch]
    assert sorted(claimed) == list(range(1, 41))  # each job exactly once


def test_failure_backs_off_then_goes_dead(conn):
    queue.enqueue(conn, "noop", {}, max_attempts=2)
    conn.commit()
    job = queue.claim(conn, "w")
    assert queue.fail(conn, job, "boom", backoff_base_s=60, backoff_max_s=600) == "queued"
    assert queue.claim(conn, "w") is None  # not ready until the backoff passes
    conn.execute("UPDATE jobs SET run_after = now()")
    conn.commit()
    job = queue.claim(conn, "w")
    assert job.attempts == 2
    assert queue.fail(conn, job, "boom again", 60, 600) == "dead"
    row = conn.execute("SELECT status, last_error FROM jobs").fetchone()
    assert row == {"status": "dead", "last_error": "boom again"}


def test_stale_running_jobs_are_requeued(conn):
    queue.enqueue(conn, "noop", {}, max_attempts=3)
    conn.commit()
    queue.claim(conn, "crashed-worker")
    conn.execute("UPDATE jobs SET locked_at = now() - interval '1 hour'")
    conn.commit()
    assert queue.requeue_stale(conn, lock_timeout_s=60) == 1
    assert queue.claim(conn, "w2").attempts == 2
