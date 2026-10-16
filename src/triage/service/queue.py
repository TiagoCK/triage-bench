"""A Postgres job queue.

Jobs move queued -> running -> done. A failure puts the job back in the queue after an
exponential backoff, until it has used `max_attempts`; then it is parked as `dead` for a
human to inspect, instead of retrying forever. Claiming uses FOR UPDATE SKIP LOCKED, so any
number of workers can poll the same table safely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from psycopg import Connection
from psycopg.types.json import Jsonb


@dataclass(frozen=True)
class Job:
    id: int
    kind: str
    payload: dict[str, Any]
    attempts: int
    max_attempts: int


def enqueue(
    conn: Connection,
    kind: str,
    payload: dict[str, Any],
    max_attempts: int,
    dedupe_key: str | None = None,
) -> int | None:
    """Add a job. Returns its id, or None if a job with the same dedupe_key exists."""
    row = conn.execute(
        """
        INSERT INTO jobs (kind, payload, max_attempts, dedupe_key)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (dedupe_key) DO NOTHING
        RETURNING id
        """,
        (kind, Jsonb(payload), max_attempts, dedupe_key),
    ).fetchone()
    return row["id"] if row else None


def claim(conn: Connection, worker_id: str) -> Job | None:
    """Take the oldest ready job, or None if there is nothing to do."""
    row = conn.execute(
        """
        UPDATE jobs SET status = 'running', attempts = attempts + 1,
               locked_at = now(), locked_by = %s, updated_at = now()
        WHERE id = (
            SELECT id FROM jobs
            WHERE status = 'queued' AND run_after <= now()
            ORDER BY run_after, id
            FOR UPDATE SKIP LOCKED
            LIMIT 1
        )
        RETURNING id, kind, payload, attempts, max_attempts
        """,
        (worker_id,),
    ).fetchone()
    conn.commit()
    return Job(**row) if row else None


def complete(conn: Connection, job: Job) -> None:
    conn.execute(
        "UPDATE jobs SET status = 'done', locked_at = NULL, locked_by = NULL,"
        " last_error = NULL, updated_at = now() WHERE id = %s",
        (job.id,),
    )
    conn.commit()


def fail(
    conn: Connection, job: Job, error: str, backoff_base_s: float, backoff_max_s: float
) -> str:
    """Record a failure; returns the job's new status ('queued' or 'dead')."""
    if job.attempts >= job.max_attempts:
        status, delay = "dead", 0.0
    else:
        status = "queued"
        delay = min(backoff_max_s, backoff_base_s * 2 ** (job.attempts - 1))
    conn.execute(
        """
        UPDATE jobs SET status = %s, last_error = %s, locked_at = NULL, locked_by = NULL,
               run_after = now() + make_interval(secs => %s), updated_at = now()
        WHERE id = %s
        """,
        (status, error[:2000], delay, job.id),
    )
    conn.commit()
    return status


def requeue_stale(conn: Connection, lock_timeout_s: float) -> int:
    """Put back jobs whose worker died mid-run. Returns how many were requeued."""
    rows = conn.execute(
        """
        UPDATE jobs SET status = 'queued', locked_at = NULL, locked_by = NULL,
               last_error = 'worker lock expired', updated_at = now()
        WHERE status = 'running' AND locked_at < now() - make_interval(secs => %s)
        RETURNING id
        """,
        (lock_timeout_s,),
    ).fetchall()
    conn.commit()
    return len(rows)


def counts(conn: Connection) -> dict[str, int]:
    rows = conn.execute("SELECT status, count(*) AS n FROM jobs GROUP BY status").fetchall()
    return {r["status"]: r["n"] for r in rows}
