"""Worker processes: claim jobs from the queue, run their handler, record the outcome.

Week 1's only real handler stores the issue. Later milestones add embedding, retrieval,
the duplicate judge and labeling to the same job.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import time
import traceback
from collections.abc import Callable
from typing import Any

from psycopg import Connection
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from triage.service import queue
from triage.service.settings import Settings

log = logging.getLogger(__name__)

Handler = Callable[[Connection, dict[str, Any]], None]


def store_issue(conn: Connection, payload: dict[str, Any]) -> None:
    """Upsert the issue as last seen. A stale event never overwrites newer data."""
    issue = payload["issue"]
    conn.execute(
        """
        INSERT INTO issues (repo, number, title, body, state, labels, author, url,
                            created_at, updated_at, source)
        VALUES (%(repo)s, %(number)s, %(title)s, %(body)s, %(state)s, %(labels)s,
                %(author)s, %(url)s, %(created_at)s, %(updated_at)s, %(source)s)
        ON CONFLICT (repo, number) DO UPDATE SET
            title = EXCLUDED.title, body = EXCLUDED.body, state = EXCLUDED.state,
            labels = EXCLUDED.labels, updated_at = EXCLUDED.updated_at, last_seen = now()
        WHERE issues.updated_at <= EXCLUDED.updated_at
        """,
        {
            **issue,
            "repo": payload["repo"],
            "labels": Jsonb(issue["labels"]),
            "source": payload.get("source", "webhook"),
        },
    )


# Store-only handlers (Week 1 behaviour); the running worker uses pipeline_handlers().
HANDLERS: dict[str, Handler] = {
    "github.issue": store_issue,
    "noop": lambda conn, payload: None,
}


def pipeline_handlers(settings: Settings) -> dict[str, Handler]:
    """Handlers that run the full pipeline. Loads the embedding model (once per process)."""
    from triage.service.embeddings import FastEmbedder
    from triage.service.pipeline import LabelerRegistry, Pipeline

    pipeline = Pipeline(
        settings,
        FastEmbedder(settings.embed_model, settings.model_cache_dir),
        LabelerRegistry(settings.labeler_dir),
    )
    return {**HANDLERS, "github.issue": pipeline.handle_issue}


def run_once(
    conn: Connection, settings: Settings, worker_id: str, handlers: dict[str, Handler] = HANDLERS
) -> queue.Job | None:
    """Claim and process one job. Returns the job, or None if the queue was empty."""
    job = queue.claim(conn, worker_id)
    if job is None:
        return None
    handler = handlers.get(job.kind)
    try:
        if handler is None:
            raise LookupError(f"no handler for job kind {job.kind!r}")
        with conn.transaction():
            handler(conn, job.payload)
    except Exception as exc:  # any handler failure is retried, then parked as dead
        conn.rollback()
        status = queue.fail(
            conn,
            job,
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            settings.job_backoff_base_s,
            settings.job_backoff_max_s,
        )
        log.warning(
            "job %s (%s) failed on attempt %d/%d -> %s: %s",
            job.id,
            job.kind,
            job.attempts,
            job.max_attempts,
            status,
            exc,
        )
        return job
    queue.complete(conn, job)
    log.info("job %s (%s) done", job.id, job.kind)
    return job


def run_forever(pool: ConnectionPool, settings: Settings) -> None:
    worker_id = f"{socket.gethostname()}:{os.getpid()}"
    stopping = False

    def stop(signum, frame):  # finish the current job, then exit
        nonlocal stopping
        stopping = True
        log.info("worker %s stopping after current job", worker_id)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    handlers = pipeline_handlers(settings)
    log.info(
        "worker %s started (mode=%s, embed_model=%s)",
        worker_id,
        settings.mode,
        settings.embed_model,
    )
    last_reap = 0.0
    while not stopping:
        with pool.connection() as conn:
            if time.monotonic() - last_reap > settings.job_lock_timeout_s / 2:
                if n := queue.requeue_stale(conn, settings.job_lock_timeout_s):
                    log.warning("requeued %d jobs from dead workers", n)
                last_reap = time.monotonic()
            job = run_once(conn, settings, worker_id, handlers)
        if job is None:
            time.sleep(settings.worker_poll_interval_s)
