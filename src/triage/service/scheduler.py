"""The scheduler: a small loop that only enqueues periodic jobs; workers run them.

Each job's dedupe key names its time slot, so two schedulers (or a restart) within the same
slot add nothing, and failed periodic jobs get the queue's retries and dead-lettering.
"""

from __future__ import annotations

import logging
import signal
import time

from psycopg import Connection
from psycopg_pool import ConnectionPool

from triage.service import queue
from triage.service.settings import Settings

log = logging.getLogger(__name__)

# (job kind, settings attribute holding its interval in seconds)
PERIODIC = (("poll.new_issues", "poll_interval_s"), ("outcomes.check", "outcome_interval_s"))


def tick(conn: Connection, settings: Settings, now: float | None = None) -> list[int]:
    """Enqueue every periodic job due in the current slot. Returns the new job ids."""
    now = time.time() if now is None else now
    created = []
    for repo in settings.watch_repos:
        for kind, attr in PERIODIC:
            slot = int(now // getattr(settings, attr))
            job_id = queue.enqueue(
                conn,
                kind,
                {"repo": repo},
                settings.job_max_attempts,
                dedupe_key=f"{kind}:{repo}:{slot}",
            )
            if job_id is not None:
                created.append(job_id)
    conn.commit()
    return created


def run_forever(pool: ConnectionPool, settings: Settings) -> None:
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if not settings.watch_repos:
        log.warning("no TRIAGE_WATCH_REPOS configured; the scheduler has nothing to do")
    log.info("scheduler started for %s", settings.watch_repos)
    while not stopping:
        with pool.connection() as conn:
            if new := tick(conn, settings):
                log.info("enqueued %d periodic jobs", len(new))
        deadline = time.monotonic() + settings.scheduler_tick_s
        while not stopping and time.monotonic() < deadline:
            time.sleep(0.5)
