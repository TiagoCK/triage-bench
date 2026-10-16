"""Postgres connection pool and a minimal migration runner.

Migrations are numbered SQL files in `migrations/`, applied in order, each in its own
transaction, and recorded in `schema_migrations`. The API and workers both run `migrate()`
at startup; an advisory lock makes concurrent starts safe.
"""

from __future__ import annotations

import logging
from importlib import resources

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

log = logging.getLogger(__name__)

# Arbitrary constant shared by every process that migrates this database.
_MIGRATION_LOCK_ID = 7_421_001


def make_pool(database_url: str, min_size: int = 1, max_size: int = 10) -> ConnectionPool:
    pool = ConnectionPool(
        database_url,
        min_size=min_size,
        max_size=max_size,
        kwargs={"row_factory": dict_row},
        open=False,
    )
    pool.open(wait=True, timeout=30)
    return pool


def migration_files() -> list[tuple[str, str]]:
    root = resources.files("triage.service") / "migrations"
    files = sorted(f for f in root.iterdir() if f.name.endswith(".sql"))
    return [(f.name, f.read_text()) for f in files]


def migrate(conn: Connection) -> list[str]:
    """Apply pending migrations; returns the names applied."""
    applied_now = []
    conn.execute("SELECT pg_advisory_lock(%s)", (_MIGRATION_LOCK_ID,))
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
        conn.commit()
        done = {r["name"] for r in conn.execute("SELECT name FROM schema_migrations")}
        for name, sql in migration_files():
            if name in done:
                continue
            with conn.transaction():
                conn.execute(sql)
                conn.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (name,))
            log.info("applied migration %s", name)
            applied_now.append(name)
        conn.commit()
    finally:
        conn.execute("SELECT pg_advisory_unlock(%s)", (_MIGRATION_LOCK_ID,))
        conn.commit()
    return applied_now
