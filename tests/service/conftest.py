"""Fixtures for service tests that need Postgres.

Set TRIAGE_TEST_DATABASE_URL to a server you can create databases on (`make service-db`
starts one in Docker). Each test session gets a throwaway database, dropped at the end.
Without the variable, or without the `service` extra, these tests are skipped.
"""

import os
import uuid

import pytest

TABLES = (
    "jobs",
    "webhook_deliveries",
    "predictions",
    "issue_embeddings",
    "outcomes",
    "poll_cursors",
    "installations",
    "issues",
)


@pytest.fixture(scope="session")
def database_url():
    pytest.importorskip("psycopg", reason="needs the service extra")
    import psycopg
    from psycopg.conninfo import make_conninfo

    admin_url = os.environ.get("TRIAGE_TEST_DATABASE_URL")
    if not admin_url:
        pytest.skip("TRIAGE_TEST_DATABASE_URL not set (run `make service-db`)")
    name = f"triage_test_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{name}"')
    yield make_conninfo(admin_url, dbname=name)
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


@pytest.fixture(scope="session")
def pool(database_url):
    from triage.service.db import make_pool, migrate

    pool = make_pool(database_url, max_size=5)
    with pool.connection() as conn:
        migrate(conn)
    yield pool
    pool.close()


@pytest.fixture
def conn(pool):
    """A connection on an empty schema."""
    with pool.connection() as conn:
        conn.execute(f"TRUNCATE {', '.join(TABLES)} RESTART IDENTITY")
        conn.commit()
        yield conn


@pytest.fixture
def settings(database_url):
    from triage.service.settings import Settings

    return Settings(
        database_url=database_url,
        webhook_secret="test-secret",
        job_max_attempts=3,
        job_backoff_base_s=0.0,
        job_backoff_max_s=0.0,
    )
