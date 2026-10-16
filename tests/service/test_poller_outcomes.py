import pytest

pytest.importorskip("fastapi", reason="needs the service extra")
pytest.importorskip("psycopg", reason="needs the service extra")

from datetime import UTC, datetime  # noqa: E402

from test_collect import marked, node  # noqa: E402

from triage.service import queue  # noqa: E402
from triage.service.outcomes import check_outcomes, due_issues, outcome_query  # noqa: E402
from triage.service.poller import poll_new_issues  # noqa: E402
from triage.service.scheduler import tick  # noqa: E402
from triage.service.settings import Settings  # noqa: E402

REPO = "acme/widgets"
NOW = datetime(2026, 10, 16, 12, 0, tzinfo=UTC)


def ts(day, hour=0):
    return f"2026-10-{day:02d}T{hour:02d}:00:00Z"


class PagedClient:
    """Answers the collector's ISSUES_QUERY with recorded pages, newest first."""

    def __init__(self, nodes, page_size=2, fail_on_page=None):
        self.nodes = nodes
        self.page_size = page_size
        self.fail_on_page = fail_on_page
        self.pages = 0

    def query(self, query, variables):
        self.pages += 1
        if self.fail_on_page == self.pages:
            raise RuntimeError("network down")
        start = int(variables["after"] or 0)
        chunk = self.nodes[start : start + self.page_size]
        end = start + len(chunk)
        return {
            "repository": {
                "issues": {
                    "pageInfo": {"hasNextPage": end < len(self.nodes), "endCursor": str(end)},
                    "nodes": chunk,
                }
            }
        }


def queued_numbers(conn):
    return sorted(
        r["payload"]["issue"]["number"]
        for r in conn.execute("SELECT payload FROM jobs WHERE kind = 'github.issue'")
    )


def cursor(conn):
    row = conn.execute("SELECT cursor FROM poll_cursors").fetchone()
    return row["cursor"] if row else None


# ---------------------------------------------------------------- scheduler


def test_scheduler_enqueues_once_per_slot(conn):
    settings = Settings(
        watch_repos=[REPO, "acme/other"], poll_interval_s=600, outcome_interval_s=3600
    )
    first = tick(conn, settings, now=1_000_000.0)
    assert len(first) == 4  # 2 repos x (poll, outcomes)
    assert tick(conn, settings, now=1_000_100.0) == []  # same slots
    later = tick(conn, settings, now=1_000_700.0)  # next poll slot, same outcome slot
    assert len(later) == 2
    kinds = [r["kind"] for r in conn.execute("SELECT kind FROM jobs WHERE id = ANY(%s)", (later,))]
    assert set(kinds) == {"poll.new_issues"}


# ---------------------------------------------------------------- poller


def test_first_poll_uses_the_lookback_window(conn):
    settings = Settings(poll_initial_lookback_s=24 * 3600)
    nodes = [node(13, ts(16, 9)), node(12, ts(15, 18)), node(11, ts(14, 9)), node(10, ts(13, 9))]
    with conn.transaction():
        n = poll_new_issues(conn, settings, REPO, PagedClient(nodes), now=NOW)
    assert n == 2 and queued_numbers(conn) == [12, 13]  # only the last 24 hours
    assert cursor(conn) == datetime(2026, 10, 16, 9, tzinfo=UTC)
    job = conn.execute("SELECT payload FROM jobs WHERE kind='github.issue' LIMIT 1").fetchone()
    assert job["payload"]["action"] == "opened" and job["payload"]["source"] == "poller"


def test_next_poll_picks_up_only_new_issues(conn):
    settings = Settings(poll_initial_lookback_s=24 * 3600)
    old = [node(13, ts(16, 9)), node(12, ts(15, 18))]
    with conn.transaction():
        poll_new_issues(conn, settings, REPO, PagedClient(old), now=NOW)
    newer = [node(15, ts(16, 11)), node(14, ts(16, 10)), *old]
    with conn.transaction():
        assert poll_new_issues(conn, settings, REPO, PagedClient(newer), now=NOW) == 2
    assert queued_numbers(conn) == [12, 13, 14, 15]
    with conn.transaction():  # nothing new: nothing queued, cursor unchanged
        assert poll_new_issues(conn, settings, REPO, PagedClient(newer), now=NOW) == 0
    assert cursor(conn) == datetime(2026, 10, 16, 11, tzinfo=UTC)


def test_failed_poll_moves_nothing(conn):
    settings = Settings(poll_initial_lookback_s=48 * 3600)
    nodes = [node(13, ts(16, 9)), node(12, ts(15, 18)), node(11, ts(15, 9))]
    with pytest.raises(RuntimeError), conn.transaction():
        poll_new_issues(conn, settings, REPO, PagedClient(nodes, fail_on_page=2), now=NOW)
    assert queued_numbers(conn) == [] and cursor(conn) is None


# ---------------------------------------------------------------- outcomes


def insert_issue(conn, number, created, labels=()):
    conn.execute(
        "INSERT INTO issues (repo, number, title, state, labels, created_at, updated_at,"
        " source) VALUES (%s, %s, 't', 'open', %s::jsonb, %s, %s, 'test')",
        (REPO, number, __import__("json").dumps(list(labels)), created, created),
    )


def insert_prediction(conn, number, created):
    conn.execute(
        "INSERT INTO predictions (repo, number, mode, issue_created_at, candidates,"
        " embed_model, code_version, timings_ms)"
        " VALUES (%s, %s, 'shadow', %s, '[]', 'm', 'v', '{}')",
        (REPO, number, created),
    )


class AliasClient:
    """Answers outcome queries (aliased issue lookups) from recorded nodes."""

    def __init__(self, nodes):
        self.nodes = nodes
        self.queries = []

    def query(self, query, variables):
        self.queries.append(query)
        wanted = [
            int(line.split(":")[0].strip()[1:])
            for line in query.splitlines()
            if line.strip().startswith("i") and ": issue(number:" in line
        ]
        return {"repository": {f"i{n}": self.nodes.get(n) for n in wanted}}


def test_outcome_query_aliases_each_issue():
    q = outcome_query([5, 9])
    assert "i5: issue(number: 5)" in q and "i9: issue(number: 9)" in q
    assert "labelEvents: timelineItems" in q  # the collector's fields


def test_outcomes_record_state_labels_and_duplicates(conn):
    settings = Settings()
    insert_issue(conn, 3, ts(1))
    insert_issue(conn, 10, ts(10))
    insert_issue(conn, 11, ts(11))
    insert_prediction(conn, 10, ts(10))
    insert_prediction(conn, 11, ts(11))
    conn.commit()
    nodes = {
        # #10 closed as a duplicate of #3, linked only on #3's timeline in reality; here
        # the reversed-fields event is on #10 itself, as seen on astral-sh/uv.
        10: {
            **node(
                10,
                ts(10),
                timeline=[marked(dup=3, canon=10)],
                labels=["bug"],
                state_reason="DUPLICATE",
            ),
            "state": "CLOSED",
            "closedAt": ts(12),
        },
        # #11 got a label from a maintainer and stays open.
        11: node(11, ts(11), labels=["area: cli"]),
    }
    client = AliasClient(nodes)
    with conn.transaction():
        assert check_outcomes(conn, settings, REPO, client) == 2
    rows = {r["number"]: r for r in conn.execute("SELECT * FROM outcomes")}
    assert rows[10]["state"] == "CLOSED" and rows[10]["state_reason"] == "DUPLICATE"
    assert rows[10]["duplicate_of"] == 3 and rows[10]["oriented_by"] == "close_reason"
    assert rows[11]["labels"] == ["area: cli"] and rows[11]["duplicate_of"] is None
    assert len(client.queries) == 1  # #3 was known locally, nothing extra fetched

    # Checked just now, so not due again until the interval passes.
    assert due_issues(conn, REPO, settings) == []
    conn.execute("UPDATE outcomes SET checked_at = now() - interval '2 hours'")
    conn.commit()
    assert sorted(due_issues(conn, REPO, settings)) == [10, 11]


def test_unknown_linked_issue_is_fetched_and_missing_issue_recorded(conn):
    settings = Settings()
    insert_issue(conn, 20, ts(10))
    insert_issue(conn, 21, ts(11))
    insert_prediction(conn, 20, ts(10))
    insert_prediction(conn, 21, ts(11))
    conn.commit()
    nodes = {
        20: {
            **node(20, ts(10), timeline=[marked(dup=20, canon=7)], state_reason="DUPLICATE"),
            "state": "CLOSED",
        },
        7: node(7, ts(1)),  # not in the issues table: must be fetched to orient the link
        # 21 is absent: deleted or transferred
    }
    client = AliasClient(nodes)
    with conn.transaction():
        check_outcomes(conn, settings, REPO, client)
    rows = {r["number"]: r for r in conn.execute("SELECT * FROM outcomes")}
    assert rows[20]["duplicate_of"] == 7
    assert rows[21]["state"] == "MISSING"
    assert len(client.queries) == 2


def test_old_predictions_are_not_checked(conn):
    insert_issue(conn, 30, ts(1))
    insert_prediction(conn, 30, ts(1))
    conn.execute("UPDATE predictions SET made_at = now() - interval '40 days'")
    conn.commit()
    assert due_issues(conn, REPO, Settings(outcome_window_days=30)) == []
    assert queue.counts(conn) == {}
