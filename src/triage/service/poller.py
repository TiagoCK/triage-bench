"""Poll watched public repos for new issues and feed them into the pipeline.

Uses the benchmark collector's GraphQL walk (newest first, stopping at the cursor), so live
issues are parsed exactly like the benchmark data. Read-only: it never writes to GitHub.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from psycopg import Connection

from triage.dataset.collect import iter_issue_nodes, normalize_issue
from triage.service import queue
from triage.service.settings import Settings

log = logging.getLogger(__name__)

KIND = "new_issues"


def poll_new_issues(
    conn: Connection, settings: Settings, repo: str, client, now: datetime | None = None
) -> int:
    """Enqueue an `opened` job per issue created since the cursor. Returns how many.

    Runs inside the job's transaction: the cursor only advances if the enqueues commit.
    """
    now = now or datetime.now(UTC)
    row = conn.execute(
        "SELECT cursor FROM poll_cursors WHERE repo = %s AND kind = %s", (repo, KIND)
    ).fetchone()
    cursor = row["cursor"] if row else now - timedelta(seconds=settings.poll_initial_lookback_s)

    records = [normalize_issue(node, repo) for node in iter_issue_nodes(client, repo, cursor)]
    records.sort(key=lambda r: r["created_at"])
    queued = 0
    for rec in records:
        job_id = queue.enqueue(
            conn,
            "github.issue",
            {
                "repo": repo,
                "action": "opened",
                "source": "poller",
                "issue": {
                    "number": rec["number"],
                    "title": rec["title"],
                    "body": rec["body"],
                    "state": (rec["state"] or "open").lower(),
                    "state_reason": rec["state_reason"],
                    "labels": rec["labels"],
                    "author": rec["author"],
                    "url": rec["url"],
                    "created_at": rec["created_at"],
                    "updated_at": rec["created_at"],
                },
            },
            settings.job_max_attempts,
            # Issues created in the cursor's own second are re-read; the key drops repeats.
            dedupe_key=f"poll:{repo}:{rec['number']}",
        )
        queued += job_id is not None
    newest = max((_ts(r["created_at"]) for r in records), default=cursor)
    conn.execute(
        """
        INSERT INTO poll_cursors (repo, kind, cursor) VALUES (%s, %s, %s)
        ON CONFLICT (repo, kind) DO UPDATE SET cursor = EXCLUDED.cursor, updated_at = now()
        """,
        (repo, KIND, max(newest, cursor)),
    )
    log.info("%s: %d new issues since %s (%d queued)", repo, len(records), cursor, queued)
    return queued


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
