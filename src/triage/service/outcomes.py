"""Outcome tracking: what maintainers later did with each predicted issue.

For recently predicted issues, fetch their current state with the collector's issue fields
(batched GraphQL aliases), and record state, final labels, label history and duplicate
links. Duplicate direction uses the benchmark's own rules (`dataset.build.resolve_duplicate`),
so live outcomes and benchmark ground truth mean the same thing.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from psycopg import Connection
from psycopg.types.json import Jsonb

from triage.dataset.build import IssueFacts, resolve_duplicate
from triage.dataset.collect import ISSUE_FIELDS, normalize_issue
from triage.service.settings import Settings

log = logging.getLogger(__name__)

BATCH = 25  # issues per GraphQL query


def outcome_query(numbers: list[int]) -> str:
    aliases = "\n".join(f"    i{n}: issue(number: {n}) {{{ISSUE_FIELDS}    }}" for n in numbers)
    return (
        "query($owner: String!, $name: String!) {\n"
        "  rateLimit { cost remaining resetAt }\n"
        "  repository(owner: $owner, name: $name) {\n"
        f"{aliases}\n"
        "  }\n"
        "}\n"
    )


def fetch_issues(client, repo: str, numbers: list[int]) -> dict[int, dict[str, Any] | None]:
    """Normalized issue records by number; None for issues that no longer exist here."""
    owner, name = repo.split("/")
    out: dict[int, dict[str, Any] | None] = {}
    for i in range(0, len(numbers), BATCH):
        chunk = numbers[i : i + BATCH]
        data = client.query(outcome_query(chunk), {"owner": owner, "name": name})
        repo_data = data.get("repository") or {}
        for n in chunk:
            node = repo_data.get(f"i{n}")
            out[n] = normalize_issue(node, repo) if node else None
    return out


def due_issues(conn: Connection, repo: str, settings: Settings) -> list[int]:
    rows = conn.execute(
        """
        SELECT p.number FROM predictions p
        LEFT JOIN outcomes o ON o.repo = p.repo AND o.number = p.number
        WHERE p.repo = %(repo)s
          AND p.made_at > now() - make_interval(days => %(days)s)
          AND (o.checked_at IS NULL OR o.checked_at < now() - make_interval(secs => %(secs)s))
        ORDER BY o.checked_at NULLS FIRST, p.made_at
        LIMIT %(limit)s
        """,
        {
            "repo": repo,
            "days": settings.outcome_window_days,
            "secs": settings.outcome_interval_s,
            "limit": settings.outcome_batch,
        },
    ).fetchall()
    return [r["number"] for r in rows]


def check_outcomes(conn: Connection, settings: Settings, repo: str, client) -> int:
    numbers = due_issues(conn, repo, settings)
    if not numbers:
        return 0
    current = fetch_issues(client, repo, numbers)
    facts = _facts_for(conn, client, repo, current)
    for n in numbers:
        rec = current[n]
        if rec is None:
            _upsert(conn, repo, n, {"state": "MISSING"}, None)
            continue
        resolved = resolve_duplicate(n, rec["dup_links"], rec["dup_mentions"], facts)
        _upsert(conn, repo, n, rec, resolved)
    log.info("%s: checked outcomes for %d issues", repo, len(numbers))
    return len(numbers)


def _facts_for(conn, client, repo, current) -> dict[int, IssueFacts]:
    """Created-at and duplicate signal for the checked issues and every issue they link to.

    Linked issues come from the database when the service has seen them, otherwise they are
    fetched, so orientation never silently skips a link.
    """
    facts = {n: _facts(rec) for n, rec in current.items() if rec}
    others = {
        int(x)
        for rec in current.values()
        if rec
        for x in [link["other"] for link in rec["dup_links"]] + rec["dup_mentions"]
    }
    others -= set(facts)
    if others:
        rows = conn.execute(
            """
            SELECT i.number, i.created_at, i.labels, o.state_reason
            FROM issues i LEFT JOIN outcomes o ON o.repo = i.repo AND o.number = i.number
            WHERE i.repo = %s AND i.number = ANY(%s)
            """,
            (repo, list(others)),
        ).fetchall()
        for r in rows:
            signal = r["state_reason"] == "DUPLICATE" or any(
                "duplicate" in lbl.lower() for lbl in r["labels"]
            )
            facts[r["number"]] = IssueFacts(pd.Timestamp(r["created_at"]), signal)
        missing = sorted(others - set(facts))
        if missing:
            for n, rec in fetch_issues(client, repo, missing).items():
                if rec:
                    facts[n] = _facts(rec)
    return facts


def _facts(rec: dict[str, Any]) -> IssueFacts:
    signal = rec["state_reason"] == "DUPLICATE" or any(
        "duplicate" in lbl.lower() for lbl in rec["labels"]
    )
    return IssueFacts(pd.Timestamp(rec["created_at"]), signal)


def _upsert(conn, repo, number, rec, resolved) -> None:
    conn.execute(
        """
        INSERT INTO outcomes (repo, number, state, state_reason, closed_at, labels,
                              label_events, dup_links, dup_mentions, duplicate_of, oriented_by)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (repo, number) DO UPDATE SET
            checked_at = now(), state = EXCLUDED.state, state_reason = EXCLUDED.state_reason,
            closed_at = EXCLUDED.closed_at, labels = EXCLUDED.labels,
            label_events = EXCLUDED.label_events, dup_links = EXCLUDED.dup_links,
            dup_mentions = EXCLUDED.dup_mentions, duplicate_of = EXCLUDED.duplicate_of,
            oriented_by = EXCLUDED.oriented_by
        """,
        (
            repo,
            number,
            rec["state"],
            rec.get("state_reason"),
            rec.get("closed_at"),
            Jsonb(rec.get("labels") or []),
            Jsonb(rec.get("label_events") or []),
            Jsonb(rec.get("dup_links") or []),
            Jsonb(rec.get("dup_mentions") or []),
            resolved[0] if resolved else None,
            resolved[1] if resolved else None,
        ),
    )
