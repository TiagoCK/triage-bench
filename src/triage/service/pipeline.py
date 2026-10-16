"""What the worker does with an issue: embed it, find similar earlier issues, suggest labels,
and record the prediction.

Text handling (`clean_body`, `embedding_text`, `issue_text`) is imported from the benchmark,
so the live service runs the logic that was evaluated. `triage service replay` checks this
end to end.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from psycopg import Connection
from psycopg.types.json import Jsonb

from triage.dataset.build import clean_body, issue_text
from triage.service.embeddings import Embedder, text_hash, to_pgvector
from triage.service.retrieval import similar_issues
from triage.service.settings import Settings
from triage.service.worker import store_issue
from triage.systems.duplicates import embedding_text
from triage.systems.labels import TfidfLabeler

log = logging.getLogger(__name__)

PREDICT_ACTIONS = {"opened", "transferred"}


class LabelerRegistry:
    """Loads each repo's LATEST exported labeler once, on first use."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._cache: dict[str, tuple[TfidfLabeler, str] | None] = {}

    def get(self, repo: str) -> tuple[TfidfLabeler, str] | None:
        if repo not in self._cache:
            self._cache[repo] = self._load(repo)
        return self._cache[repo]

    def set(self, repo: str, labeler: TfidfLabeler, version: str) -> None:
        self._cache[repo] = (labeler, version)

    def _load(self, repo: str) -> tuple[TfidfLabeler, str] | None:
        latest = self.root / repo.replace("/", "__") / "LATEST"
        if not latest.exists():
            log.warning("no exported labeler for %s (triage export-labeler); labels skipped", repo)
            return None
        version = latest.read_text().strip()
        return TfidfLabeler.load(latest.parent / version), version


class Pipeline:
    def __init__(self, settings: Settings, embedder: Embedder, labelers: LabelerRegistry):
        self.settings = settings
        self.embedder = embedder
        self.labelers = labelers

    def handle_issue(self, conn: Connection, payload: dict[str, Any]) -> None:
        store_issue(conn, payload)
        repo, number = payload["repo"], payload["issue"]["number"]
        # Work from the stored row, not the event: a late, stale event must not win.
        issue = conn.execute(
            "SELECT title, body, created_at FROM issues WHERE repo = %s AND number = %s",
            (repo, number),
        ).fetchone()
        timings: dict[str, float] = {}

        t = time.perf_counter()
        body_clean = clean_body(issue["body"])
        self._embed_if_changed(conn, repo, number, issue["title"], body_clean)
        timings["embed"] = _ms(t)

        if payload.get("action") not in PREDICT_ACTIONS or self._has_prediction(conn, repo, number):
            return
        t = time.perf_counter()
        candidates = similar_issues(
            conn, repo, number, issue["created_at"], self.embedder.name, self.settings.retrieval_k
        )
        timings["retrieve"] = _ms(t)

        t = time.perf_counter()
        labels = scores = version = None
        loaded = self.labelers.get(repo)
        if loaded:
            labeler, version = loaded
            proba = labeler.proba([issue_text(issue["title"], body_clean)])[0]
            labels = labeler.pick(proba)
            scores = {
                str(c): round(float(p), 6) for c, p in zip(labeler.classes, proba, strict=True)
            }
        timings["label"] = _ms(t)

        conn.execute(
            """
            INSERT INTO predictions (repo, number, mode, issue_created_at, candidates, labels,
                                     label_scores, embed_model, labeler_version, code_version,
                                     timings_ms)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (repo, number, kind) DO NOTHING
            """,
            (
                repo,
                number,
                self.settings.mode,
                issue["created_at"],
                Jsonb([{"number": n, "similarity": round(s, 6)} for n, s in candidates]),
                Jsonb(labels) if labels is not None else None,
                Jsonb(scores) if scores is not None else None,
                self.embedder.name,
                version,
                self.settings.code_version,
                Jsonb(timings),
            ),
        )

    def _embed_if_changed(self, conn, repo, number, title, body_clean) -> None:
        text = embedding_text(title, body_clean)
        h = text_hash(text)
        row = conn.execute(
            "SELECT text_hash, model FROM issue_embeddings WHERE repo = %s AND number = %s",
            (repo, number),
        ).fetchone()
        if row and row["text_hash"] == h and row["model"] == self.embedder.name:
            return
        vec = self.embedder.embed([text])[0]
        _upsert_embedding(conn, repo, number, self.embedder.name, h, vec)

    @staticmethod
    def _has_prediction(conn, repo, number) -> bool:
        return (
            conn.execute(
                "SELECT 1 FROM predictions WHERE repo = %s AND number = %s AND kind = 'open'",
                (repo, number),
            ).fetchone()
            is not None
        )


def _upsert_embedding(conn, repo, number, model, h, vec) -> None:
    conn.execute(
        """
        INSERT INTO issue_embeddings (repo, number, model, text_hash, embedding)
        VALUES (%s, %s, %s, %s, %s::vector)
        ON CONFLICT (repo, number) DO UPDATE SET model = EXCLUDED.model,
            text_hash = EXCLUDED.text_hash, embedding = EXCLUDED.embedding, updated_at = now()
        """,
        (repo, number, model, h, to_pgvector(vec)),
    )


def backfill(
    conn: Connection,
    repo: str,
    records: Iterable[dict[str, Any]],
    embedder: Embedder,
    batch_size: int = 256,
) -> int:
    """Load historical issues (no predictions) so new issues have earlier ones to match.

    `records` use the benchmark's raw format (`dataset.collect.normalize_issue`). Idempotent:
    re-running updates rows in place and skips unchanged embeddings.
    """
    batch: list[dict[str, Any]] = []
    total = 0
    for rec in records:
        batch.append(rec)
        if len(batch) >= batch_size:
            total += _backfill_batch(conn, repo, batch, embedder)
            batch = []
    if batch:
        total += _backfill_batch(conn, repo, batch, embedder)
    return total


def _backfill_batch(conn, repo, batch, embedder) -> int:
    with conn.transaction():
        for rec in batch:
            updated = rec.get("updated_at") or rec.get("closed_at") or rec["created_at"]
            store_issue(
                conn,
                {
                    "repo": repo,
                    "source": "backfill",
                    "issue": {
                        "number": rec["number"],
                        "title": rec["title"] or "",
                        "body": rec["body"] or "",
                        "state": (rec.get("state") or "open").lower(),
                        "labels": rec.get("labels") or [],
                        "author": rec.get("author"),
                        "url": rec.get("url"),
                        "created_at": rec["created_at"],
                        "updated_at": max(updated, rec["created_at"]),
                    },
                },
            )
        texts, keys = [], []
        for rec in batch:
            text = embedding_text(rec["title"] or "", clean_body(rec["body"] or ""))
            h = text_hash(text)
            row = conn.execute(
                "SELECT text_hash, model FROM issue_embeddings WHERE repo = %s AND number = %s",
                (repo, rec["number"]),
            ).fetchone()
            if row and row["text_hash"] == h and row["model"] == embedder.name:
                continue
            texts.append(text)
            keys.append((rec["number"], h))
        if texts:
            for (number, h), vec in zip(keys, embedder.embed(texts), strict=True):
                _upsert_embedding(conn, repo, number, embedder.name, h, vec)
    return len(batch)


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)
