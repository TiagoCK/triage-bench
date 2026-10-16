"""Similar-issue search in Postgres.

Exact cosine search (no ANN index), restricted to issues created strictly before the new
one: the benchmark's time rule. Exact search keeps recall at 100%; a filtered HNSW index can
quietly return fewer than k rows. At up to about 50k issues per repo a scan is fast enough.
"""

from __future__ import annotations

from datetime import datetime

from psycopg import Connection


def similar_issues(
    conn: Connection, repo: str, number: int, created_at: datetime, model: str, k: int
) -> list[tuple[int, float]]:
    """[(issue number, cosine similarity)], best first, among earlier issues of the repo."""
    rows = conn.execute(
        """
        WITH q AS (
            SELECT embedding FROM issue_embeddings WHERE repo = %(repo)s AND number = %(number)s
        )
        SELECT e.number, 1 - (e.embedding <=> q.embedding) AS similarity
        FROM issue_embeddings e
        JOIN issues i ON i.repo = e.repo AND i.number = e.number
        CROSS JOIN q
        WHERE e.repo = %(repo)s AND e.model = %(model)s
          AND i.created_at < %(created_at)s AND e.number <> %(number)s
        ORDER BY e.embedding <=> q.embedding, e.number
        LIMIT %(k)s
        """,
        {"repo": repo, "number": number, "created_at": created_at, "model": model, "k": k},
    ).fetchall()
    return [(r["number"], float(r["similarity"])) for r in rows]
