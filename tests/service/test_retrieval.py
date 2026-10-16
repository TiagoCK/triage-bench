import pytest

pytest.importorskip("fastapi", reason="needs the service extra")
pytest.importorskip("psycopg", reason="needs the service extra")

from datetime import UTC, datetime, timedelta  # noqa: E402

import numpy as np  # noqa: E402

from triage.service.embeddings import to_pgvector  # noqa: E402
from triage.service.retrieval import similar_issues  # noqa: E402

REPO = "acme/widgets"
MODEL = "fake"
T0 = datetime(2026, 1, 1, tzinfo=UTC)


def seed(conn, n=60, seed=0, same_time=()):
    rng = np.random.default_rng(seed)
    vecs = rng.normal(size=(n, 384)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    created = [T0 + timedelta(hours=i) for i in range(n)]
    for i in same_time:  # issue i is created at the same instant as issue i - 1
        created[i] = created[i - 1]
    for i in range(n):
        conn.execute(
            "INSERT INTO issues (repo, number, title, state, created_at, updated_at, source)"
            " VALUES (%s, %s, 't', 'open', %s, %s, 'test')",
            (REPO, i, created[i], created[i]),
        )
        conn.execute(
            "INSERT INTO issue_embeddings (repo, number, model, text_hash, embedding)"
            " VALUES (%s, %s, %s, 'h', %s::vector)",
            (REPO, i, MODEL, to_pgvector(vecs[i])),
        )
    conn.commit()
    return vecs, created


def brute_force(vecs, created, i, k):
    earlier = [j for j in range(len(vecs)) if created[j] < created[i] and j != i]
    sims = {j: float(vecs[j] @ vecs[i]) for j in earlier}
    return sorted(earlier, key=lambda j: (-sims[j], j))[:k]


def test_matches_numpy_brute_force(conn):
    vecs, created = seed(conn)
    for i in (5, 30, 59):
        got = similar_issues(conn, REPO, i, created[i], MODEL, k=5)
        assert [n for n, _ in got] == brute_force(vecs, created, i, 5)
        for n, s in got:
            assert s == pytest.approx(float(vecs[n] @ vecs[i]), abs=1e-5)


def test_never_returns_self_later_or_same_instant_issues(conn):
    vecs, created = seed(conn, same_time=(20,))
    got = [n for n, _ in similar_issues(conn, REPO, 20, created[20], MODEL, k=60)]
    assert 20 not in got and 19 not in got  # 19 was created at the same instant
    assert all(created[n] < created[20] for n in got)
    assert len(got) == 19  # issues 0-18


def test_fewer_than_k_only_when_history_is_short(conn):
    vecs, created = seed(conn)
    assert similar_issues(conn, REPO, 0, created[0], MODEL, k=5) == []
    assert len(similar_issues(conn, REPO, 2, created[2], MODEL, k=5)) == 2
    assert len(similar_issues(conn, REPO, 40, created[40], MODEL, k=5)) == 5


def test_other_repos_and_models_are_ignored(conn):
    vecs, created = seed(conn)
    conn.execute(
        "UPDATE issue_embeddings SET model = 'other' WHERE number = %s",
        (brute_force(vecs, created, 30, 1)[0],),
    )
    conn.commit()
    got = [n for n, _ in similar_issues(conn, REPO, 30, created[30], MODEL, k=5)]
    assert brute_force(vecs, created, 30, 1)[0] not in got
    assert similar_issues(conn, "other/repo", 30, created[30], MODEL, k=5) == []
