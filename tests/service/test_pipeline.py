import pytest

pytest.importorskip("fastapi", reason="needs the service extra")
pytest.importorskip("psycopg", reason="needs the service extra")

import hashlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from triage.service.pipeline import LabelerRegistry, Pipeline, backfill  # noqa: E402
from triage.systems.labels import TfidfLabeler  # noqa: E402

FIXTURE_RAW = Path(__file__).parents[1] / "fixtures" / "raw" / "acme__widgets.jsonl"
REPO = "acme/widgets"


class FakeEmbedder:
    """Bag-of-words hashed into 384 dimensions: similar wording, similar vectors."""

    name = "fake-bow"
    dim = 384

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += len(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                out[i, int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


@pytest.fixture
def records():
    return [json.loads(line) for line in FIXTURE_RAW.open()]


@pytest.fixture
def labeler(fixture_repo):
    _, repo_dir, meta = fixture_repo
    issues = pd.read_parquet(repo_dir / "issues.parquet")
    train = issues[(issues["split"] == "train") & (issues["target_labels"].map(len) > 0)]
    lab = TfidfLabeler()
    lab.fit(train, meta["label_vocab"])
    return lab


def make_pipeline(settings, labeler=None):
    registry = LabelerRegistry(Path("/nonexistent"))
    if labeler is not None:
        registry.set(REPO, labeler, "test-v1")
    return Pipeline(settings, FakeEmbedder(), registry)


def event(rec, action="opened", **changes):
    issue = {
        "number": rec["number"],
        "title": rec["title"],
        "body": rec["body"],
        "state": "open",
        "labels": rec["labels"],
        "author": rec["author"],
        "url": rec["url"],
        "created_at": rec["created_at"],
        "updated_at": rec["created_at"],
    }
    issue.update(changes)
    return {"repo": REPO, "action": action, "source": "webhook", "issue": issue}


def handle(conn, pipeline, payload):
    with conn.transaction():
        pipeline.handle_issue(conn, payload)


def count(conn, table):
    return conn.execute(f"SELECT count(*) AS n FROM {table}").fetchone()["n"]


def test_backfill_adds_history_without_predictions_and_is_idempotent(conn, settings, records):
    emb = FakeEmbedder()
    assert backfill(conn, REPO, records[:100], emb, batch_size=30) == 100
    assert count(conn, "issues") == 100 and count(conn, "issue_embeddings") == 100
    assert count(conn, "predictions") == 0
    calls = emb.calls
    backfill(conn, REPO, records[:100], emb)
    assert emb.calls == calls  # unchanged text is never re-embedded
    assert count(conn, "issues") == 100


def test_opened_issue_gets_one_prediction(conn, settings, records, labeler):
    backfill(conn, REPO, records[:150], FakeEmbedder())
    pipeline = make_pipeline(settings, labeler)
    new = records[150]
    handle(conn, pipeline, event(new))
    pred = conn.execute("SELECT * FROM predictions").fetchone()
    assert pred["number"] == new["number"] and pred["mode"] == "shadow"
    assert len(pred["candidates"]) == 5
    earlier = {r["number"] for r in records[:150] if r["created_at"] < new["created_at"]}
    assert {c["number"] for c in pred["candidates"]} <= earlier
    sims = [c["similarity"] for c in pred["candidates"]]
    assert sims == sorted(sims, reverse=True)
    assert pred["labels"] and set(pred["labels"]) <= set(pred["label_scores"])
    assert pred["labeler_version"] == "test-v1" and pred["embed_model"] == "fake-bow"
    assert set(pred["timings_ms"]) == {"embed", "retrieve", "label"}

    # Repeat delivery or retry: still one prediction.
    handle(conn, pipeline, event(new))
    assert count(conn, "predictions") == 1


def test_edits_reembed_but_label_changes_do_not(conn, settings, records):
    backfill(conn, REPO, records[:20], FakeEmbedder())
    pipeline = make_pipeline(settings)
    new = records[20]
    handle(conn, pipeline, event(new))
    first = conn.execute(
        "SELECT text_hash, updated_at FROM issue_embeddings WHERE number = %s", (new["number"],)
    ).fetchone()

    handle(
        conn,
        pipeline,
        event(
            new, action="labeled", labels=["bug", "area: cli"], updated_at="2030-01-01T00:00:00Z"
        ),
    )
    same = conn.execute(
        "SELECT text_hash, updated_at FROM issue_embeddings WHERE number = %s", (new["number"],)
    ).fetchone()
    assert same == first

    handle(
        conn,
        pipeline,
        event(new, action="edited", body="Completely new text", updated_at="2030-01-02T00:00:00Z"),
    )
    edited = conn.execute(
        "SELECT text_hash FROM issue_embeddings WHERE number = %s", (new["number"],)
    ).fetchone()
    assert edited["text_hash"] != first["text_hash"]
    assert count(conn, "predictions") == 1  # edits never add a prediction


def test_stale_event_does_not_change_the_embedding(conn, settings, records):
    pipeline = make_pipeline(settings)
    rec = records[0]
    handle(
        conn,
        pipeline,
        event(rec, action="edited", body="newest", updated_at="2030-01-02T00:00:00Z"),
    )
    newest = conn.execute("SELECT text_hash FROM issue_embeddings").fetchone()["text_hash"]
    handle(
        conn, pipeline, event(rec, action="edited", body="older", updated_at="2030-01-01T00:00:00Z")
    )
    assert conn.execute("SELECT text_hash FROM issue_embeddings").fetchone()["text_hash"] == newest


def test_without_a_labeler_prediction_is_retrieval_only(conn, settings, records):
    backfill(conn, REPO, records[:30], FakeEmbedder())
    handle(conn, make_pipeline(settings, labeler=None), event(records[30]))
    pred = conn.execute("SELECT candidates, labels, labeler_version FROM predictions").fetchone()
    assert pred["candidates"] and pred["labels"] is None and pred["labeler_version"] is None


def test_labeler_registry_reads_latest_export(tmp_path, labeler):
    target = tmp_path / "acme__widgets" / "20261016-abc"
    labeler.save(target)
    (tmp_path / "acme__widgets" / "LATEST").write_text("20261016-abc\n")
    loaded, version = LabelerRegistry(tmp_path).get(REPO)
    assert version == "20261016-abc"
    assert LabelerRegistry(tmp_path).get("other/repo") is None
