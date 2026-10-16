"""Golden tests for logic shared by the benchmark and the live service. If one of these
changes, cached LLM results and the service's predictions change with it."""

import pandas as pd
import pytest

from triage.systems.duplicates import (
    IssueDoc,
    build_judge_prompt,
    embedding_text,
    parse_judge_reply,
)
from triage.systems.labels import TfidfLabeler


def test_embedding_text_format():
    assert embedding_text("Title", "b" * 2000) == "Title\n\n" + "b" * 1500


def test_judge_prompt_golden():
    new = IssueDoc(10, "Crash on start", "It crashes.")
    cands = [IssueDoc(3, "Crash at launch", "x" * 900), IssueDoc(4, "Slow", "")]
    system, user = build_judge_prompt("acme/widgets", new, cands)
    assert system.startswith("You help maintainers of the acme/widgets GitHub repository")
    assert user == (
        "New issue #10:\nTitle: Crash on start\nIt crashes.\n\n"
        "Candidates:\n"
        "\n#3:\nTitle: Crash at launch\n" + "x" * 800 + " ...\n"
        "\n#4:\nTitle: Slow\n"
    )


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ({"duplicate_of": 3, "confidence": 0.8}, (3, 0.8)),
        ({"duplicate_of": "4", "confidence": "0.5"}, (4, 0.5)),
        ({"duplicate_of": 99, "confidence": 0.9}, (None, 0.0)),  # not a shown candidate
        ({"duplicate_of": None, "confidence": 0.9}, (None, 0.0)),
        ({"duplicate_of": "x"}, (None, 0.0)),
    ],
)
def test_parse_judge_reply(reply, expected):
    assert parse_judge_reply(reply, [3, 4]) == expected


def test_labeler_save_load_round_trip(tmp_path, fixture_repo):
    _, repo_dir, meta = fixture_repo
    issues = pd.read_parquet(repo_dir / "issues.parquet")
    labeled = issues[issues["target_labels"].map(len) > 0]
    train, test = labeled[labeled["split"] == "train"], labeled[labeled["split"] == "test"]
    labeler = TfidfLabeler()
    labeler.fit(train, meta["label_vocab"])
    labeler.save(tmp_path / "v1")
    loaded = TfidfLabeler.load(tmp_path / "v1")
    assert loaded.predict(test) == labeler.predict(test)
    scores = loaded.predict_scores(list(test["text"][:3]))
    assert set(scores[0]) == {str(c) for c in labeler.classes}
    assert all(0.0 <= p <= 1.0 for p in scores[0].values())
