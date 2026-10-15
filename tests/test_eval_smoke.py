from triage.eval.report import render
from triage.eval.run import EvalOptions, run_repo


def test_end_to_end_non_llm(fixture_repo):
    _, repo_dir, _ = fixture_repo
    rows = run_repo(
        repo_dir,
        EvalOptions(systems=["tfidf", "difflib", "bm25"], sample=40, min_test_duplicates=5),
    )
    by = {(r["task"], r["subset"], r["system"]): r for r in rows}
    assert ("labels", "full", "tfidf-lr") in by
    assert ("duplicates", "full", "bm25") in by
    assert ("duplicates", "sample40-seed0", "difflib") in by
    for r in rows:
        for k, v in r.items():
            if k.endswith("_ci"):
                assert 0.0 <= v[0] <= v[1] <= 1.0, (k, v)
            elif k.startswith(("micro", "macro", "recall", "mrr", "average")):
                assert 0.0 <= v <= 1.0, (k, v)
    # The synthetic duplicates are rephrasings, so lexical retrieval should find most.
    assert by[("duplicates", "full", "bm25")]["recall@5"] > 0.5
    lo, hi = by[("duplicates", "full", "bm25")]["recall@5_ci"]
    assert lo <= by[("duplicates", "full", "bm25")]["recall@5"] <= hi
    assert "_ci" in "".join(by[("labels", "full", "tfidf-lr")])
    assert "## duplicates: acme/widgets (full" in render(rows)


def test_llm_systems_offline_with_fake_model(fixture_repo, monkeypatch, tmp_path):
    """Exercise the LLM code paths with a stub that returns the gold answer."""
    import pandas as pd

    from triage.config import ModelConfig
    from triage.eval import run as run_mod
    from triage.llm import CallRecord
    from triage.systems import duplicates, labels

    _, repo_dir, _ = fixture_repo
    issues = pd.read_parquet(repo_dir / "issues.parquet").set_index("number")
    record = CallRecord("fake", 10, 2, 0.001, 0.5, cached=False)

    def fake_complete(model, system, user, max_tokens=512):
        if "find duplicate issues" in system:
            number = int(user.split("New issue #")[1].split(":")[0])
            canon = issues.loc[number, "dup_of"]
            return {"duplicate_of": None if pd.isna(canon) else int(canon), "confidence": 1}, record
        new_issue = user.rsplit("New issue:\n\n", 1)[1]
        match = issues[issues["text"].str.startswith(new_issue[:200])]
        return {"labels": list(match["target_labels"].iloc[0])}, record

    monkeypatch.setattr(labels, "complete_json", fake_complete)
    monkeypatch.setattr(duplicates, "complete_json", fake_complete)
    monkeypatch.setattr(run_mod, "_model_available", lambda m: True)
    monkeypatch.setattr(
        run_mod.importlib.util, "find_spec", lambda name: object() if name == "litellm" else None
    )

    fake = ModelConfig(key="fake", id="fake/model", concurrency=2)
    rows = run_mod.run_repo(
        repo_dir,
        EvalOptions(
            systems=["llm-zeroshot", "llm-dups"],
            models=[fake],
            sample=40,
            min_test_duplicates=5,
            with_validation=True,
            predictions_dir=tmp_path,
        ),
    )
    by = {(r["task"], r["system"], r["model"]): r for r in rows}
    lab = by[("labels", "llm-zeroshot", "fake")]
    assert lab["micro_f1"] == 1.0
    assert lab["usd_per_1k"] == 1.0
    assert ("labels", "tfidf-lr", "-") in by  # baseline added on the same sample
    dup = by[("duplicates", "bm25+llm", "fake")]
    # The stub flags exactly the true duplicates that bm25 surfaced, with the right target.
    assert dup["false_dup_rate"] == 0.0
    assert dup["precision@base_rate"] == 1.0
    assert ("duplicates", "bm25", "-") in by  # first stage reported on the same sample

    # Per-item predictions for test and validation; validation is never in the table.
    import json

    files = {f.name: json.loads(f.read_text()) for f in (tmp_path / "acme__widgets").iterdir()}
    test = files["bm25+llm__fake__test.json"]
    val = files["bm25+llm__fake__validation.json"]
    assert not any("validation" in r["subset"] for r in rows)
    item = test["items"][0]
    assert {"number", "is_dup", "chosen", "correct", "confidence", "sim_chosen"} <= set(item)
    test_numbers = {i["number"] for i in test["items"]}
    val_numbers = {i["number"] for i in val["items"]}
    assert not test_numbers & val_numbers
    assert issues.loc[sorted(val_numbers), "is_validation"].all()
    assert 0 < val["base_rate"] < 1


def test_duplicate_task_skipped_below_minimum(fixture_repo):
    _, repo_dir, meta = fixture_repo
    too_many = meta["n_duplicates"]["test"] + 1
    rows = run_repo(
        repo_dir,
        EvalOptions(
            systems=["tfidf", "bm25"], sample=40, min_test_duplicates=too_many, bootstrap=0
        ),
    )
    assert {r["task"] for r in rows} == {"labels"}


def test_label_task_skipped_below_minimum(fixture_repo):
    _, repo_dir, meta = fixture_repo
    too_many = meta["n_labeled"]["test"] + 1
    rows = run_repo(
        repo_dir,
        EvalOptions(
            systems=["tfidf", "bm25"],
            sample=40,
            min_test_duplicates=5,
            min_test_labeled=too_many,
            bootstrap=0,
        ),
    )
    assert {r["task"] for r in rows} == {"duplicates"}
