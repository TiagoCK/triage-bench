import json

import numpy as np
import pytest

from triage.eval import gates, metrics
from triage.eval.report import render

REPO = "acme/widgets"
SYSTEM = "hybrid+llm"


def item(number, is_dup, chosen, correct, conf, sim=0.5):
    return {
        "number": number,
        "is_dup": is_dup,
        "chosen": chosen,
        "correct": correct,
        "confidence": conf,
        "candidates": [],
        "sim_chosen": sim if chosen is not None else None,
        "sim_top1": sim,
        "sim_top2": sim - 0.1,
        "gold": [],
    }


def payload(items, base_rate):
    return {"repo": REPO, "system": SYSTEM, "split": "x", "base_rate": base_rate, "items": items}


def judge_items(seed, n=40):
    """Duplicates the judge gets right at high confidence; false flags at low confidence."""
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n):  # duplicates: right when confident
        conf = float(rng.uniform(0.5, 1.0))
        items.append(item(i, True, 1000 + i, conf > 0.7, conf, sim=conf))
    for i in range(n, 2 * n):  # non-duplicates: flagged at low confidence, or not at all
        conf = float(rng.uniform(0.0, 0.6))
        chosen = 2000 + i if conf > 0.3 else None
        items.append(item(i, False, chosen, False, conf if chosen else 0.0, sim=conf))
    return items


def test_base_rate_weights():
    w = metrics.base_rate_weights([True, True, False, False], base_rate=0.1)
    assert w.tolist() == pytest.approx([0.2, 0.2, 1.8, 1.8])


def test_calibration_perfect_and_overconfident():
    conf = [0.9] * 10
    is_dup = [True] * 9 + [False]
    correct = [True] * 9 + [False]
    flagged = [True] * 10
    perfect = metrics.calibration(conf, flagged, correct, is_dup, base_rate=0.9)
    assert perfect["ece"] == pytest.approx(0.0)
    # Same calls, but duplicates are rare in reality: the false flag dominates.
    rare = metrics.calibration(conf, flagged, correct, is_dup, base_rate=0.1)
    assert rare["ece"] > 0.5
    top = rare["bins"][-1]
    assert top["n"] == 10 and top["accuracy"] < 0.5


def test_select_threshold_prefers_coverage_and_respects_minimum():
    curve = [
        {
            "threshold": 0.95,
            "precision@base_rate": 1.0,
            "coverage": 0.1,
            "n_correct": 3,
            "n_flags": 3,
        },
        {
            "threshold": 0.9,
            "precision@base_rate": 0.92,
            "coverage": 0.3,
            "n_correct": 9,
            "n_flags": 10,
        },
        {
            "threshold": 0.8,
            "precision@base_rate": 0.85,
            "coverage": 0.5,
            "n_correct": 15,
            "n_flags": 18,
        },
        {
            "threshold": 0.5,
            "precision@base_rate": 0.4,
            "coverage": 0.9,
            "n_correct": 27,
            "n_flags": 60,
        },
    ]
    assert gates.select_threshold(curve, 0.9) == 0.9  # 0.95 has too few correct calls
    assert gates.select_threshold(curve, 0.8) == 0.8
    assert gates.select_threshold(curve, 0.99) is None


def test_ungated_items_never_pass():
    split = gates.Split(payload([item(1, False, None, False, 0.99)], 0.1), [])
    assert gates.gate_scores(split, "llm-confidence")[0] == -np.inf


def test_agreement_requires_same_pick():
    mine = payload([item(1, True, 7, True, 0.9), item(2, True, 8, False, 0.9)], 0.5)
    other = payload([item(1, True, 7, True, 0.8), item(2, True, 9, False, 0.8)], 0.5)
    assert gates.Split(mine, [other]).agreement.tolist() == [1.0, 0.0]


def _write(tmp_path, model, split, items, base_rate):
    path = gates.predictions_path(tmp_path, REPO, SYSTEM, model, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload(items, base_rate)))


def test_analyze_uses_validation_thresholds_only(tmp_path):
    for model, seed in (("a", 1), ("b", 2)):
        _write(tmp_path, model, "validation", judge_items(seed), 0.3)
        _write(tmp_path, model, "test", judge_items(seed + 10), 0.3)
    rows = gates.analyze(tmp_path, REPO, ["a", "b"], [0.8], n_boot=50)

    val = gates.Split(
        json.loads(gates.predictions_path(tmp_path, REPO, SYSTEM, "a", "validation").read_text()),
        [],
    )
    expected = gates.select_threshold(
        gates.sweep(val, gates.gate_scores(val, "llm-confidence")), 0.8
    )
    conf_row = next(r for r in rows if r["system"] == "gate:llm-confidence" and r["model"] == "a")
    assert conf_row["threshold"] == expected is not None
    assert conf_row["precision@base_rate"] >= 0.8  # the synthetic signal is clean
    assert {r["system"] for r in rows} >= {
        "gate:llm-confidence",
        "gate:similarity",
        "gate:agreement",
        "gate:learned",
    }
    assert sum(r["task"] == "calibration" for r in rows) == 2

    table = render(rows)
    assert "duplicates-gated: acme/widgets (target P@base>=0.8" in table
    assert "calibration: acme/widgets" in table


def test_unreachable_target_is_reported_not_faked(tmp_path):
    bad = [item(i, i % 2 == 0, 100 + i, False, 0.9) for i in range(20)]  # always wrong
    _write(tmp_path, "a", "validation", bad, 0.1)
    _write(tmp_path, "a", "test", bad, 0.1)
    rows = gates.analyze(tmp_path, REPO, ["a"], [0.5], n_boot=10)
    conf_row = next(r for r in rows if r["system"] == "gate:llm-confidence")
    assert conf_row["threshold"] is None
    assert "not reachable on validation" in render(rows)
