import pytest

from triage.eval import metrics


def test_multilabel_perfect_and_partial():
    vocab = ["a", "b", "c"]
    perfect = metrics.multilabel_scores([["a"], ["b", "c"]], [["a"], ["b", "c"]], vocab)
    assert perfect["micro_f1"] == 1.0 and perfect["exact_match"] == 1.0

    s = metrics.multilabel_scores([["a"], ["b"]], [["a", "b"], ["c"]], vocab)
    # tp=1 (a), fp=2 (b in #1, c in #2), fn=1 (b in #2)
    assert s["micro_precision"] == pytest.approx(1 / 3)
    assert s["micro_recall"] == pytest.approx(1 / 2)
    assert s["exact_match"] == 0.0


def test_multilabel_ignores_out_of_vocab_predictions():
    s = metrics.multilabel_scores([["a"]], [["a", "not-a-label"]], ["a"])
    assert s["micro_precision"] == 1.0


def test_retrieval_recall_and_mrr():
    ranked = [[5, 1, 2], [9, 8, 7], [3]]
    gold = [{1}, {4}, {3}]
    s = metrics.retrieval_scores(ranked, gold, ks=(1, 2))
    assert s["recall@1"] == pytest.approx(1 / 3)
    assert s["recall@2"] == pytest.approx(2 / 3)
    assert s["mrr@10"] == pytest.approx((1 / 2 + 0 + 1) / 3)


def test_detection_wrong_target_is_not_a_true_positive():
    # Two positives with the highest scores, but the top one points at the wrong issue.
    s = metrics.detection_scores(
        is_dup=[True, True, False, False],
        scores=[0.9, 0.8, 0.2, 0.1],
        correct_target=[False, True, False, False],
    )
    assert s["average_precision"] == pytest.approx((1 / 2) / 2)
    assert s["recall@p90"] == 0.0


def test_detection_perfect_ranking():
    s = metrics.detection_scores([True, False, True], [0.9, 0.1, 0.8])
    assert s["average_precision"] == 1.0
    assert s["recall@p90"] == 1.0


def test_binary_decisions():
    s = metrics.binary_decision_scores(
        is_dup=[True, True, False, False],
        flagged=[True, False, True, False],
        correct_target=[True, False, False, False],
    )
    assert s["precision"] == 0.5
    assert s["recall"] == 0.5
    assert s["false_dup_rate"] == 0.5


def test_cost_unknown_if_any_call_unpriced():
    assert metrics.latency_cost_summary([1.0, 2.0], [0.001, None])["usd_per_1k"] is None
    assert metrics.latency_cost_summary([1.0], [0.002])["usd_per_1k"] == pytest.approx(2.0)


def test_precision_at_base_rate_reweights_balanced_sample():
    # Balanced sample: 2 dups (both flagged, both right), 2 non-dups (1 falsely flagged).
    is_dup, flagged, correct = (
        [True, True, False, False],
        [True, True, True, False],
        [True, True, False, False],
    )
    # Raw precision is 2/3, but at a 10% duplicate rate false flags dominate:
    # 1.0 * 0.1 / (1.0 * 0.1 + 0.5 * 0.9)
    p = metrics.precision_at_base_rate(is_dup, flagged, correct, base_rate=0.1)
    assert p == pytest.approx(0.1 / 0.55)


def test_bootstrap_ci_brackets_estimate_and_is_reproducible():
    import numpy as np

    hits = np.array([1] * 21 + [0] * 15, dtype=float)  # 21 of 36, like uv R@5

    def score(idx):
        return {"recall@5": float(hits[idx].mean())}

    ci = metrics.bootstrap_ci(score, len(hits), ["recall@5"], n_boot=2000, seed=0)["recall@5_ci"]
    assert ci[0] < 21 / 36 < ci[1]
    assert 0.25 < ci[1] - ci[0] < 0.40  # about +/- 0.16 for n=36
    again = metrics.bootstrap_ci(score, len(hits), ["recall@5"], n_boot=2000, seed=0)
    assert again["recall@5_ci"] == ci


def test_bootstrap_strata_keep_group_sizes():
    import numpy as np

    y = np.array([True] * 3 + [False] * 7)
    seen = []

    def score(idx):
        seen.append(int(y[idx].sum()))
        return {"x": 0.0}

    metrics.bootstrap_ci(score, len(y), ["x"], strata=y, n_boot=50)
    assert set(seen) == {3}
    assert metrics.bootstrap_ci(score, len(y), ["x"], n_boot=0) == {}


def test_detection_is_invariant_to_input_order_with_ties():
    # One duplicate and three non-duplicates all share the same score.
    a = metrics.detection_scores([True, False, False, False], [0.5, 0.5, 0.5, 0.5])
    b = metrics.detection_scores([False, False, False, True], [0.5, 0.5, 0.5, 0.5])
    assert a == b
    assert a["average_precision"] == pytest.approx(0.25)
