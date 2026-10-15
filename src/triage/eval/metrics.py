"""Metrics for the two triage tasks. Pure functions over plain Python data, no models."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence

import numpy as np


def multilabel_scores(
    gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]], vocab: Sequence[str]
) -> dict[str, float]:
    """Micro/macro precision, recall and F1 over a fixed label vocabulary.

    Macro averages only over labels that appear in gold or predictions, so a label that is
    absent from the test window does not count as a free 0 or 1.
    """
    if len(gold) != len(pred):
        raise ValueError("gold and pred must have the same length")
    idx = {lbl: i for i, lbl in enumerate(vocab)}
    tp = np.zeros(len(vocab))
    fp = np.zeros(len(vocab))
    fn = np.zeros(len(vocab))
    exact = 0
    for g, p in zip(gold, pred, strict=True):
        gs = {x for x in g if x in idx}
        ps = {x for x in p if x in idx}
        exact += gs == ps
        for lbl in gs & ps:
            tp[idx[lbl]] += 1
        for lbl in ps - gs:
            fp[idx[lbl]] += 1
        for lbl in gs - ps:
            fn[idx[lbl]] += 1

    micro_p = _safe_div(tp.sum(), tp.sum() + fp.sum())
    micro_r = _safe_div(tp.sum(), tp.sum() + fn.sum())
    per_label_f1 = [
        _f1(_safe_div(tp[i], tp[i] + fp[i]), _safe_div(tp[i], tp[i] + fn[i]))
        for i in range(len(vocab))
        if tp[i] + fp[i] + fn[i] > 0
    ]
    return {
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": _f1(micro_p, micro_r),
        "macro_f1": float(np.mean(per_label_f1)) if per_label_f1 else 0.0,
        "exact_match": _safe_div(exact, len(gold)),
        "n": float(len(gold)),
    }


def retrieval_scores(
    ranked: Sequence[Sequence[int]], gold: Sequence[set[int]], ks: Sequence[int] = (1, 5, 10)
) -> dict[str, float]:
    """Recall@k (any gold item in the top k) and MRR@10 for duplicate retrieval."""
    if len(ranked) != len(gold):
        raise ValueError("ranked and gold must have the same length")
    out: dict[str, float] = {}
    for k in ks:
        hits = [bool(set(r[:k]) & g) for r, g in zip(ranked, gold, strict=True)]
        out[f"recall@{k}"] = _safe_div(sum(hits), len(hits))
    rr = []
    for r, g in zip(ranked, gold, strict=True):
        rank = next((i + 1 for i, x in enumerate(r[:10]) if x in g), None)
        rr.append(1.0 / rank if rank else 0.0)
    out["mrr@10"] = float(np.mean(rr)) if rr else 0.0
    out["n"] = float(len(ranked))
    return out


def detection_scores(
    is_dup: Sequence[bool], scores: Sequence[float], correct_target: Sequence[bool] | None = None
) -> dict[str, float]:
    """Precision-oriented metrics for "should the bot post a duplicate comment?".

    `scores` is the system's confidence that the issue is a duplicate. `correct_target`
    marks positives where the top suggestion was actually right; a confident suggestion that
    points at the wrong issue is a false positive from the maintainer's point of view.

    Reports average precision (threshold-free) and recall at 90% precision: the share of
    duplicates the bot can flag while keeping 9 in 10 of its comments correct.
    """
    y = np.asarray(is_dup, dtype=bool)
    s = np.asarray(scores, dtype=float)
    ok = y if correct_target is None else (y & np.asarray(correct_target, dtype=bool))
    n_pos = int(y.sum())
    if n_pos == 0 or len(s) == 0:
        return {
            "average_precision": 0.0,
            "recall@p90": 0.0,
            "n_pos": float(n_pos),
            "n_neg": float((~y).sum()),
        }

    # One threshold per distinct score, so tied scores are all in or all out. Without this,
    # the input order decides ties (e.g. many identical difflib scores) and biases the result.
    order = np.argsort(-s, kind="stable")
    s_sorted, ok_sorted = s[order], ok[order]
    tp_cum = np.cumsum(ok_sorted)
    block_ends = np.r_[np.flatnonzero(np.diff(s_sorted)), len(s_sorted) - 1]
    tp = tp_cum[block_ends]
    precision = tp / (block_ends + 1)
    recall = tp / n_pos

    ap = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    meets = precision >= 0.9
    r_at_p90 = float(recall[meets].max()) if meets.any() else 0.0
    return {
        "average_precision": ap,
        "recall@p90": r_at_p90,
        "n_pos": float(n_pos),
        "n_neg": float((~y).sum()),
    }


def binary_decision_scores(
    is_dup: Sequence[bool], flagged: Sequence[bool], correct_target: Sequence[bool]
) -> dict[str, float]:
    """For systems that make a yes/no call (LLM confirmation) instead of a score."""
    y = np.asarray(is_dup, dtype=bool)
    f = np.asarray(flagged, dtype=bool)
    c = np.asarray(correct_target, dtype=bool)
    tp = int((f & y & c).sum())
    flagged_n = int(f.sum())
    precision = _safe_div(tp, flagged_n)
    recall = _safe_div(tp, int(y.sum()))
    return {
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "false_dup_rate": _safe_div(int((f & ~y).sum()), int((~y).sum())),
    }


def precision_at_base_rate(
    is_dup: Sequence[bool],
    flagged: Sequence[bool],
    correct_target: Sequence[bool],
    base_rate: float,
) -> float:
    """Precision a bot would see at the repo's natural duplicate rate.

    LLM samples are balanced (half duplicates), which inflates raw precision. This re-weights
    the per-class rates to `base_rate`. A comment counts as correct only when it flags a real
    duplicate *and* links an acceptable original.
    """
    y = np.asarray(is_dup, dtype=bool)
    f = np.asarray(flagged, dtype=bool)
    c = np.asarray(correct_target, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    pos_flag_rate = _safe_div(int((f & y).sum()), n_pos)
    correct_rate = _safe_div(int((f & y & c).sum()), n_pos)
    false_rate = _safe_div(int((f & ~y).sum()), n_neg)
    denom = pos_flag_rate * base_rate + false_rate * (1 - base_rate)
    return correct_rate * base_rate / denom if denom else 0.0


def base_rate_weights(is_dup: Sequence[bool], base_rate: float) -> np.ndarray:
    """Per-item weights that turn a balanced sample into one at the natural duplicate rate."""
    y = np.asarray(is_dup, dtype=bool)
    sample_rate = y.mean() if len(y) else 0.0
    if sample_rate in (0.0, 1.0):
        return np.ones(len(y))
    return np.where(y, base_rate / sample_rate, (1 - base_rate) / (1 - sample_rate))


CALIBRATION_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def calibration(
    confidence: Sequence[float],
    flagged: Sequence[bool],
    correct: Sequence[bool],
    is_dup: Sequence[bool],
    base_rate: float,
    bins: Sequence[float] = CALIBRATION_BINS,
) -> dict:
    """Does "0.9 confident" mean right 90% of the time, at the repo's real duplicate rate?

    Over flagged items only, grouped by stated confidence: the base-rate-weighted share that
    were correct, against the mean stated confidence. ECE is the weighted mean gap.
    """
    conf = np.asarray(confidence, dtype=float)
    f = np.asarray(flagged, dtype=bool)
    c = np.asarray(correct, dtype=bool)
    w = base_rate_weights(is_dup, base_rate)
    out_bins, gaps, total = [], 0.0, float(w[f].sum())
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:], strict=True)):
        upper = conf <= hi if i == len(bins) - 2 else conf < hi
        m = f & (conf >= lo) & upper
        weight = float(w[m].sum())
        accuracy = float((w[m] * c[m]).sum() / weight) if weight else None
        mean_conf = float(conf[m].mean()) if m.any() else None
        out_bins.append(
            {
                "lo": lo,
                "hi": hi,
                "n": int(m.sum()),
                "mean_confidence": mean_conf,
                "accuracy": accuracy,
            }
        )
        if weight:
            gaps += weight * abs(accuracy - mean_conf)
    return {"ece": gaps / total if total else None, "bins": out_bins}


def bootstrap_ci(
    score_fn: Callable[[np.ndarray], dict[str, float]],
    n: int,
    keys: Sequence[str],
    strata: Sequence | None = None,
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, list[float]]:
    """Percentile bootstrap confidence intervals over test items.

    `score_fn` takes an array of item indices (with repeats) and returns metrics. Items are
    resampled with replacement `n_boot` times; with `strata`, each stratum is resampled
    separately so its size stays fixed (used to keep the duplicate / non-duplicate mix
    constant). Returns {"<metric>_ci": [low, high]} for each requested key.
    """
    if n == 0 or n_boot <= 0:
        return {}
    rng = np.random.default_rng(seed)
    if strata is None:
        groups = [np.arange(n)]
    else:
        labels = np.asarray(strata)
        groups = [np.flatnonzero(labels == s) for s in np.unique(labels)]
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        scores = score_fn(idx)
        for k in keys:
            if scores.get(k) is not None:
                draws[k].append(scores[k])
    tail = (1 - level) / 2
    return {
        f"{k}_ci": [float(np.quantile(v, tail)), float(np.quantile(v, 1 - tail))]
        for k, v in draws.items()
    }


def latency_cost_summary(latencies: Sequence[float], costs: Sequence[float | None]) -> dict:
    lat = np.asarray(latencies, dtype=float)
    known = [c for c in costs if c is not None]
    return {
        "p50_latency_s": float(np.percentile(lat, 50)) if len(lat) else 0.0,
        "p95_latency_s": float(np.percentile(lat, 95)) if len(lat) else 0.0,
        # Unknown if any call could not be priced: a partial average would understate cost.
        "usd_per_1k": 1000 * float(np.mean(known)) if known and len(known) == len(costs) else None,
    }


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else 0.0
