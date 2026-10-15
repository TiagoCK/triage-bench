"""Confidence gates for the LLM duplicate judge.

A gate lets the bot post a "duplicate of #N" comment only when a signal clears a threshold.
Thresholds are chosen on the validation sample (end of the train period) for a target
precision, then applied once to the test sample, so nothing is tuned on test.

Signals:
  llm-confidence  the judge's own stated confidence
  similarity      embedding cosine between the new issue and the chosen original
  agreement       every other model picked the same original (1) or not (0)
  learned         logistic regression on the signals above plus the gap between the two most
                  similar candidates, fit on validation

All metrics are re-weighted to the repo's natural duplicate rate, because the samples are
balanced and raw precision would overstate what maintainers see.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from triage.eval import metrics

SIGNALS = ("llm-confidence", "similarity", "agreement", "learned")
GATE_CI_KEYS = ("precision@base_rate", "recall", "false_dup_rate")
# A threshold must catch at least this many duplicates on validation, so it is not chosen
# because of one or two lucky calls.
MIN_CORRECT_ON_VALIDATION = 5


def predictions_path(predictions_dir: Path, repo: str, system: str, model: str, split: str):
    return predictions_dir / repo.replace("/", "__") / f"{system}__{model}__{split}.json"


def load_predictions(predictions_dir: Path, repo: str, system: str, model: str, split: str):
    path = predictions_path(predictions_dir, repo, system, model, split)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `triage eval --repo {repo} --systems llm-dups "
            f"--models {model} --with-validation` first"
        )
    return json.loads(path.read_text())


def find_judge_system(predictions_dir: Path, repo: str) -> str:
    """The judge system name, when the repo has predictions from exactly one."""
    names = {
        f.name.split("__")[0] for f in (predictions_dir / repo.replace("/", "__")).glob("*.json")
    }
    if len(names) != 1:
        raise ValueError(f"expected predictions from one judge system for {repo}, found {names}")
    return names.pop()


class Split:
    """One model's predictions on one split, as arrays aligned by item."""

    def __init__(self, payload: dict, others: list[dict]) -> None:
        items = payload["items"]
        self.base_rate = float(payload["base_rate"])
        self.numbers = [i["number"] for i in items]
        self.is_dup = np.array([i["is_dup"] for i in items], dtype=bool)
        self.chosen = np.array([i["chosen"] is not None for i in items], dtype=bool)
        self.correct = np.array([i["correct"] for i in items], dtype=bool)
        self.confidence = np.array([float(i["confidence"] or 0.0) for i in items])
        self.sim_chosen = np.array([_num(i["sim_chosen"]) for i in items])
        self.sim_gap = np.array([_num(i["sim_top1"]) - _num(i["sim_top2"]) for i in items])
        self.agreement = self._agreement(items, others)

    @staticmethod
    def _agreement(items: list[dict], others: list[dict]) -> np.ndarray:
        if not others:
            return np.zeros(len(items))
        picks = [{i["number"]: i["chosen"] for i in o["items"]} for o in others]
        return np.array(
            [
                float(
                    i["chosen"] is not None
                    and all(p.get(i["number"]) == i["chosen"] for p in picks)
                )
                for i in items
            ]
        )

    def features(self) -> np.ndarray:
        return np.column_stack([self.confidence, self.sim_chosen, self.sim_gap, self.agreement])


def _num(v) -> float:
    return float(v) if v is not None else 0.0


def gate_scores(split: Split, signal: str, learned: LogisticRegression | None = None):
    """Higher means more willing to post. Items the judge did not flag can never pass."""
    if signal == "llm-confidence":
        raw = split.confidence
    elif signal == "similarity":
        raw = split.sim_chosen
    elif signal == "agreement":
        raw = split.agreement
    elif signal == "learned":
        raw = learned.predict_proba(split.features())[:, 1]
    else:
        raise ValueError(f"unknown signal {signal!r}")
    return np.where(split.chosen, raw, -np.inf)


def fit_learned_gate(val: Split) -> LogisticRegression | None:
    """P(correct | flagged) from validation; None if validation lacks both outcomes."""
    x, y = val.features()[val.chosen], val.correct[val.chosen]
    if len(set(y)) < 2:
        return None
    return LogisticRegression(C=1.0, max_iter=1000).fit(x, y)


def sweep(split: Split, scores: np.ndarray) -> list[dict]:
    """Precision at the natural duplicate rate and coverage for every candidate threshold."""
    out = []
    for t in sorted(set(scores[np.isfinite(scores)]), reverse=True):
        flagged = scores >= t
        out.append(
            {
                "threshold": float(t),
                "precision@base_rate": metrics.precision_at_base_rate(
                    split.is_dup, flagged, split.correct, split.base_rate
                ),
                "coverage": float((flagged & split.correct).sum() / max(1, split.is_dup.sum())),
                "n_correct": int((flagged & split.correct).sum()),
                "n_flags": int(flagged.sum()),
            }
        )
    return out


def select_threshold(curve: list[dict], target: float) -> float | None:
    """Lowest threshold (most coverage) that reaches `target` precision on validation."""
    ok = [
        p
        for p in curve
        if p["precision@base_rate"] >= target and p["n_correct"] >= MIN_CORRECT_ON_VALIDATION
    ]
    if not ok:
        return None
    return max(ok, key=lambda p: (p["coverage"], p["threshold"]))["threshold"]


def evaluate(split: Split, scores: np.ndarray, threshold: float, n_boot: int, seed: int):
    """Fixed-threshold metrics on test, with stratified bootstrap intervals."""
    flagged_all = scores >= threshold

    def score(idx):
        idx = np.asarray(idx)
        y, f, c = split.is_dup[idx], flagged_all[idx], split.correct[idx]
        out = metrics.binary_decision_scores(y, f, c)
        out["precision@base_rate"] = metrics.precision_at_base_rate(y, f, c, split.base_rate)
        return out

    result = score(np.arange(len(split.is_dup)))
    result.update(
        metrics.bootstrap_ci(
            score, len(split.is_dup), GATE_CI_KEYS, strata=split.is_dup, n_boot=n_boot, seed=seed
        )
    )
    result["n_flags"] = int(flagged_all.sum())
    return result


def analyze(
    predictions_dir: Path,
    repo: str,
    models: list[str],
    targets: list[float],
    system: str | None = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> list[dict]:
    """Result rows: one per (target, gate, model), plus one calibration row per model."""
    system = system or find_judge_system(predictions_dir, repo)
    raw = {
        (m, split): load_predictions(predictions_dir, repo, system, m, split)
        for m in models
        for split in ("validation", "test")
    }
    rows = []
    for model in models:
        others = {s: [raw[(o, s)] for o in models if o != model] for s in ("validation", "test")}
        val = Split(raw[(model, "validation")], others["validation"])
        test = Split(raw[(model, "test")], others["test"])
        learned = fit_learned_gate(val)
        n = len(test.is_dup)

        for signal in SIGNALS:
            if signal == "agreement" and len(models) < 2:
                continue
            if signal == "learned" and learned is None:
                continue
            curve = sweep(val, gate_scores(val, signal, learned))
            test_scores = gate_scores(test, signal, learned)
            for target in targets:
                row = {
                    "task": "duplicates-gated",
                    "subset": f"target P@base>={target:g}",
                    "system": f"gate:{signal}",
                    "model": model,
                    "repo": repo,
                    "target": target,
                    "n": float(n),
                    "judge": system,
                }
                threshold = select_threshold(curve, target)
                row["threshold"] = threshold
                if threshold is not None:
                    row.update(evaluate(test, test_scores, threshold, n_boot, seed))
                rows.append(row)

        cal = metrics.calibration(
            test.confidence, test.chosen, test.correct, test.is_dup, test.base_rate
        )
        rows.append(
            {
                "task": "calibration",
                "subset": "test",
                "system": "llm-confidence",
                "model": model,
                "repo": repo,
                "n": float(test.chosen.sum()),
                "judge": system,
                **cal,
            }
        )
    return rows
