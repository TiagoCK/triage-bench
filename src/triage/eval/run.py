"""Run systems x models on one repository and record results.

Non-LLM systems are scored on the full test split ("full") and on the seeded sample used for
LLM systems ("sample<N>-seed<S>"), so LLM rows always have a like-for-like baseline next to
them. Changing --sample or --seed creates a new subset instead of mixing samples.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from triage import llm
from triage.config import ModelConfig
from triage.eval import metrics
from triage.systems import duplicates as dups
from triage.systems import labels

log = logging.getLogger(__name__)

LABEL_SYSTEMS = {"tfidf", "llm-zeroshot", "llm-fewshot"}
DUP_SYSTEMS = {"difflib", "bm25", "embed", "hybrid", "rerank", "llm-dups"}
LLM_SYSTEMS = {"llm-zeroshot", "llm-fewshot", "llm-dups"}
EMBED_SYSTEMS = {"embed", "hybrid", "rerank"}
ALL_SYSTEMS = LABEL_SYSTEMS | DUP_SYSTEMS

# Metrics that get bootstrap confidence intervals.
LABEL_CI_KEYS = ("micro_f1", "macro_f1", "micro_precision", "micro_recall")
RETRIEVAL_CI_KEYS = ("recall@1", "recall@5", "recall@10", "mrr@10")
DETECTION_CI_KEYS = ("average_precision", "recall@p90")
DECISION_CI_KEYS = (
    "recall@1",
    "recall@5",
    "mrr@10",
    "recall",
    "false_dup_rate",
    "precision@base_rate",
)


@dataclass
class EvalOptions:
    systems: list[str]
    models: list[ModelConfig] = field(default_factory=list)
    sample: int = 200
    seed: int = 0
    embed_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    few_shot_k: int = 5
    bootstrap: int = 1000  # resamples for confidence intervals; 0 disables them
    # Below this, duplicate metrics are noise (e.g. huggingface/transformers has 1), so the
    # duplicate task is skipped rather than reported.
    min_test_duplicates: int = 20
    # Same idea for labels: huggingface/transformers has 1 labeled test issue once
    # issue-form labels are excluded.
    min_test_labeled: int = 50


def run_repo(repo_dir: Path, opts: EvalOptions) -> list[dict]:
    meta = json.loads((repo_dir / "meta.json").read_text())
    issues = pd.read_parquet(repo_dir / "issues.parquet")
    pairs = pd.read_parquet(repo_dir / "dup_pairs.parquet")
    systems = _available_systems(opts.systems)
    models = [m for m in opts.models if _model_available(m)]

    rows: list[dict] = []
    n_test_labeled = meta["n_labeled"]["test"]
    if systems & LABEL_SYSTEMS and meta["label_vocab"]:
        if n_test_labeled >= opts.min_test_labeled:
            rows += _run_labels(meta, issues, systems, models, opts)
        else:
            log.warning(
                "%s: only %d labeled test issues (< %d); skipping the label task",
                meta["repo"],
                n_test_labeled,
                opts.min_test_labeled,
            )
    n_test_dups = pairs.loc[pairs["split"] == "test", "duplicate"].nunique()
    if systems & DUP_SYSTEMS:
        if n_test_dups >= opts.min_test_duplicates:
            rows += _run_dups(meta, issues, pairs, systems, models, opts)
        else:
            log.warning(
                "%s: only %d test duplicates (< %d); skipping the duplicate task",
                meta["repo"],
                n_test_dups,
                opts.min_test_duplicates,
            )

    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    for r in rows:
        r["repo"] = meta["repo"]
        r["timestamp"] = stamp
    return rows


# ---------------------------------------------------------------- labels


def _run_labels(meta, issues, systems, models, opts) -> list[dict]:
    vocab = meta["label_vocab"]
    names = [v["name"] for v in vocab]
    labeled = issues[issues["target_labels"].map(len) > 0]
    train = labeled[labeled["split"] == "train"]
    test = labeled[labeled["split"] == "test"]
    sample = test.sample(min(opts.sample, len(test)), random_state=opts.seed)
    log.info(
        "labels: %d train, %d test, sample %d, vocab %d",
        len(train),
        len(test),
        len(sample),
        len(names),
    )

    rows = []
    subset_name = _sample_name(opts)
    llm_label_systems = systems & {"llm-zeroshot", "llm-fewshot"}
    # The baseline is cheap, so always score it on the same sample the LLMs see.
    if "tfidf" in systems or (llm_label_systems and models):
        sys_ = labels.TfidfLabeler()
        sys_.fit(train, vocab)
        for subset, df in (("full", test), (subset_name, sample)):
            scores = _label_scores(list(df["target_labels"]), sys_.predict(df), names, opts)
            rows.append(_row("labels", subset, sys_.name, "-", scores))

    for model in models:
        for kind, k in (("llm-zeroshot", 0), ("llm-fewshot", opts.few_shot_k)):
            if kind not in systems:
                continue
            sys_ = labels.LLMLabeler(model, meta["repo"], few_shot_k=k)
            sys_.fit(train, vocab)
            log.info("labels: %s on %s (%d issues)", sys_.name, model.key, len(sample))
            preds = sys_.predict(sample)
            scores = _label_scores(list(sample["target_labels"]), preds, names, opts)
            scores.update(_ops(sys_.records))
            scores["failures"] = sys_.failures
            rows.append(_row("labels", subset_name, sys_.name, model.key, scores))
    return rows


def _label_scores(gold: list, pred: list, names: list[str], opts: EvalOptions) -> dict:
    def score(idx):
        return metrics.multilabel_scores([gold[i] for i in idx], [pred[i] for i in idx], names)

    scores = score(range(len(gold)))
    scores.update(_ci(opts, score, len(gold), LABEL_CI_KEYS))
    return scores


# ---------------------------------------------------------------- duplicates


def gold_sets(issues: pd.DataFrame, pairs: pd.DataFrame) -> dict[int, set[int]]:
    """Acceptable answers for each duplicate: its canonical issue, plus any *earlier* issue
    that was itself marked a duplicate of the same canonical (linking to a sibling duplicate
    still points the reporter at the right thread)."""
    created = issues.set_index("number")["created_at"]
    by_canon: dict[int, list[int]] = {}
    for d, c in zip(pairs["duplicate"], pairs["canonical"], strict=True):
        by_canon.setdefault(int(c), []).append(int(d))
    gold: dict[int, set[int]] = {}
    for d, c in zip(pairs["duplicate"], pairs["canonical"], strict=True):
        d, c = int(d), int(c)
        siblings = {s for s in by_canon[c] if s != d and created[s] < created[d]}
        gold.setdefault(d, set()).update({c} | siblings)
    return gold


def _run_dups(meta, issues, pairs, systems, models, opts) -> list[dict]:
    gold = gold_sets(issues, pairs)
    test_pairs = pairs[pairs["split"] == "test"]
    pos = sorted({int(n) for n in test_pairs["duplicate"]})  # one query per duplicate
    neg_df = issues[(issues["split"] == "test") & ~issues["is_dup_flagged"]]
    neg = [int(n) for n in neg_df["number"]]
    base_rate = len(pos) / (len(pos) + len(neg)) if neg else 1.0

    half = max(1, opts.sample // 2)
    pos_s = pd.Series(pos).sample(min(half, len(pos)), random_state=opts.seed).tolist()
    neg_s = (
        pd.Series(neg).sample(min(half, len(neg)), random_state=opts.seed).tolist() if neg else []
    )
    log.info(
        "dups: %d test pairs, %d negatives (base rate %.3f), sample %d+%d",
        len(pos),
        len(neg),
        base_rate,
        len(pos_s),
        len(neg_s),
    )

    retrievers: dict[str, dups.Retriever] = {}
    if "difflib" in systems:
        retrievers["difflib"] = dups.DifflibRetriever()
    if "bm25" in systems:
        retrievers["bm25"] = dups.BM25Retriever()
    if systems & {"embed", "hybrid", "rerank"}:
        retrievers["embed"] = dups.EmbeddingRetriever(opts.embed_model)
    embed_ok = importlib.util.find_spec("sentence_transformers") is not None
    run_llm = "llm-dups" in systems and bool(models)
    first_key = "hybrid" if embed_ok else "bm25"
    if systems & {"hybrid", "rerank"} or (run_llm and first_key == "hybrid"):
        retrievers["hybrid"] = dups.HybridRetriever(dups.EmbeddingRetriever(opts.embed_model))
    if run_llm and first_key == "bm25" and "bm25" not in retrievers:
        retrievers["bm25"] = dups.BM25Retriever()
    if "rerank" in systems:
        retrievers["rerank"] = dups.RerankedRetriever(
            dups.HybridRetriever(dups.EmbeddingRetriever(opts.embed_model)), opts.reranker_model
        )

    rows = []
    subset_name = _sample_name(opts)
    for key, r in retrievers.items():
        log.info("dups: indexing %s", r.name)
        r.index(issues)
        if key not in systems and not (run_llm and key == first_key):
            continue  # an intermediate stage, not a system to report
        log.info("dups: searching with %s", r.name)
        hits = {n: r.search(n, 10) for n in pos + neg}
        for subset, p, ng in (("full", pos, neg), (subset_name, pos_s, neg_s)):
            scores = _retrieval_scores(
                [[h[0] for h in hits[n]] for n in p], [gold[n] for n in p], opts
            )
            if subset == "full":
                scores.update(_detection(*_detection_inputs(hits, gold, p, ng), opts))
            rows.append(_row("duplicates", subset, r.name, "-", scores))

    if run_llm:
        first = retrievers[first_key]
        for model in models:
            judge = dups.LLMDuplicateJudge(first, model, meta["repo"])
            log.info("dups: %s on %s (%d issues)", judge.name, model.key, len(pos_s) + len(neg_s))
            items = pos_s + neg_s
            results = judge.judge_many(items)
            scores = _decision_scores(
                is_dup=[n in set(pos_s) for n in items],
                ranked=[[h[0] for h in res[0]] for res in results],
                flagged=[res[1] is not None for res in results],
                correct=[
                    res[1] in gold.get(n, set()) for n, res in zip(items, results, strict=True)
                ],
                golds=[gold.get(n, set()) for n in items],
                base_rate=base_rate,
                opts=opts,
            )
            scores.update(_ops(judge.records))
            scores["failures"] = judge.failures
            rows.append(_row("duplicates", subset_name, judge.name, model.key, scores))
    return rows


def _retrieval_scores(ranked: list, golds: list, opts: EvalOptions) -> dict:
    def score(idx):
        return metrics.retrieval_scores([ranked[i] for i in idx], [golds[i] for i in idx])

    scores = score(range(len(ranked)))
    scores.update(_ci(opts, score, len(ranked), RETRIEVAL_CI_KEYS))
    return scores


def _detection(is_dup: list, top: list, correct: list, opts: EvalOptions) -> dict:
    y, s, c = np.asarray(is_dup), np.asarray(top), np.asarray(correct)

    def score(idx):
        return metrics.detection_scores(y[idx], s[idx], c[idx])

    scores = score(np.arange(len(y)))
    # Stratified, so every resample keeps the real duplicate rate.
    scores.update(_ci(opts, score, len(y), DETECTION_CI_KEYS, strata=y))
    return scores


def _decision_scores(is_dup, ranked, flagged, correct, golds, base_rate, opts) -> dict:
    """Scores for the LLM judge on a balanced sample of duplicates and non-duplicates."""
    y, f, c = np.asarray(is_dup), np.asarray(flagged), np.asarray(correct)

    def score(idx):
        idx = np.asarray(idx)
        pos_idx = idx[y[idx]]
        out = metrics.retrieval_scores([ranked[i] for i in pos_idx], [golds[i] for i in pos_idx])
        out.update(metrics.binary_decision_scores(y[idx], f[idx], c[idx]))
        out["precision@base_rate"] = metrics.precision_at_base_rate(
            y[idx], f[idx], c[idx], base_rate
        )
        return out

    scores = score(np.arange(len(y)))
    scores.update(_ci(opts, score, len(y), DECISION_CI_KEYS, strata=y))
    return scores


def _ci(opts: EvalOptions, score_fn, n: int, keys, strata=None) -> dict:
    return metrics.bootstrap_ci(
        score_fn, n, keys, strata=strata, n_boot=opts.bootstrap, seed=opts.seed
    )


def _detection_inputs(hits, gold, pos, neg):
    """Top-1 score as the 'is this a duplicate?' confidence for score-based retrievers."""
    labels_, top, correct = [], [], []
    for n, is_pos in [(n, True) for n in pos] + [(n, False) for n in neg]:
        h = hits[n]
        labels_.append(is_pos)
        top.append(h[0][1] if h else float("-inf"))
        correct.append(bool(h) and is_pos and h[0][0] in gold[n])
    return labels_, top, correct


# ---------------------------------------------------------------- helpers


def _sample_name(opts: EvalOptions) -> str:
    return f"sample{opts.sample}-seed{opts.seed}"


def _row(task: str, subset: str, system: str, model: str, scores: dict) -> dict:
    return {"task": task, "subset": subset, "system": system, "model": model, **scores}


def _ops(records) -> dict:
    if not records:
        return {}
    summary = metrics.latency_cost_summary(
        [r.latency_s for r in records], [r.cost_usd for r in records]
    )
    summary["cache_hit_rate"] = sum(r.cached for r in records) / len(records)
    resolved = sorted({r.resolved_model for r in records if r.resolved_model})
    if resolved:
        summary["resolved_models"] = resolved
    return summary


def _available_systems(requested: list[str]) -> set[str]:
    unknown = set(requested) - ALL_SYSTEMS
    if unknown:
        raise ValueError(f"unknown systems: {sorted(unknown)}; choose from {sorted(ALL_SYSTEMS)}")
    out = set(requested)
    if out & EMBED_SYSTEMS and importlib.util.find_spec("sentence_transformers") is None:
        log.warning(
            "sentence-transformers not installed (uv sync --extra embed); skipping %s",
            sorted(out & EMBED_SYSTEMS),
        )
        out -= EMBED_SYSTEMS
    if out & LLM_SYSTEMS and importlib.util.find_spec("litellm") is None:
        log.warning(
            "litellm not installed (uv sync --extra llm); skipping %s", sorted(out & LLM_SYSTEMS)
        )
        out -= LLM_SYSTEMS
    return out


def _model_available(model: ModelConfig) -> bool:
    if model.id.startswith(llm.CLI_PREFIX):
        return _claude_cli_ready(model)
    if model.id.startswith("anthropic/") and not os.environ.get("ANTHROPIC_API_KEY"):
        log.warning("ANTHROPIC_API_KEY not set; skipping model %s", model.key)
        return False
    if model.id.startswith(("ollama/", "ollama_chat/")):
        base = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
        try:
            tags = httpx.get(f"{base}/api/tags", timeout=3).json()
        except (httpx.HTTPError, ValueError):
            log.warning("Ollama not reachable at %s; skipping model %s", base, model.key)
            return False
        wanted = model.id.split("/", 1)[1]
        have = {m["name"] for m in tags.get("models", [])}
        if wanted not in have and f"{wanted}:latest" not in have:
            log.warning("Ollama model %s not pulled (ollama pull %s); skipping", wanted, wanted)
            return False
    return True


def _claude_cli_ready(model: ModelConfig) -> bool:
    """The CLI exists and is signed in with a Claude subscription (not an API key)."""
    binary = llm.claude_bin()
    if not binary:
        log.warning("`claude` CLI not found (install it or set CLAUDE_BIN); skipping %s", model.key)
        return False
    try:
        out = subprocess.run(
            [binary, "auth", "status"],
            capture_output=True,
            text=True,
            env=llm.cli_env(),
            timeout=30,
        )
        method = json.loads(out.stdout).get("authMethod")
    except (OSError, subprocess.TimeoutExpired, ValueError):
        method = None
    if method != "claude.ai":
        log.warning(
            "claude CLI auth method is %r, not a subscription login (run `claude auth "
            "login`); skipping %s",
            method,
            model.key,
        )
        return False
    return True


# ---------------------------------------------------------------- results store


def merge_results(path: Path, new_rows: list[dict]) -> list[dict]:
    """Keep the latest row per (repo, task, subset, system, model)."""
    existing = json.loads(path.read_text()) if path.exists() else []
    key = lambda r: (r["repo"], r["task"], r["subset"], r["system"], r["model"])  # noqa: E731
    merged = {key(r): r for r in existing}
    merged.update({key(r): r for r in new_rows})
    rows = sorted(merged.values(), key=key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, default=str))
    return rows
