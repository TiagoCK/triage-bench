"""Replay parity: run benchmark issues through the live service and compare with the
offline benchmark.

The replay set is the benchmark's LLM sample (`sample200-seed0`). Every other issue in the
benchmark window is backfilled first; the replay issues are then sent through the worker's
real handler as `opened` events, in creation order. So the service searches exactly the
corpus the benchmark searched, under the same time rule.

Checks:
  retrieval  service top-5 vs the benchmark EmbeddingRetriever's top-5 (mean overlap@5)
  labels     service labels vs the benchmark TfidfLabeler (trained on the train split)
  prompt     where both return the same candidates, the duplicate-judge prompts built from
             service data and from benchmark data must be byte-identical
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
import psycopg
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from triage.dataset.build import clean_body
from triage.eval.run import gold_sets
from triage.service.db import make_pool, migrate
from triage.service.embeddings import FastEmbedder
from triage.service.pipeline import LabelerRegistry, Pipeline, backfill
from triage.service.settings import Settings
from triage.systems.duplicates import EmbeddingRetriever, IssueDoc, build_judge_prompt
from triage.systems.labels import TfidfLabeler

log = logging.getLogger(__name__)

OVERLAP_BAR = 0.90


def replay_numbers(predictions_dir: Path, slug: str) -> list[int]:
    files = sorted((predictions_dir / slug).glob("*__test.json"))
    if not files:
        raise FileNotFoundError(f"no benchmark test predictions under {predictions_dir / slug}")
    return [i["number"] for i in json.loads(files[0].read_text())["items"]]


def run(
    repo: str,
    settings: Settings,
    processed_dir: Path,
    raw_dir: Path,
    predictions_dir: Path,
    out_path: Path,
    keep_db: bool = False,
) -> dict:
    slug = repo.replace("/", "__")
    issues = pd.read_parquet(processed_dir / slug / "issues.parquet")
    meta = json.loads((processed_dir / slug / "meta.json").read_text())
    pairs = pd.read_parquet(processed_dir / slug / "dup_pairs.parquet")
    replay = set(replay_numbers(predictions_dir, slug))
    window = set(issues["number"].astype(int))
    raw = [r for r in map(json.loads, (raw_dir / f"{slug}.jsonl").open()) if r["number"] in window]
    log.info("%s: %d issues in the benchmark window, replaying %d", repo, len(raw), len(replay))

    db_name = f"triage_replay_{slug.replace('-', '_').lower()}"
    admin = make_conninfo(settings.database_url, dbname="postgres")
    with psycopg.connect(admin, autocommit=True) as c:
        c.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')
        c.execute(f'CREATE DATABASE "{db_name}"')
    db_url = make_conninfo(settings.database_url, dbname=db_name)
    log.info("replay database %s on %s", db_name, conninfo_to_dict(db_url).get("host"))

    # The benchmark's labeler, trained on the train split, so labels are comparable.
    labeled = issues[issues["target_labels"].map(len) > 0]
    labeler = TfidfLabeler()
    labeler.fit(labeled[labeled["split"] == "train"], meta["label_vocab"])
    registry = LabelerRegistry(Path("/nonexistent"))
    registry.set(repo, labeler, "benchmark-train")
    embedder = FastEmbedder(settings.embed_model, settings.model_cache_dir)
    pipeline = Pipeline(settings, embedder, registry)

    timings = {}
    with make_pool(db_url, max_size=2) as pool, pool.connection() as conn:
        migrate(conn)
        t = time.perf_counter()
        history = [r for r in raw if r["number"] not in replay]
        backfill(conn, repo, history, embedder)
        timings["backfill_s"] = round(time.perf_counter() - t, 1)
        log.info("backfilled %d issues in %.0fs", len(history), timings["backfill_s"])

        t = time.perf_counter()
        for r in sorted((r for r in raw if r["number"] in replay), key=lambda r: r["created_at"]):
            payload = {
                "repo": repo,
                "action": "opened",
                "source": "replay",
                "issue": {
                    "number": r["number"],
                    "title": r["title"] or "",
                    "body": r["body"] or "",
                    "state": "open",
                    "labels": r.get("labels") or [],
                    "author": r.get("author"),
                    "url": r.get("url"),
                    "created_at": r["created_at"],
                    "updated_at": r["created_at"],
                },
            }
            with conn.transaction():
                pipeline.handle_issue(conn, payload)
        timings["replay_s"] = round(time.perf_counter() - t, 1)

        preds = {
            r["number"]: r
            for r in conn.execute(
                "SELECT number, candidates, labels, timings_ms FROM predictions WHERE repo = %s",
                (repo,),
            )
        }
        service_docs = {
            r["number"]: IssueDoc(r["number"], r["title"], clean_body(r["body"]))
            for r in conn.execute("SELECT number, title, body FROM issues WHERE repo = %s", (repo,))
        }

    if not keep_db:
        with psycopg.connect(admin, autocommit=True) as c:
            c.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')

    # Benchmark side, on the same issues (embeddings come from the benchmark's disk cache).
    bench = EmbeddingRetriever("BAAI/bge-small-en-v1.5")
    bench.index(issues)
    by_number = issues.set_index("number")
    numbers = sorted(replay)
    bench_top = {n: [h[0] for h in bench.search(n, settings.retrieval_k)] for n in numbers}
    service_top = {n: [c["number"] for c in preds[n]["candidates"]] for n in numbers}
    bench_labels = dict(zip(numbers, labeler.predict(by_number.loc[numbers]), strict=True))

    overlap = [
        len(set(service_top[n]) & set(bench_top[n])) / max(1, len(bench_top[n])) for n in numbers
    ]
    exact = [service_top[n] == bench_top[n] for n in numbers]
    label_same = [preds[n]["labels"] == bench_labels[n] for n in numbers]

    gold = gold_sets(issues, pairs)
    positives = [n for n in numbers if n in gold]
    recall = {
        side: sum(bool(set(top[n]) & gold[n]) for n in positives) / max(1, len(positives))
        for side, top in (("service", service_top), ("benchmark", bench_top))
    }

    prompts_compared = prompts_same = 0
    for n in numbers:
        if service_top[n] != bench_top[n]:
            continue
        prompts_compared += 1
        bench_docs = [
            IssueDoc(m, by_number.at[m, "title"], by_number.at[m, "body_clean"])
            for m in [n, *bench_top[n]]
        ]
        svc = build_judge_prompt(repo, service_docs[n], [service_docs[m] for m in service_top[n]])
        ref = build_judge_prompt(repo, bench_docs[0], bench_docs[1:])
        prompts_same += svc == ref

    stage_ms = pd.DataFrame([preds[n]["timings_ms"] for n in numbers])
    report = {
        "repo": repo,
        "replayed": len(numbers),
        "corpus": len(raw),
        "embed_model": settings.embed_model,
        "retrieval": {
            "mean_overlap@5": round(sum(overlap) / len(overlap), 4),
            "exact_match_rate": round(sum(exact) / len(exact), 4),
            "duplicates": len(positives),
            "recall@5_service": round(recall["service"], 4),
            "recall@5_benchmark": round(recall["benchmark"], 4),
        },
        "labels": {"identical_rate": round(sum(label_same) / len(label_same), 4)},
        "prompt": {"compared": prompts_compared, "identical": prompts_same},
        "timings": {
            **timings,
            **{f"p50_{k}_ms": round(float(stage_ms[k].median()), 1) for k in stage_ms.columns},
        },
    }
    report["pass"] = (
        report["retrieval"]["mean_overlap@5"] >= OVERLAP_BAR
        and report["labels"]["identical_rate"] == 1.0
        and prompts_same == prompts_compared
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    return report
