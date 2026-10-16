from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv

from triage.config import load_models, load_repos

app = typer.Typer(add_completion=False, no_args_is_help=True, help="GitHub issue triage benchmark.")

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RESULTS = Path("results/latest.json")


@app.callback()
def _setup(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # LiteLLM and httpx are chatty at INFO.
    for noisy in ("httpx", "LiteLLM", "litellm"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _select(repo: list[str] | None, config: Path):
    repos = load_repos(config)
    if repo:
        repos = [r for r in repos if r.name in repo]
        if not repos:
            raise typer.BadParameter(f"none of {repo} are in {config}")
    return repos


@app.command()
def collect(
    repo: list[str] = typer.Option(None, help="Only these repos (owner/name). Default: all."),
    limit: int = typer.Option(None, help="Stop after this many issues per repo (for testing)."),
    config: Path = typer.Option(Path("configs/repos.yaml")),
    raw_dir: Path = typer.Option(RAW_DIR),
) -> None:
    """Download issues, labels and duplicate links from GitHub into raw JSONL."""
    from triage.dataset.collect import collect_repo
    from triage.github_client import GitHubClient

    client = GitHubClient(os.environ.get("GITHUB_TOKEN", ""))
    for cfg in _select(repo, config):
        collect_repo(client, cfg.name, cfg.since, raw_dir, limit=limit)


@app.command()
def build(
    repo: list[str] = typer.Option(None),
    config: Path = typer.Option(Path("configs/repos.yaml")),
    raw_dir: Path = typer.Option(RAW_DIR),
    out_dir: Path = typer.Option(PROCESSED_DIR),
) -> None:
    """Build train/test tables, label vocabulary and duplicate pairs from raw JSONL."""
    from triage.dataset.build import build_repo

    for cfg in _select(repo, config):
        if not (raw_dir / f"{cfg.slug}.jsonl").exists():
            typer.echo(f"skip {cfg.name}: no raw data (run `triage collect` first)")
            continue
        meta = build_repo(cfg, raw_dir, out_dir)
        typer.echo(
            f"{cfg.name}: issues {meta['n_issues']}, labeled {meta['n_labeled']}, "
            f"duplicates {meta['n_duplicates']} (of {meta['n_closed_as_duplicate']} closed as "
            f"duplicate), vocab {len(meta['label_vocab'])} labels"
        )
        typer.echo(f"  duplicate pairs: {meta['dup_pair_stats']}")


@app.command("eval")
def eval_(
    systems: str = typer.Option("tfidf,difflib,bm25", help="Comma-separated system names."),
    models: str = typer.Option("", help="Comma-separated model keys from configs/models.yaml."),
    repo: list[str] = typer.Option(None),
    sample: int = typer.Option(200, help="Test issues per task for LLM systems."),
    seed: int = typer.Option(0),
    bootstrap: int = typer.Option(1000, help="Bootstrap resamples for 95% CIs; 0 disables."),
    min_test_duplicates: int = typer.Option(
        20, help="Skip the duplicate task for repos with fewer test duplicates than this."
    ),
    min_test_labeled: int = typer.Option(
        50, help="Skip the label task for repos with fewer labeled test issues than this."
    ),
    with_validation: bool = typer.Option(
        False, help="Also run the LLM duplicate judge on a validation sample (for gates)."
    ),
    embed_model: str = typer.Option("BAAI/bge-small-en-v1.5"),
    reranker_model: str = typer.Option("BAAI/bge-reranker-base"),
    config: Path = typer.Option(Path("configs/repos.yaml")),
    models_config: Path = typer.Option(Path("configs/models.yaml")),
    processed_dir: Path = typer.Option(PROCESSED_DIR),
    results: Path = typer.Option(RESULTS),
) -> None:
    """Score systems (and LLM models) on the built benchmark; update results and table."""
    from triage.eval.run import EvalOptions, merge_results, run_repo

    model_cfgs = load_models(models_config)
    model_keys = [m for m in models.split(",") if m]
    missing = [m for m in model_keys if m not in model_cfgs]
    if missing:
        raise typer.BadParameter(f"unknown models {missing}; see {models_config}")
    opts = EvalOptions(
        systems=[s for s in systems.split(",") if s],
        models=[model_cfgs[m] for m in model_keys],
        sample=sample,
        seed=seed,
        bootstrap=bootstrap,
        min_test_duplicates=min_test_duplicates,
        min_test_labeled=min_test_labeled,
        with_validation=with_validation,
        predictions_dir=results.parent / "predictions",
        embed_model=embed_model,
        reranker_model=reranker_model,
    )
    rows = []
    for cfg in _select(repo, config):
        repo_dir = processed_dir / cfg.slug
        if not (repo_dir / "meta.json").exists():
            typer.echo(f"skip {cfg.name}: not built (run `triage build` first)")
            continue
        rows += run_repo(repo_dir, opts)

    all_rows = merge_results(results, rows)
    run_log = results.parent / "runs" / f"{datetime.now(UTC):%Y%m%dT%H%M%S}.json"
    run_log.parent.mkdir(parents=True, exist_ok=True)
    run_log.write_text(json.dumps(rows, indent=2, default=str))
    _write_table(all_rows, results.parent / "table.md")
    typer.echo(f"{len(rows)} result rows; table at {results.parent / 'table.md'}")


@app.command("export-labeler")
def export_labeler(
    repo: str = typer.Option(..., help="owner/name"),
    train_only: bool = typer.Option(
        False, help="Train on the benchmark's train split only (reproduces its model)."
    ),
    processed_dir: Path = typer.Option(PROCESSED_DIR),
    out_dir: Path = typer.Option(Path("artifacts/labelers")),
) -> None:
    """Fit the TF-IDF labeler and save it as a versioned artifact for the live service.

    The default trains on every labeled issue and becomes the repo's LATEST model.
    --train-only reproduces the benchmark's model exactly, for parity checks.
    """
    import hashlib

    import pandas as pd

    from triage.systems.labels import TfidfLabeler

    repo_dir = processed_dir / repo.replace("/", "__")
    meta = json.loads((repo_dir / "meta.json").read_text())
    issues = pd.read_parquet(repo_dir / "issues.parquet")
    labeled = issues[issues["target_labels"].map(len) > 0]
    if train_only:
        labeled = labeled[labeled["split"] == "train"]
    data_key = json.dumps(
        [
            meta["label_vocab"],
            labeled["number"].tolist(),
            [list(x) for x in labeled["target_labels"]],
            train_only,
        ]
    )
    digest = hashlib.sha256(data_key.encode()).hexdigest()[:8]
    version = f"{datetime.now(UTC):%Y%m%d}-{digest}" + ("-train" if train_only else "")

    labeler = TfidfLabeler()
    labeler.fit(labeled, meta["label_vocab"])
    target = out_dir / repo.replace("/", "__") / version
    labeler.save(target)
    (target / "meta.json").write_text(
        json.dumps(
            {
                "repo": repo,
                "version": version,
                "train_only": train_only,
                "n_issues": len(labeled),
                "trained_on": {
                    "from": str(labeled["created_at"].min()),
                    "to": str(labeled["created_at"].max()),
                },
                "label_vocab": [v["name"] for v in meta["label_vocab"]],
            },
            indent=2,
        )
    )
    if not train_only:
        (target.parent / "LATEST").write_text(version + "\n")
    typer.echo(f"{repo}: {len(labeled)} issues -> {target}")


@app.command("analyze-gates")
def analyze_gates(
    repo: str = typer.Option(..., help="owner/name"),
    models: str = typer.Option("haiku-sub,qwen3.5-9b", help="Comma-separated model keys."),
    targets: str = typer.Option("0.5,0.8,0.9", help="Target precisions at the base rate."),
    system: str = typer.Option(None, help="Judge system name; default: the only one found."),
    bootstrap: int = typer.Option(1000),
    seed: int = typer.Option(0),
    results: Path = typer.Option(RESULTS),
) -> None:
    """Choose confidence-gate thresholds on validation and score them on test.

    Reads saved judge predictions (from `triage eval ... --with-validation`); no model calls.
    """
    from triage.eval.gates import analyze
    from triage.eval.run import merge_results

    rows = analyze(
        results.parent / "predictions",
        repo,
        [m for m in models.split(",") if m],
        [float(t) for t in targets.split(",") if t],
        system=system,
        n_boot=bootstrap,
        seed=seed,
    )
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    for r in rows:
        r["timestamp"] = stamp
    all_rows = merge_results(results, rows)
    _write_table(all_rows, results.parent / "table.md")
    typer.echo(f"{len(rows)} gate rows; table at {results.parent / 'table.md'}")


@app.command()
def report(results: Path = typer.Option(RESULTS)) -> None:
    """Re-render results/table.md from results/latest.json."""
    _write_table(json.loads(results.read_text()), results.parent / "table.md")


@app.command()
def gate(
    thresholds: Path = typer.Option(Path("configs/smoke_gate.yaml")),
    results: Path = typer.Option(RESULTS),
) -> None:
    """Fail (exit 1) if any metric falls below its floor. Used by CI as a regression gate."""
    rows = json.loads(results.read_text())
    spec = yaml.safe_load(thresholds.read_text())["floors"]
    failures = []
    for f in spec:
        match = [
            r
            for r in rows
            if all(r.get(k) == f[k] for k in ("repo", "task", "subset", "system", "model"))
        ]
        if not match:
            failures.append(f"missing result for {f}")
            continue
        value = match[0].get(f["metric"])
        if value is None or value < f["min"]:
            failures.append(f"{f['system']} {f['task']} {f['metric']} = {value} < {f['min']}")
    for msg in failures:
        typer.echo(f"FAIL {msg}")
    if failures:
        raise typer.Exit(1)
    typer.echo(f"gate passed ({len(spec)} checks)")


service_app = typer.Typer(no_args_is_help=True, help="The live triage service (shadow mode).")
app.add_typer(service_app, name="service")


@service_app.command("migrate")
def service_migrate() -> None:
    """Apply pending database migrations."""
    from triage.service.db import make_pool, migrate
    from triage.service.settings import get_settings

    with make_pool(get_settings().database_url, max_size=1) as pool, pool.connection() as conn:
        applied = migrate(conn)
    typer.echo(f"applied {applied}" if applied else "database is up to date")


@service_app.command("api")
def service_api(
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(8000),
) -> None:
    """Run the webhook receiver and health check."""
    import uvicorn

    from triage.service.app import create_app

    uvicorn.run(create_app(), host=host, port=port)


@service_app.command("worker")
def service_worker() -> None:
    """Run a worker that processes queued jobs until stopped."""
    from triage.service.db import make_pool, migrate
    from triage.service.settings import get_settings
    from triage.service.worker import run_forever

    settings = get_settings()
    with make_pool(settings.database_url, max_size=2) as pool:
        with pool.connection() as conn:
            migrate(conn)
        run_forever(pool, settings)


@service_app.command("backfill")
def service_backfill(
    repo: str = typer.Option(..., help="owner/name"),
    from_raw: Path = typer.Option(None, help="A benchmark raw JSONL file (offline)."),
    from_api: bool = typer.Option(False, help="Fetch from GitHub (needs GITHUB_TOKEN)."),
    since: str = typer.Option(None, help="Only issues created on or after this date."),
) -> None:
    """Load historical issues so new issues have earlier ones to match. No predictions."""
    from triage.service.db import make_pool, migrate
    from triage.service.embeddings import FastEmbedder
    from triage.service.pipeline import backfill
    from triage.service.settings import get_settings

    if bool(from_raw) == from_api:
        raise typer.BadParameter("pass exactly one of --from-raw or --from-api")
    since_dt = datetime.fromisoformat(since).replace(tzinfo=UTC) if since else None
    if from_raw:
        records = (json.loads(line) for line in from_raw.open())
        if since_dt:
            records = (
                r
                for r in records
                if datetime.fromisoformat(r["created_at"].replace("Z", "+00:00")) >= since_dt
            )
    else:
        from triage.dataset.collect import iter_issue_nodes, normalize_issue
        from triage.github_client import GitHubClient

        client = GitHubClient(os.environ.get("GITHUB_TOKEN", ""))
        start = since_dt or datetime(2008, 1, 1, tzinfo=UTC)
        records = (normalize_issue(n, repo) for n in iter_issue_nodes(client, repo, start))

    settings = get_settings()
    embedder = FastEmbedder(settings.embed_model, settings.model_cache_dir)
    with make_pool(settings.database_url, max_size=1) as pool, pool.connection() as conn:
        migrate(conn)
        n = backfill(conn, repo, records, embedder)
    typer.echo(f"{repo}: backfilled {n} issues")


@service_app.command("replay")
def service_replay(
    repo: str = typer.Option(..., help="owner/name"),
    keep_db: bool = typer.Option(False, help="Keep the replay database afterwards."),
    processed_dir: Path = typer.Option(PROCESSED_DIR),
    raw_dir: Path = typer.Option(RAW_DIR),
    results: Path = typer.Option(RESULTS),
) -> None:
    """Replay the benchmark's LLM sample through the service and check parity with the
    offline benchmark (retrieval, labels, judge prompt). Exits 1 if a parity bar is missed."""
    from triage.service import replay
    from triage.service.settings import get_settings

    slug = repo.replace("/", "__")
    out = results.parent / "parity" / f"{slug}.json"
    report = replay.run(
        repo,
        get_settings(),
        processed_dir,
        raw_dir,
        results.parent / "predictions",
        out,
        keep_db=keep_db,
    )
    typer.echo(json.dumps(report, indent=2))
    if not report["pass"]:
        raise typer.Exit(1)


def _write_table(rows: list[dict], path: Path) -> None:
    from triage.eval.report import render

    path.write_text(render(rows))
