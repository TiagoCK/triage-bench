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


def _write_table(rows: list[dict], path: Path) -> None:
    from triage.eval.report import render

    path.write_text(render(rows))
