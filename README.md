# triage-bench

An open, time-split benchmark and evaluation harness for GitHub issue triage: **label
prediction** and **duplicate detection**, measured on real maintainer decisions from public
repositories. It compares classic baselines, dense retrieval, and LLMs (hosted Claude and
local Ollama models through one interface) on quality, cost and latency.

## Why

Plenty of AI issue-triage tools already exist: Dosu (commercial), GitHub's own duplicate
detection (public preview, June 2026), and dozens of GitHub Actions built on difflib or a
single LLM prompt. Almost none of them publish how often they are right. A triage bot that
posts a wrong "this is a duplicate of #123" comment costs maintainers more than it saves.
So this project starts by measuring:

- How well does each approach predict the labels maintainers actually applied?
- How often does it find the issue a maintainer marked as the original?
- At a repository's real duplicate rate, what share of a bot's duplicate comments would be
  correct, and how many good duplicates does it catch while staying above 90% precision?
- What does that cost per 1,000 issues, and how slow is it?

Phase 2 (planned) turns the best configuration into a deployed GitHub App with shadow mode
and a feedback loop.

## How the benchmark avoids leakage

- **Time split.** Issues created before a cutoff date are train / search index; issues after
  are test. The label vocabulary is chosen from train issues only.
- **Per-query time mask.** When looking for the original of a duplicate, a system can only
  search issues created *before* that duplicate. Tests in `tests/test_leakage.py` enforce
  this for every retriever.
- **Ground truth comes from maintainers.** Duplicate pairs come from GitHub's
  `MarkedAsDuplicateEvent` timeline events (un-marking cancels them) and "Duplicate of #N"
  comments, including full issue URLs. Labels are the ones maintainers applied.
- **Duplicate direction is inferred, not trusted.** On astral-sh/uv, the event's `canonical`
  field was always the issue whose timeline it came from, whether that issue was the original
  or the duplicate. Links are therefore stored undirected and oriented at build time by
  (1) an explicit comment, (2) which side was closed as / labeled duplicate, (3) creation
  order. Issues closed as duplicate with no link fall back to a single unambiguous issue
  reference in their comments (source `mention`). `meta.json` records how many pairs each
  rule produced.
- **Label targets are triage decisions, not form choices.** Workflow labels (verified,
  released, needs-*, pending, test-plan, WIP, bot `ai-*` tags) are excluded. So are template
  labels: labels that, for most of their train uses, were added within 60 seconds of the
  issue being opened. On transformers, 1,608 of 1,619 `bug` labels were added at creation by
  the issue's own author, so they come from the issue form. `meta.json` lists every excluded
  template label with its at-creation share.
- **Repos with too little ground truth are skipped, not reported.** A task runs only with at
  least 20 test duplicates or 50 labeled test issues. huggingface/transformers fails both
  (1 test duplicate, 1 labeled test issue once form labels are removed), so it is collected
  but not scored.
- **The search corpus reaches back further than the test window.** Originals of recent
  duplicates are often old, so a repo's `since` date can predate its train window.
- **Known caveats.** Labels are the issue's final labels, not the labels at open time. BM25
  IDF is computed over the whole window (corpus statistics only, not labels or links).
  Linking to an earlier duplicate of the same original also counts as correct.

## Systems

| Task | System | Notes |
|---|---|---|
| Labels | `tfidf` | TF-IDF + one-vs-rest logistic regression |
| Labels | `llm-zeroshot` | Label names + descriptions in the prompt |
| Labels | `llm-fewshot` | Plus the 5 most similar train issues and their labels |
| Duplicates | `difflib` | Title string similarity over the 2,000 most recent issues (the common hobby-bot approach) |
| Duplicates | `bm25` | Sparse BM25, title weighted x2 |
| Duplicates | `embed` | `bge-small-en-v1.5` cosine similarity |
| Duplicates | `hybrid` | Reciprocal rank fusion of BM25 + embeddings |
| Duplicates | `rerank` | Hybrid, then a `bge-reranker-base` cross-encoder over the top 20 |
| Duplicates | `llm-dups` | Hybrid top 5, then an LLM picks the duplicate or rejects all |

LLM models live in [configs/models.yaml](configs/models.yaml) and go through
[src/triage/llm.py](src/triage/llm.py) (LiteLLM). Any LiteLLM model string works, so
comparing Claude against a local model is a flag:
`--models haiku,qwen3.5-9b`. Every call is cached on disk, so reruns are free.

## Metrics

- **Labels:** micro / macro F1, precision, recall.
- **Duplicate retrieval:** Recall@1/5/10 and MRR@10 (is the original in the top k?).
- **Duplicate detection (score-based systems):** average precision, and recall at 90%
  precision (R@P90), over all test issues at the repository's natural duplicate rate.
- **Duplicate detection (LLM):** flag recall, false-duplicate rate on non-duplicates, and
  **P@base**: the balanced sample is re-weighted to the natural duplicate rate, so this is the
  share of bot comments that would actually be correct.
- **Ops:** $ per 1,000 issues, p50 / p95 latency, cache hit rate, failed calls.

Every quality metric comes with a **95% bootstrap confidence interval** (1,000 resamples of
the test issues, shown in brackets in the table). Duplicate-detection metrics are resampled
separately within duplicates and non-duplicates, so every resample keeps the real duplicate
rate. Test sets here are small (uv has 36 test duplicates), so the intervals are wide. When
two systems' intervals overlap heavily, the benchmark cannot tell them apart. Use
`--bootstrap 0` to skip the intervals for a quick run.

LLM systems run on a seeded sample (`--sample`, default 200). The TF-IDF baseline and the
LLM's first-stage retriever are always scored on the same sample, so every LLM row has a
like-for-like baseline beside it.

## Quickstart

```bash
uv sync --all-extras          # base + embeddings (torch) + LiteLLM
cp .env.example .env          # add GITHUB_TOKEN (and ANTHROPIC_API_KEY for Claude rows)
```

Collect and build (edit [configs/repos.yaml](configs/repos.yaml) to choose repositories and
dates):

```bash
uv run triage collect --repo astral-sh/uv
uv run triage build --repo astral-sh/uv
```

Evaluate:

```bash
uv run triage eval --repo astral-sh/uv --systems tfidf,difflib,bm25,embed,hybrid,rerank
uv run triage eval --repo astral-sh/uv --systems llm-zeroshot,llm-fewshot,llm-dups \
  --models qwen3.5-9b,haiku --sample 200
```

Results accumulate in `results/latest.json` (one row per repo, task, subset, system and model)
and render to [results/table.md](results/table.md).

For local models, run Ollama and pull a model listed in `configs/models.yaml`. Thinking is
disabled through `params: {think: false}`; with it on, Qwen 3.5 spent its whole token budget
reasoning and returned nothing.

## Running Claude on a subscription

The `haiku-sub` and `sonnet-sub` models run through the Claude Code CLI (`claude -p`) on
your own Claude subscription login, not an API key:

```bash
claude auth login            # sign in with your Claude account (browser)
claude auth status           # should report "authMethod": "claude.ai"
uv run triage eval --repo microsoft/vscode --systems llm-zeroshot,llm-fewshot,llm-dups \
  --models haiku-sub --sample 200
```

- **Policy.** Anthropic's support article ([Use the Claude Agent SDK with your Claude
  plan](https://support.claude.com/en/articles/15036540-use-the-claude-agent-sdk-with-your-claude-plan))
  says `claude -p` usage draws from your subscription limits. Its [usage
  policy](https://code.claude.com/docs/en/legal-and-compliance) says those limits assume
  ordinary, individual use, and that products serving other people must use API keys.
  Running this benchmark on your own machine fits that. A deployed triage bot does not.
- **Not identical to the API.** Each call replaces Claude Code's system prompt with ours and
  disables all tools. It still goes through the Claude Code harness, and the CLI cannot set
  `max_tokens` or `temperature`. Rows are therefore labeled `haiku-sub`, kept apart from
  `haiku` (API).
- **Cost is an estimate.** `$/1k issues` for `-sub` rows is the CLI's estimate at API
  prices; nothing is billed per call. Latency is the CLI's API time, excluding its startup.
- **Usage limits.** A 200-issue run is about 600 calls. If you hit your plan's usage window,
  re-run the same command later: finished calls come from the cache.
- `ANTHROPIC_API_KEY` is removed from the CLI's environment so these calls cannot fall back
  to API billing. `--bare` mode is not used, because it never reads the subscription login.

## Development

```bash
make test     # unit tests, offline
make lint
make smoke    # build + eval + regression gate on a synthetic fixture, offline (what CI runs)
```

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs lint, the tests, and the smoke
eval with a regression gate ([configs/smoke_gate.yaml](configs/smoke_gate.yaml)). It installs
only the base dependencies, so no torch, no LiteLLM and no network calls to model providers.
The fixture in `tests/fixtures` is synthetic and only checks the plumbing. Its numbers say
nothing about real-world quality.

## Layout

```
configs/                 repos, models, CI gate floors
src/triage/
  github_client.py       GraphQL client: retries, rate limits, page cache
  dataset/collect.py     issues, labels, duplicate links -> raw JSONL
  dataset/build.py       cleaning, time split, label vocab, duplicate pairs -> parquet
  llm.py                 the one LLM entry point (LiteLLM, cache, cost, latency)
  systems/labels.py      label systems
  systems/duplicates.py  retrievers and the LLM duplicate judge
  eval/metrics.py        metric functions
  eval/run.py            systems x models runner
  eval/report.py         Markdown results table
tests/                   unit, leakage, end-to-end smoke tests
```

## Supply-chain note

LiteLLM 1.82.7 and 1.82.8 on PyPI were backdoored in March 2026. `pyproject.toml` excludes
them, and `uv.lock` pins exact versions with hashes. Install with `uv sync --locked`.
