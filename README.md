# triage-bench

[![ci](https://github.com/TiagoCK/triage-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/TiagoCK/triage-bench/actions/workflows/ci.yml)

An open, time-split benchmark and evaluation harness for GitHub issue triage: **label
prediction** and **duplicate detection**, measured on real maintainer decisions from public
repositories. It compares classic baselines, dense retrieval, and LLMs (hosted Claude and
local Ollama models through one interface) on quality, cost and latency.

## Key findings

Measured on microsoft/vscode (11,198 test issues, 603 test duplicates) and astral-sh/uv
(972 test issues, 36 test duplicates). Brackets are 95% bootstrap confidence intervals. Full
tables are in [results/table.md](results/table.md).

**1. For finding duplicates, embeddings clearly beat keyword search, which clearly beats the
approach most hobby bots use.** Share of duplicates whose original appears in the top 5
results, on vscode:

| System | Recall@5 |
|---|---:|
| difflib title matching | 0.29 [0.25, 0.33] |
| BM25 keyword search | 0.39 [0.35, 0.42] |
| Embeddings (bge-small) | **0.55** [0.51, 0.59] |
| Embeddings + BM25 | 0.53 [0.49, 0.58] |

None of the first three intervals overlap. Adding BM25 to embeddings gains nothing. uv shows
the same order (difflib 0.17, embeddings 0.58), with wider intervals.

**2. No system here can safely decide on its own that an issue is a duplicate.**
- Similarity scores alone can't separate duplicates from new issues: average precision is
  0.01-0.02 for every retriever on vscode, and at most 0.04 on uv.
- An LLM judge on top of retrieval helps pick the right original. Both LLMs raise top-1
  accuracy (vscode: 0.23 for retrieval alone, 0.35-0.39 with an LLM), though the intervals
  overlap.
- But both LLMs also flag many non-duplicates as duplicates:

| False-duplicate rate (non-duplicates flagged) | vscode | uv |
|---|---:|---:|
| Qwen 3.5 9B (local, Ollama) | 0.48 [0.38, 0.58] | 0.56 [0.46, 0.65] |
| Claude Haiku 5.5 | **0.17** [0.10, 0.24] | **0.34** [0.25, 0.43] |

Haiku cuts false flags roughly threefold on vscode, and the intervals don't overlap on
either repo. At each repo's real duplicate rate, though, only about 10% of Haiku's duplicate
comments would be correct on vscode (4% on uv). A bot built on either model needs a
confidence gate and a shadow period before it posts anything.

**A confidence gate doesn't fix this, as far as the recorded ground truth can tell.** Each
gate's threshold was chosen on a validation sample (the last 3 months before the test period)
and then applied once to test. No gate built on Haiku's stated confidence, embedding
similarity, or a learned combination reaches even 50% precision at the real duplicate rate.
The best any threshold manages on validation is 22% (vscode) and 4% (uv). Haiku's confidence
is also poorly calibrated: on vscode, its calls stated at 0.8-1.0 confidence are correct 22%
of the time. Two caveats:
- **Choosing thresholds on test would have faked a result.** On vscode's test set, a 0.93
  confidence cutoff looks like 83% precision, from 5 correct calls out of 6. Nothing like it
  appears on validation.
- **The recorded duplicates undercount correct calls, so these numbers are a lower bound.**
  About half of Haiku's flags on real duplicates name a different original than the one
  maintainers linked. Many are plausibly the same bug. For example, "Cannot create new
  profile from Python Template" was matched to "Creating a new Profile from Python template
  silently fails", but a maintainer had linked it elsewhere. Maintainers link each duplicate
  to one issue, and the others never get recorded. Measuring true precision needs a
  human-checked sample.

**3. For labels, a classic TF-IDF classifier matches or beats both LLMs.** Micro F1 on the
same 200 issues per repo:

| System | uv | vscode |
|---|---:|---:|
| TF-IDF + logistic regression | **0.74** [0.69, 0.79] | **0.67** [0.62, 0.72] |
| Haiku, 5 example issues | 0.55 [0.49, 0.61] | 0.65 [0.61, 0.70] |
| Qwen, 5 example issues | 0.50 [0.44, 0.55] | 0.65 [0.61, 0.70] |
| Haiku, label descriptions only | 0.46 [0.42, 0.50] | 0.57 [0.53, 0.61] |
| Qwen, label descriptions only | 0.35 [0.31, 0.40] | 0.49 [0.44, 0.53] |

Example issues help both LLMs a lot. Without them, Haiku beats Qwen. TF-IDF runs in
milliseconds, while the LLMs take 1-5 seconds per issue, and Haiku costs an estimated
$0.40-0.80 per 1,000 issues at API prices.

**4. GitHub's own records are unreliable as ground truth, in specific ways.**
- The `canonical` and `duplicate` fields of `MarkedAsDuplicateEvent` don't reliably say
  which issue is the duplicate. On uv, `canonical` was always the issue the event was read
  from. The first collector trusted those fields and found 7 duplicate pairs. Inferring
  direction from close reasons, comments and creation order, reading full issue URLs in
  "Duplicate of" comments, and searching uv's full history found 401.
- Issue forms apply labels automatically. On huggingface/transformers, 1,608 of 1,619 `bug`
  labels were added by the reporter within seconds of opening the issue, so predicting them
  measures which form was picked, not triage.
- Some repos barely record duplicates (transformers: 1 test duplicate; uv: 2-7 a month in
  2026). transformers is collected but not scored on either task.

Haiku rows ran through the Claude Code CLI on a subscription (`haiku-sub`); see
[Running Claude on a subscription](#running-claude-on-a-subscription) for how that differs
from a raw API call.

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

## Live service (Phase 2, in progress)

`src/triage/service/` is the start of a deployed triage bot that runs in **shadow mode**: it
records what it would do and never posts to GitHub.

**Receiving and queueing** (week 1)
- A webhook receiver (FastAPI). It checks GitHub's HMAC signature, drops repeat deliveries
  by delivery ID, enqueues a job and answers 202, recording the delivery and the job in one
  transaction.
- A Postgres job queue with `FOR UPDATE SKIP LOCKED` claiming, so any number of workers can
  run. Failures retry with exponential backoff; after 5 attempts a job is parked as `dead`.
  Jobs held by a crashed worker are requeued.

**Triage pipeline** (week 2). For each new issue the worker:
1. Embeds the issue with bge-small-en-v1.5. It's the benchmark's model, run as BAAI's
   full-precision ONNX export through fastembed (no torch), and it matched the benchmark's
   embeddings exactly. Unchanged text is never re-embedded.
2. Finds the 5 most similar issues with an exact pgvector cosine search, limited to issues
   created before this one: the benchmark's time rule.
3. Suggests labels with the exported TF-IDF model and keeps every label's probability.
4. Stores one open-time prediction per issue, with model, labeler and code versions and
   per-step timings.

Retries and repeat deliveries never add a second prediction. The LLM duplicate judge isn't
wired in yet. Its prompt-building and reply-parsing code is shared with the benchmark
(`build_judge_prompt`, `parse_judge_reply`), and re-running the benchmark after the refactor
hit the LLM cache 100%, so the prompts are byte-identical.

**Parity with the benchmark.** `triage service replay --repo R` rebuilds a repo's benchmark
corpus in a fresh database, replays the benchmark's LLM sample through the real worker, and
compares retrieval, labels and judge prompts with the offline benchmark. Results are in
[results/parity/](results/parity/).

Run it locally (needs Docker):

```bash
uv run triage export-labeler --repo astral-sh/uv            # label model for the worker
TRIAGE_WEBHOOK_SECRET=local-dev-only make up                # Postgres + api on :8000 + worker
TRIAGE_DATABASE_URL=postgresql://triage:triage@localhost:5433/triage \
  uv run triage service backfill --repo astral-sh/uv --from-raw data/raw/astral-sh__uv.jsonl
make service-test                                           # tests against a throwaway database
```

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

## License

MIT. See [LICENSE](LICENSE). The collected GitHub issue data is not part of this repository;
`triage collect` downloads it from the GitHub API, and it remains subject to GitHub's terms
and its authors' rights.
