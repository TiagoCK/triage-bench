"""Turn raw JSONL into the benchmark tables for one repository.

Outputs, under `<out_dir>/<repo slug>/`:
  issues.parquet     every issue in the window, with cleaned text, split and vocab labels
  dup_pairs.parquet  (duplicate, canonical) pairs where both issues are in the corpus
  meta.json          label vocabulary with descriptions, counts and the config used

Leakage rules (enforced by tests/test_build.py):
  - The label vocabulary is chosen from train issues only.
  - A duplicate pair is kept only if the canonical issue was created before the duplicate,
    so a system searching "issues that existed when this one was opened" can find it.

Duplicate direction: GitHub timeline links are stored undirected by the collector (the API's
canonical/duplicate fields are unreliable). Each pair is oriented by, in order:
  1. a "Duplicate of #N" comment on one side (explicit),
  2. close reason / label: if exactly one side was closed as or labeled duplicate, it is the
     duplicate,
  3. creation order: the newer issue is the duplicate.
Issues closed as duplicate with no link at all fall back to their comments: if exactly one
referenced issue is an older issue in the corpus, it becomes a pair with source "mention".
These are less certain than explicit links, so they are tagged and can be excluded.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from triage.config import RepoConfig

BODY_MAX_CHARS = 2000

# Process labels say what happened to an issue, not what it is about, so they are not
# prediction targets.
EXCLUDED_LABEL_PATTERNS = [
    r"duplicate",
    r"stale",
    r"triage",
    r"wont.?fix",
    r"won't fix",
    r"invalid",
    r"spam",
    r"locked",
    r"info.?needed",
    r"needs.?(more.?)?info",
    r"awaiting",
    r"\*",  # vscode-style "*duplicate", "*caused-by-extension" resolution labels
]
_EXCLUDED_RE = re.compile("|".join(EXCLUDED_LABEL_PATTERNS), re.IGNORECASE)

_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_WS_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def clean_body(body: str) -> str:
    """Strip issue-template comments and squeeze whitespace; truncate long bodies."""
    text = _HTML_COMMENT_RE.sub("", body or "")
    text = _WS_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text).strip()
    return text[:BODY_MAX_CHARS]


def load_raw(raw_dir: Path, slug: str) -> tuple[pd.DataFrame, list[dict]]:
    issues = pd.read_json(raw_dir / f"{slug}.jsonl", lines=True)
    if "dup_mentions" not in issues.columns:
        raise ValueError(
            f"{slug}.jsonl predates undirected duplicate links; re-run `triage collect` "
            "(cached pages make this cheap)"
        )
    labels_path = raw_dir / f"{slug}.labels.json"
    label_defs = json.loads(labels_path.read_text()) if labels_path.exists() else []
    return issues, label_defs


def build_label_vocab(train: pd.DataFrame, top_k: int, min_count: int) -> list[str]:
    counts = Counter(lbl for labels in train["labels"] for lbl in labels)
    eligible = [
        (lbl, c) for lbl, c in counts.items() if c >= min_count and not _EXCLUDED_RE.search(lbl)
    ]
    eligible.sort(key=lambda x: (-x[1], x[0]))
    return [lbl for lbl, _ in eligible[:top_k]]


@dataclass
class _Link:
    sources: set[str] = field(default_factory=set)
    directed: tuple[int, int] | None = None  # (duplicate, canonical) when a comment says so


def build_dup_pairs(issues: pd.DataFrame) -> tuple[pd.DataFrame, Counter]:
    """Orient every duplicate link into a (duplicate, canonical) pair; see module docstring.

    Returns the pairs plus counts of how each pair was oriented or why it was dropped.
    """
    info = issues.set_index("number")
    links: dict[frozenset[int], _Link] = {}
    for me, raw_links in zip(issues["number"], issues["dup_links"], strict=True):
        for link in raw_links:
            other = int(link["other"])
            if other not in info.index:
                continue  # outside the collected window
            entry = links.setdefault(frozenset((int(me), other)), _Link())
            entry.sources.add(link["source"])
            if link["directed"]:
                entry.directed = (int(me), other)

    stats: Counter = Counter()
    rows = []
    for key, entry in links.items():
        a, b = sorted(key)
        if entry.directed:
            dup, canon = entry.directed
            how = "comment"
        elif info.at[a, "has_dup_signal"] != info.at[b, "has_dup_signal"]:
            dup = a if info.at[a, "has_dup_signal"] else b
            canon = b if dup == a else a
            how = "close_reason"
        else:
            dup, canon = (b, a) if info.at[b, "created_at"] > info.at[a, "created_at"] else (a, b)
            how = "created_order"
        if info.at[canon, "created_at"] >= info.at[dup, "created_at"]:
            stats["dropped_canonical_not_older"] += 1
            continue
        stats[f"oriented_by_{how}"] += 1
        rows.append(
            {
                "duplicate": dup,
                "canonical": canon,
                "created_at": info.at[dup, "created_at"],
                "split": info.at[dup, "split"],
                "source": "+".join(sorted(entry.sources)),
                "oriented_by": how,
            }
        )
    # Fallback for issues closed as duplicate that no explicit link resolved.
    resolved = {r["duplicate"] for r in rows}
    flagged = issues[issues["has_dup_signal"] & ~issues["number"].isin(resolved)]
    for me, mentions in zip(flagged["number"], flagged["dup_mentions"], strict=True):
        older = [
            int(m)
            for m in mentions
            if int(m) in info.index and info.at[int(m), "created_at"] < info.at[me, "created_at"]
        ]
        if len(older) != 1:
            stats["mention_ambiguous_or_none"] += 1
            continue
        stats["oriented_by_mention"] += 1
        rows.append(
            {
                "duplicate": int(me),
                "canonical": older[0],
                "created_at": info.at[me, "created_at"],
                "split": info.at[me, "split"],
                "source": "mention",
                "oriented_by": "mention",
            }
        )

    cols = ["duplicate", "canonical", "created_at", "split", "source", "oriented_by"]
    pairs = pd.DataFrame(rows, columns=cols).sort_values(["created_at", "canonical"])
    return pairs.reset_index(drop=True), stats


def build_repo(cfg: RepoConfig, raw_dir: Path, out_dir: Path) -> dict:
    issues, label_defs = load_raw(raw_dir, cfg.slug)
    issues["created_at"] = pd.to_datetime(issues["created_at"], utc=True)
    issues = issues[(issues["created_at"] >= cfg.since) & (issues["created_at"] < cfg.until)]
    issues = issues.sort_values("created_at").drop_duplicates("number").reset_index(drop=True)

    issues["split"] = (issues["created_at"] >= cfg.cutoff).map({True: "test", False: "train"})
    issues["body_clean"] = issues["body"].map(clean_body)
    issues["text"] = issues["title"] + "\n\n" + issues["body_clean"]
    issues["has_dup_signal"] = (issues["state_reason"] == "DUPLICATE") | issues["labels"].map(
        lambda ls: any("duplicate" in lbl.lower() for lbl in ls)
    )

    pairs, pair_stats = build_dup_pairs(issues)
    # Earliest canonical per duplicate, for convenience; evaluation uses all pairs.
    first_canon = pairs.drop_duplicates("duplicate").set_index("duplicate")["canonical"]
    issues["dup_of"] = issues["number"].map(first_canon).astype("Int64")
    # Anything a maintainer flagged as a duplicate, even when we could not resolve the
    # canonical issue. Used to keep such issues out of the "not a duplicate" negatives.
    issues["is_dup_flagged"] = issues["has_dup_signal"] | issues["dup_of"].notna()

    vocab = build_label_vocab(
        issues[issues["split"] == "train"], cfg.top_k_labels, cfg.min_label_count
    )
    vocab_set = set(vocab)
    issues["target_labels"] = issues["labels"].map(lambda ls: [x for x in ls if x in vocab_set])

    repo_dir = out_dir / cfg.slug
    repo_dir.mkdir(parents=True, exist_ok=True)
    keep = [
        "number",
        "title",
        "body_clean",
        "text",
        "url",
        "created_at",
        "split",
        "state",
        "state_reason",
        "labels",
        "target_labels",
        "dup_of",
        "is_dup_flagged",
    ]
    issues[keep].to_parquet(repo_dir / "issues.parquet", index=False)
    pairs.to_parquet(repo_dir / "dup_pairs.parquet", index=False)

    descriptions = {d["name"]: d["description"] for d in label_defs}
    label_counts = Counter(lbl for ls in issues["target_labels"] for lbl in ls)
    meta = {
        "repo": cfg.name,
        "since": cfg.since.isoformat(),
        "until": cfg.until.isoformat(),
        "cutoff": cfg.cutoff.isoformat(),
        "n_issues": {s: int((issues["split"] == s).sum()) for s in ("train", "test")},
        "n_labeled": {
            s: int(((issues["split"] == s) & (issues["target_labels"].map(len) > 0)).sum())
            for s in ("train", "test")
        },
        "n_dup_pairs": {s: int((pairs["split"] == s).sum()) for s in ("train", "test")},
        "n_duplicates": {
            s: int(pairs.loc[pairs["split"] == s, "duplicate"].nunique()) for s in ("train", "test")
        },
        "n_closed_as_duplicate": int(issues["has_dup_signal"].sum()),
        "dup_pair_stats": dict(pair_stats),
        "label_vocab": [
            {"name": lbl, "description": descriptions.get(lbl, ""), "count": label_counts[lbl]}
            for lbl in vocab
        ],
    }
    (repo_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta
