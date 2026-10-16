"""Pull issues, labels and duplicate links for one repository into raw JSONL.

Issues are walked newest-first and collection stops once it passes `since`, so a window is
cheap even on very large repositories.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from triage.github_client import GitHubClient

log = logging.getLogger(__name__)

# The issue fields the collector, the live poller and the outcome tracker all fetch.
ISSUE_FIELDS = """
        number title body url createdAt closedAt state stateReason
        author { login }
        labels(first: 30) { nodes { name } }
        comments(last: 5) { nodes { body author { login } } }
        labelEvents: timelineItems(last: 30, itemTypes: [LABELED_EVENT]) {
          nodes { ... on LabeledEvent { createdAt label { name } actor { login } } }
        }
        timelineItems(
          first: 10
          itemTypes: [MARKED_AS_DUPLICATE_EVENT, UNMARKED_AS_DUPLICATE_EVENT]
        ) {
          nodes {
            __typename
            ... on MarkedAsDuplicateEvent {
              createdAt
              canonical { ... on Issue { number repository { nameWithOwner } } }
              duplicate { ... on Issue { number repository { nameWithOwner } } }
            }
            ... on UnmarkedAsDuplicateEvent {
              createdAt
              canonical { ... on Issue { number repository { nameWithOwner } } }
              duplicate { ... on Issue { number repository { nameWithOwner } } }
            }
          }
        }
"""

# Built from ISSUE_FIELDS; the text must not change (the page cache is keyed on it, and a
# test pins its hash).
ISSUES_QUERY = (
    """
query($owner: String!, $name: String!, $pageSize: Int!, $after: String) {
  rateLimit { cost remaining resetAt }
  repository(owner: $owner, name: $name) {
    issues(first: $pageSize, after: $after, orderBy: {field: CREATED_AT, direction: DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
"""
    + ISSUE_FIELDS[1:]
    + """      }
    }
  }
}
"""
)

LABELS_QUERY = """
query($owner: String!, $name: String!, $after: String) {
  rateLimit { cost remaining resetAt }
  repository(owner: $owner, name: $name) {
    labels(first: 100, after: $after) {
      pageInfo { hasNextPage endCursor }
      nodes { name description }
    }
  }
}
"""

# "Duplicate of #123" is GitHub's own keyword; maintainers also paste the full issue URL, and
# bots write "dupe of #123". Group 1 is an optional owner/name, group 2 the issue number.
DUP_COMMENT_RE = re.compile(
    r"\b(?:duplicate|dup|dupe)\s+of\s+"
    r"(?:#|https?://github\.com/([\w.-]+/[\w.-]+)/issues/)(\d+)\b",
    re.IGNORECASE,
)


def iter_issue_nodes(
    client: GitHubClient, repo: str, since: datetime, page_size: int = 50
) -> Iterator[dict[str, Any]]:
    owner, name = repo.split("/")
    after = None
    while True:
        data = client.query(
            ISSUES_QUERY, {"owner": owner, "name": name, "pageSize": page_size, "after": after}
        )
        issues = data["repository"]["issues"]
        for node in issues["nodes"]:
            if _parse_ts(node["createdAt"]) < since:
                return
            yield node
        if not issues["pageInfo"]["hasNextPage"]:
            return
        after = issues["pageInfo"]["endCursor"]


def fetch_labels(client: GitHubClient, repo: str) -> list[dict[str, str]]:
    owner, name = repo.split("/")
    out, after = [], None
    while True:
        data = client.query(LABELS_QUERY, {"owner": owner, "name": name, "after": after})
        labels = data["repository"]["labels"]
        out.extend(
            {"name": n["name"], "description": n["description"] or ""} for n in labels["nodes"]
        )
        if not labels["pageInfo"]["hasNextPage"]:
            return out
        after = labels["pageInfo"]["endCursor"]


def normalize_issue(node: dict[str, Any], repo: str) -> dict[str, Any]:
    return {
        "repo": repo,
        "number": node["number"],
        "title": node["title"] or "",
        "body": node["body"] or "",
        "url": node["url"],
        "created_at": node["createdAt"],
        "closed_at": node["closedAt"],
        "state": node["state"],
        "state_reason": node["stateReason"],
        "author": (node.get("author") or {}).get("login"),
        "labels": [lbl["name"] for lbl in node["labels"]["nodes"]],
        "dup_links": duplicate_links(node, repo),
        "dup_mentions": duplicate_mentions(node, repo),
        # When each label was added, so the build can tell issue-template labels (applied
        # at creation) from triage decisions.
        "label_events": [
            {
                "label": ev["label"]["name"],
                "at": ev["createdAt"],
                "actor": (ev.get("actor") or {}).get("login"),
            }
            for ev in node["labelEvents"]["nodes"]
            if ev.get("label")
        ],
    }


def duplicate_links(node: dict[str, Any], repo: str) -> list[dict[str, Any]]:
    """Every same-repo issue this issue is linked to as a duplicate pair.

    Timeline links are *undirected*. The API's `canonical` / `duplicate` fields cannot be
    trusted for direction: on astral-sh/uv, `canonical` was always the issue whose timeline
    the event came from, whether that issue was the original or the duplicate. The build step
    orients each pair using close reasons and creation order (see dataset/build.py).

    Comment links are directed: "Duplicate of #N" on this issue means this one is the
    duplicate. They only count when the issue also carries a duplicate signal (closed as
    duplicate, or a duplicate label), which filters out "possible duplicate of #N" guesses.
    """
    me = node["number"]
    linked: dict[int, None] = {}  # ordered set of linked issue numbers
    events = sorted(node["timelineItems"]["nodes"], key=lambda e: e.get("createdAt") or "")
    for ev in events:
        sides = [ev.get("canonical") or {}, ev.get("duplicate") or {}]
        if any("number" not in side for side in sides):
            continue
        if any(_repo_of(side, repo).lower() != repo.lower() for side in sides):
            continue  # cross-repo: the other issue is not in our corpus
        numbers = {side["number"] for side in sides}
        if me not in numbers or len(numbers) != 2:
            continue
        (other,) = numbers - {me}
        if ev["__typename"] == "MarkedAsDuplicateEvent":
            linked[other] = None
        elif ev["__typename"] == "UnmarkedAsDuplicateEvent":
            linked.pop(other, None)
    links = [{"other": n, "source": "timeline", "directed": False} for n in linked]

    if _has_dup_signal(node):
        for comment in reversed(node["comments"]["nodes"]):
            for m in DUP_COMMENT_RE.finditer(comment["body"] or ""):
                target_repo, number = m.group(1), int(m.group(2))
                if target_repo and target_repo.lower() != repo.lower():
                    continue
                if number != me:
                    links.append({"other": number, "source": "comment", "directed": True})
    return links


def duplicate_mentions(node: dict[str, Any], repo: str) -> list[int]:
    """Issue numbers referenced in comments of an issue closed as (or labeled) duplicate.

    Maintainers often close duplicates with "See #N" or "tracked in #N" instead of the
    "Duplicate of" keyword. The build step uses these only when nothing explicit links the
    issue and exactly one referenced issue is an older issue in the corpus.
    """
    if not _has_dup_signal(node):
        return []
    owner_repo = re.escape(repo)
    ref_re = re.compile(
        rf"(?:https?://github\.com/{owner_repo}/issues/|(?<![\w/]){owner_repo}#|(?<![\w/&])#)(\d+)",
        re.IGNORECASE,
    )
    found: dict[int, None] = {}
    for comment in node["comments"]["nodes"]:
        for m in ref_re.finditer(comment["body"] or ""):
            if int(m.group(1)) != node["number"]:
                found[int(m.group(1))] = None
    return list(found)


def _has_dup_signal(node: dict[str, Any]) -> bool:
    return node["stateReason"] == "DUPLICATE" or any(
        "duplicate" in lbl["name"].lower() for lbl in node["labels"]["nodes"]
    )


def _repo_of(side: dict[str, Any], default: str) -> str:
    return (side.get("repository") or {}).get("nameWithOwner", default)


def collect_repo(
    client: GitHubClient,
    repo: str,
    since: datetime,
    out_dir: Path,
    limit: int | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = repo.replace("/", "__")
    issues_path = out_dir / f"{slug}.jsonl"
    labels_path = out_dir / f"{slug}.labels.json"

    labels_path.write_text(json.dumps(fetch_labels(client, repo), indent=2))

    n = n_dup = 0
    with issues_path.open("w") as f:
        for node in iter_issue_nodes(client, repo, since):
            rec = normalize_issue(node, repo)
            f.write(json.dumps(rec) + "\n")
            n += 1
            n_dup += bool(rec["dup_links"])
            if n % 500 == 0:
                log.info("%s: %d issues (%d with duplicate links)", repo, n, n_dup)
            if limit and n >= limit:
                break
    log.info("%s: wrote %d issues (%d with duplicate links) to %s", repo, n, n_dup, issues_path)
    return issues_path


def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))
