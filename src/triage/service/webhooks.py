"""GitHub webhook handling that doesn't depend on the web framework: signature checks and
deciding which events become jobs."""

from __future__ import annotations

import hashlib
import hmac
from typing import Any

# Issue events that can change what the bot would say. Comments matter because maintainers
# mark duplicates with "Duplicate of #N" comments.
ISSUE_ACTIONS = {"opened", "edited", "closed", "reopened", "labeled", "unlabeled", "transferred"}
COMMENT_ACTIONS = {"created", "edited"}


def verify_signature(secret: str, body: bytes, header: str | None) -> bool:
    """Check GitHub's X-Hub-Signature-256 header (HMAC-SHA256 of the raw body).

    Uses a constant-time comparison so the check doesn't leak timing information.
    """
    if not header or not header.startswith("sha256="):
        return False
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header.removeprefix("sha256="))


def job_for_event(event: str, payload: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """The (job kind, job payload) for an event, or None if the service ignores it.

    The job payload keeps only what the worker needs, not the whole webhook body.
    """
    issue = payload.get("issue")
    repo = (payload.get("repository") or {}).get("full_name")
    if not issue or not repo or "pull_request" in issue:
        return None  # pull requests arrive as issues too; the bot only triages issues
    action = payload.get("action")
    if event == "issues" and action in ISSUE_ACTIONS:
        return "github.issue", {"repo": repo, "action": action, "issue": _issue(issue)}
    if event == "issue_comment" and action in COMMENT_ACTIONS:
        comment = payload.get("comment") or {}
        return "github.issue", {
            "repo": repo,
            "action": f"comment_{action}",
            "issue": _issue(issue),
            "comment": {
                "id": comment.get("id"),
                "body": comment.get("body") or "",
                "author": (comment.get("user") or {}).get("login"),
                "created_at": comment.get("created_at"),
            },
        }
    return None


def _issue(issue: dict[str, Any]) -> dict[str, Any]:
    return {
        "number": issue["number"],
        "title": issue.get("title") or "",
        "body": issue.get("body") or "",
        "state": issue.get("state") or "open",
        "state_reason": issue.get("state_reason"),
        "labels": [lbl["name"] for lbl in issue.get("labels") or [] if "name" in lbl],
        "author": (issue.get("user") or {}).get("login"),
        "url": issue.get("html_url"),
        "created_at": issue["created_at"],
        "updated_at": issue.get("updated_at") or issue["created_at"],
    }
