import json
from datetime import UTC, datetime

import httpx

from triage.dataset.collect import duplicate_mentions, iter_issue_nodes, normalize_issue
from triage.github_client import GitHubClient

REPO = "acme/widgets"


def node(
    number, created="2025-05-01T00:00:00Z", timeline=(), labels=(), comments=(), state_reason=None
):
    return {
        "number": number,
        "title": f"t{number}",
        "body": "b",
        "url": "u",
        "createdAt": created,
        "closedAt": None,
        "state": "OPEN",
        "stateReason": state_reason,
        "author": {"login": "x"},
        "labels": {"nodes": [{"name": n} for n in labels]},
        "comments": {"nodes": [{"body": c, "author": {"login": "m"}} for c in comments]},
        "timelineItems": {"nodes": list(timeline)},
        "labelEvents": {
            "nodes": [
                {"createdAt": created, "label": {"name": n}, "actor": {"login": "form"}}
                for n in labels
            ]
        },
    }


def marked(dup, canon, at="2025-05-02T00:00:00Z", repo=REPO):
    return {
        "__typename": "MarkedAsDuplicateEvent",
        "createdAt": at,
        "canonical": {"number": canon, "repository": {"nameWithOwner": repo}},
        "duplicate": {"number": dup, "repository": {"nameWithOwner": REPO}},
    }


def unmarked(dup, canon, at="2025-05-03T00:00:00Z"):
    return {
        "__typename": "UnmarkedAsDuplicateEvent",
        "createdAt": at,
        "canonical": {"number": canon, "repository": {"nameWithOwner": REPO}},
        "duplicate": {"number": dup, "repository": {"nameWithOwner": REPO}},
    }


def others(rec):
    return [(link["other"], link["source"], link["directed"]) for link in rec["dup_links"]]


def test_timeline_link_is_undirected_from_either_side():
    # Same event seen from the duplicate's and from the original's timeline.
    assert others(normalize_issue(node(10, timeline=[marked(10, 3)]), REPO)) == [
        (3, "timeline", False)
    ]
    assert others(normalize_issue(node(3, timeline=[marked(10, 3)]), REPO)) == [
        (10, "timeline", False)
    ]


def test_reversed_fields_seen_on_uv_still_link():
    # On astral-sh/uv, an issue saying "Duplicate of #13878" carried an event whose
    # `canonical` was itself and `duplicate` was #13878. The link must survive.
    rec = normalize_issue(node(14959, timeline=[marked(dup=13878, canon=14959)]), REPO)
    assert others(rec) == [(13878, "timeline", False)]


def test_unmarked_cancels_mark():
    rec = normalize_issue(node(10, timeline=[unmarked(10, 3), marked(10, 3)]), REPO)
    assert rec["dup_links"] == []  # sorted by time: marked on 05-02, unmarked on 05-03


def test_cross_repo_duplicate_dropped():
    rec = normalize_issue(node(10, timeline=[marked(10, 3, repo="other/repo")]), REPO)
    assert rec["dup_links"] == []


def test_comment_link_is_directed_and_needs_duplicate_signal():
    with_label = node(10, labels=["duplicate"], comments=["Duplicate of #7"])
    assert others(normalize_issue(with_label, REPO)) == [(7, "comment", True)]
    no_signal = node(10, comments=["Duplicate of #7"])
    assert normalize_issue(no_signal, REPO)["dup_links"] == []


def test_comment_link_accepts_issue_urls_from_same_repo_only():
    same = node(
        10,
        state_reason="DUPLICATE",
        comments=[f"Duplicate of https://github.com/{REPO}/issues/7"],
    )
    assert others(normalize_issue(same, REPO)) == [(7, "comment", True)]
    other = node(
        10,
        state_reason="DUPLICATE",
        comments=["Duplicate of https://github.com/astral-sh/ruff/issues/7"],
    )
    assert normalize_issue(other, REPO)["dup_links"] == []


class FakeClient:
    def __init__(self, pages):
        self.pages = pages
        self.calls = 0

    def query(self, query, variables):
        page = self.pages[self.calls]
        self.calls += 1
        return page


def page(nodes, has_next):
    return {
        "repository": {
            "issues": {"pageInfo": {"hasNextPage": has_next, "endCursor": "c"}, "nodes": nodes}
        }
    }


def test_iteration_stops_before_since():
    pages = [
        page([node(5, "2025-06-01T00:00:00Z"), node(4, "2025-05-01T00:00:00Z")], True),
        page([node(3, "2025-03-01T00:00:00Z"), node(2, "2025-02-01T00:00:00Z")], True),
        page([node(1, "2025-01-01T00:00:00Z")], False),
    ]
    client = FakeClient(pages)
    got = [n["number"] for n in iter_issue_nodes(client, REPO, datetime(2025, 4, 1, tzinfo=UTC))]
    assert got == [5, 4]
    assert client.calls == 2  # never fetched the third page


def test_client_retries_then_caches(tmp_path):
    responses = [httpx.Response(502), httpx.Response(200, json={"data": {"ok": 1}})]
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return responses[len(seen) - 1]

    client = GitHubClient(
        "t", cache_dir=tmp_path, transport=httpx.MockTransport(handler), sleep=lambda s: None
    )
    assert client.query("q", {"a": 1}) == {"ok": 1}
    assert client.query("q", {"a": 1}) == {"ok": 1}  # served from cache
    assert len(seen) == 2


def test_mentions_collected_only_for_duplicate_closed_issues():
    comments = [
        "See https://github.com/acme/widgets/issues/20792#issuecomment-1",
        "This shares the root cause of #8441, not &#39; an entity",
        "Unrelated: astral-sh/ruff#5 and acme/widgets#77",
    ]
    closed = node(30000, state_reason="DUPLICATE", comments=comments)
    assert duplicate_mentions(closed, REPO) == [20792, 8441, 77]
    assert duplicate_mentions(node(30000, comments=comments), REPO) == []


def test_label_events_recorded():
    rec = normalize_issue(node(10, labels=["bug"]), REPO)
    assert rec["label_events"] == [{"label": "bug", "at": "2025-05-01T00:00:00Z", "actor": "form"}]


def test_collector_query_text_is_unchanged():
    # The page cache is keyed on this text: ISSUE_FIELDS was factored out of it without
    # changing a byte. If this fails, every cached collection page is invalidated.
    import hashlib

    from triage.dataset.collect import ISSUES_QUERY

    assert (
        hashlib.sha256(ISSUES_QUERY.encode()).hexdigest()
        == "e9d96721a62da33963a5410da9aa921d1592b605c8aca7e6b425df9e2560abf4"
    )


def test_not_found_errors_tolerated_only_when_asked(tmp_path):
    import pytest

    from triage.github_client import GitHubError

    body = {
        "data": {"repository": {"i1": {"number": 1}, "i2": None}},
        "errors": [{"type": "NOT_FOUND", "message": "Could not resolve to an Issue"}],
    }

    def handler(request):
        return httpx.Response(200, json=body)

    strict = GitHubClient("t", cache_dir=None, transport=httpx.MockTransport(handler))
    with pytest.raises(GitHubError):
        strict.query("q", {})
    lenient = GitHubClient(
        "t", cache_dir=None, transport=httpx.MockTransport(handler), tolerate_not_found=True
    )
    assert lenient.query("q", {})["repository"]["i2"] is None
