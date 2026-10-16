import pandas as pd

from triage.dataset.build import (
    build_dup_pairs,
    build_label_vocab,
    clean_body,
    labels_at_creation,
    template_labels,
)


def test_clean_body_strips_template_comments():
    assert clean_body("<!-- template\nstuff -->Hello   world\n\n\n\nbye") == "Hello world\n\nbye"


def test_vocab_uses_train_only_and_excludes_process_labels():
    train = pd.DataFrame({"labels": [["bug", "duplicate"]] * 6 + [["docs"]] * 6})
    vocab = build_label_vocab(train, top_k=10, min_count=5)
    assert vocab == ["bug", "docs"]


def test_workflow_labels_excluded():
    names = [
        "verified",
        "insiders-released",
        "verification-needed",
        "testplan-item",
        "under-discussion",
        "ai-translated",
        "needs-mre",
        "WIP",
        "confirmation-pending",
        "chat",
        "area:cli",
    ]
    train = pd.DataFrame({"labels": [names] * 5})
    assert build_label_vocab(train, top_k=20, min_count=5) == ["area:cli", "chat"]


def test_labels_at_creation_uses_latest_event():
    created = pd.Timestamp("2025-01-01T00:00:00Z")
    events = [
        {"label": "bug", "at": "2025-01-01T00:00:01Z"},  # from the issue form
        {"label": "area", "at": "2025-01-01T00:00:02Z"},  # form, removed, re-added later
        {"label": "area", "at": "2025-01-03T00:00:00Z"},
    ]
    got = labels_at_creation(["bug", "area", "no-event"], events, created, window_s=60)
    assert got == ["bug"]


def test_template_labels_excluded_by_share():
    train = pd.DataFrame(
        {
            "labels": [["bug", "cli"]] * 10,
            "creation_labels": [["bug"]] * 8 + [["bug", "cli"]] * 2,
        }
    )
    templated = template_labels(train, max_share=0.5)
    assert templated == {"bug": 1.0}
    assert build_label_vocab(train, 10, 5, exclude=set(templated)) == ["cli"]


def test_split_and_vocab(fixture_repo):
    cfg, repo_dir, meta = fixture_repo
    issues = pd.read_parquet(repo_dir / "issues.parquet")
    train = issues[issues["split"] == "train"]
    test = issues[issues["split"] == "test"]
    assert (train["created_at"] < cfg.cutoff).all()
    assert (test["created_at"] >= cfg.cutoff).all()
    names = [v["name"] for v in meta["label_vocab"]]
    assert "duplicate" not in names
    assert {"bug", "area: cli"} <= set(names)


def test_dup_pairs_point_backwards_in_time(fixture_repo):
    _, repo_dir, meta = fixture_repo
    issues = pd.read_parquet(repo_dir / "issues.parquet").set_index("number")
    pairs = pd.read_parquet(repo_dir / "dup_pairs.parquet")
    assert len(pairs) > 0
    for d, c in zip(pairs["duplicate"], pairs["canonical"], strict=True):
        assert issues.loc[c, "created_at"] < issues.loc[d, "created_at"]
    assert meta["n_dup_pairs"]["test"] > 0


def _issues(rows, mentions=None):
    df = pd.DataFrame(rows, columns=["number", "created_at", "has_dup_signal", "dup_links"])
    df["dup_mentions"] = [(mentions or {}).get(n, []) for n in df["number"]]
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["split"] = "train"
    return df


def link(other, source="timeline", directed=False):
    return {"other": other, "source": source, "directed": directed}


def test_orientation_rules():
    issues = _issues(
        [
            # 1 <-> 2: link only on the original's timeline; 2 closed as duplicate.
            (1, "2025-01-01", False, [link(2)]),
            (2, "2025-01-02", True, []),
            # 3 <-> 4: no close-reason signal on either side; newer one is the duplicate.
            (3, "2025-01-03", False, []),
            (4, "2025-01-04", False, [link(3)]),
            # 6 says "Duplicate of #5" in a comment.
            (5, "2025-01-05", False, []),
            (6, "2025-01-06", True, [link(5, "comment", True), link(5)]),
            # 7 was closed as a duplicate of the *newer* 8: not findable, dropped.
            (7, "2025-01-07", True, [link(8)]),
            (8, "2025-01-08", False, []),
            # link to an issue outside the window is ignored
            (9, "2025-01-09", True, [link(12345)]),
        ]
    )
    pairs, stats = build_dup_pairs(issues)
    got = {
        (d, c, how)
        for d, c, how in zip(
            pairs["duplicate"], pairs["canonical"], pairs["oriented_by"], strict=True
        )
    }
    assert got == {(2, 1, "close_reason"), (4, 3, "created_order"), (6, 5, "comment")}
    assert stats["dropped_canonical_not_older"] == 1
    assert pairs.loc[pairs["duplicate"] == 6, "source"].item() == "comment+timeline"


def test_mention_fallback_only_when_unambiguous():
    issues = _issues(
        [
            (1, "2025-01-01", False, []),
            (2, "2025-01-02", False, []),
            (3, "2025-01-03", True, []),  # mentions 1 only -> pair (3, 1)
            (4, "2025-01-04", True, []),  # mentions 1 and 2 -> ambiguous, dropped
            (5, "2025-01-05", True, []),  # mentions newer 6 and outside 999 -> nothing
            (6, "2025-01-06", True, [link(2, "comment", True)]),  # explicit link wins
        ],
        mentions={3: [1], 4: [1, 2], 5: [6, 999], 6: [1]},
    )
    pairs, stats = build_dup_pairs(issues)
    got = {
        (d, c, how)
        for d, c, how in zip(
            pairs["duplicate"], pairs["canonical"], pairs["oriented_by"], strict=True
        )
    }
    assert got == {(3, 1, "mention"), (6, 2, "comment")}
    assert stats["mention_ambiguous_or_none"] == 2


def test_validation_is_the_end_of_train(fixture_repo):
    cfg, repo_dir, meta = fixture_repo
    issues = pd.read_parquet(repo_dir / "issues.parquet")
    val = issues[issues["is_validation"]]
    assert len(val) > 0
    assert (val["split"] == "train").all()
    start = cfg.cutoff - pd.DateOffset(months=cfg.validation_months)
    assert (val["created_at"] >= start).all()
    assert not issues.loc[issues["created_at"] < start, "is_validation"].any()
    assert meta["validation"]["n_issues"] == len(val)


def test_issue_text_matches_build_output(fixture_repo):
    from triage.dataset.build import issue_text

    _, repo_dir, _ = fixture_repo
    row = pd.read_parquet(repo_dir / "issues.parquet").iloc[0]
    assert row["text"] == issue_text(row["title"], row["body_clean"])
    assert issue_text("T", "B") == "T\n\nB"


def test_resolve_duplicate_uses_the_benchmark_rules():
    from triage.dataset.build import IssueFacts, resolve_duplicate

    t = lambda d: pd.Timestamp(f"2025-01-{d:02d}", tz="UTC")  # noqa: E731
    facts = {
        1: IssueFacts(t(1), False),
        2: IssueFacts(t(2), False),
        3: IssueFacts(t(3), True),  # closed as duplicate
        4: IssueFacts(t(4), True),
        5: IssueFacts(t(5), False),
    }
    # Timeline link seen from the duplicate: close reason decides.
    assert resolve_duplicate(3, [link(1)], [], facts) == (1, "close_reason")
    # Explicit comment wins, even pointing at a later-numbered (but older) issue.
    assert resolve_duplicate(4, [link(2, "comment", True)], [], facts) == (2, "comment")
    # Neither side flagged: the newer issue is the duplicate, so 1 is not a duplicate of 2.
    assert resolve_duplicate(2, [link(1)], [], facts) == (1, "created_order")
    assert resolve_duplicate(1, [link(2)], [], facts) is None
    # No links: a single older mention resolves it; several don't.
    assert resolve_duplicate(4, [], [2], facts) == (2, "mention")
    assert resolve_duplicate(4, [], [1, 2], facts) is None
    # Several originals: the earliest wins.
    assert resolve_duplicate(4, [link(2), link(1)], [], facts) == (1, "close_reason")
    # Links to unknown issues are ignored.
    assert resolve_duplicate(3, [link(999)], [], facts) is None
