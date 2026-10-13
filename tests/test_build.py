import pandas as pd

from triage.dataset.build import build_dup_pairs, build_label_vocab, clean_body


def test_clean_body_strips_template_comments():
    assert clean_body("<!-- template\nstuff -->Hello   world\n\n\n\nbye") == "Hello world\n\nbye"


def test_vocab_uses_train_only_and_excludes_process_labels():
    train = pd.DataFrame({"labels": [["bug", "duplicate"]] * 6 + [["docs"]] * 6})
    vocab = build_label_vocab(train, top_k=10, min_count=5)
    assert vocab == ["bug", "docs"]


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
