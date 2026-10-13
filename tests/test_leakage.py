import pandas as pd
import pytest

from triage.systems.duplicates import BM25Retriever, DifflibRetriever


@pytest.mark.parametrize("retriever_cls", [BM25Retriever, DifflibRetriever])
def test_search_only_returns_earlier_issues(fixture_repo, retriever_cls):
    _, repo_dir, _ = fixture_repo
    issues = pd.read_parquet(repo_dir / "issues.parquet")
    created = issues.set_index("number")["created_at"]
    r = retriever_cls()
    r.index(issues)
    for n in issues["number"].sample(60, random_state=0):
        for hit, _ in r.search(int(n), k=50):
            assert created[hit] < created[n], f"{hit} is not earlier than query {n}"


def test_first_issue_has_no_candidates(fixture_repo):
    _, repo_dir, _ = fixture_repo
    issues = pd.read_parquet(repo_dir / "issues.parquet")
    r = BM25Retriever()
    r.index(issues)
    first = int(issues.sort_values("created_at")["number"].iloc[0])
    assert r.search(first) == []
