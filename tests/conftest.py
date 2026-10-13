from pathlib import Path

import pytest

from triage.config import load_repos
from triage.dataset.build import build_repo

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixture_repo(tmp_path_factory):
    """Build the synthetic repo once per test session; returns (config, repo_dir, meta)."""
    cfg = load_repos(FIXTURES / "repos.yaml")[0]
    out = tmp_path_factory.mktemp("processed")
    meta = build_repo(cfg, FIXTURES / "raw", out)
    return cfg, out / cfg.slug, meta
