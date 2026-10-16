"""Service configuration, read from environment variables prefixed `TRIAGE_`.

Secrets (the webhook secret) come only from the environment, never from files in the repo.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRIAGE_", env_file=".env", extra="ignore")

    database_url: str = "postgresql://triage:triage@localhost:5432/triage"
    # GitHub App webhook secret. Without it the webhook endpoint refuses every request.
    webhook_secret: SecretStr | None = None
    # Only shadow mode exists: the service records what it would do and never posts.
    mode: Literal["shadow"] = "shadow"

    # GitHub caps webhook payloads at 25 MB; issue events are far smaller.
    max_webhook_bytes: int = 5_000_000

    worker_poll_interval_s: float = 1.0
    job_max_attempts: int = 5
    # Retry delay is backoff_base * 2^(attempt - 1), capped at backoff_max.
    job_backoff_base_s: float = 5.0
    job_backoff_max_s: float = 600.0
    # A job locked longer than this is assumed to belong to a dead worker and is requeued.
    job_lock_timeout_s: float = 300.0

    # Pipeline. The embedding model must match the one the benchmark evaluated.
    embed_model: str = "triage/bge-small-en-v1.5-fp32"
    model_cache_dir: Path | None = None  # where the embedding model is stored
    labeler_dir: Path = Path("artifacts/labelers")  # <repo slug>/LATEST -> version dir
    retrieval_k: int = 5
    code_version: str = "dev"  # set at image build time


@lru_cache
def get_settings() -> Settings:
    return Settings()
