from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RepoConfig:
    name: str
    since: datetime
    until: datetime
    cutoff: datetime
    top_k_labels: int
    min_label_count: int

    @property
    def slug(self) -> str:
        return self.name.replace("/", "__")


@dataclass(frozen=True)
class ModelConfig:
    key: str
    id: str
    input_price: float | None = None  # USD per 1M tokens; None means "ask LiteLLM"
    output_price: float | None = None
    concurrency: int = 4
    # Extra keyword arguments for litellm.completion, e.g. {"think": false} for Ollama.
    params: dict[str, Any] = field(default_factory=dict)


def _date(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=UTC)


def load_repos(path: Path = Path("configs/repos.yaml")) -> list[RepoConfig]:
    raw = yaml.safe_load(path.read_text())
    defaults = raw.get("defaults", {})
    out = []
    for entry in raw["repos"]:
        merged = {**defaults, **entry}
        out.append(
            RepoConfig(
                name=merged["name"],
                since=_date(merged["since"]),
                until=_date(merged["until"]),
                cutoff=_date(merged["cutoff"]),
                top_k_labels=int(merged["top_k_labels"]),
                min_label_count=int(merged["min_label_count"]),
            )
        )
    return out


def load_models(path: Path = Path("configs/models.yaml")) -> dict[str, ModelConfig]:
    raw = yaml.safe_load(path.read_text())["models"]
    return {key: ModelConfig(key=key, **spec) for key, spec in raw.items()}
