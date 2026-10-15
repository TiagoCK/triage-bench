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
    # A label added within this many seconds of issue creation counts as template-applied.
    creation_window_s: int = 60
    # Labels template-applied for more than this share of their train uses are not targets.
    max_creation_share: float = 0.5
    # The last N months of the train period double as a validation set for choosing
    # thresholds (e.g. the duplicate confidence gate), so nothing is tuned on test.
    validation_months: int = 3

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
                creation_window_s=int(merged.get("creation_window_s", 60)),
                max_creation_share=float(merged.get("max_creation_share", 0.5)),
                validation_months=int(merged.get("validation_months", 3)),
            )
        )
    return out


def load_models(path: Path = Path("configs/models.yaml")) -> dict[str, ModelConfig]:
    raw = yaml.safe_load(path.read_text())["models"]
    return {key: ModelConfig(key=key, **spec) for key, spec in raw.items()}
