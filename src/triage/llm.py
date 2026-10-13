"""One narrow entry point for every LLM call: `complete_json`.

Everything model-specific stays here so the rest of the code never imports LiteLLM. Calls
are cached on disk by (model id, prompt) so re-running an eval costs nothing, and each call
records tokens, cost and latency for the results table.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from triage.config import ModelConfig

CACHE_DIR = Path(".cache/llm")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class CallRecord:
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    latency_s: float
    cached: bool


class LLMError(RuntimeError):
    pass


def complete_json(
    model: ModelConfig, system: str, user: str, max_tokens: int = 512
) -> tuple[dict[str, Any], CallRecord]:
    """Ask for a JSON object and parse it. Raises LLMError if the reply has no valid JSON."""
    key_parts = [model.id, model.params, system, user, max_tokens]
    key = hashlib.sha256(json.dumps(key_parts, sort_keys=True).encode()).hexdigest()
    cache_path = CACHE_DIR / f"{key}.json"
    if cache_path.exists():
        hit = json.loads(cache_path.read_text())
        record = CallRecord(**{**hit["record"], "cached": True})
        return _parse_json(hit["text"]), record

    import litellm  # imported lazily so the non-LLM baselines work without the extra

    start = time.perf_counter()
    try:
        resp = litellm.completion(
            model=model.id,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            max_tokens=max_tokens,
            **model.params,
        )
    except Exception as exc:  # LiteLLM wraps provider errors in many types
        raise LLMError(f"{model.key}: {exc}") from exc
    latency = time.perf_counter() - start

    text = resp.choices[0].message.content or ""
    usage = resp.usage
    record = CallRecord(
        model=model.key,
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        cost_usd=_cost(model, resp, usage.prompt_tokens, usage.completion_tokens),
        latency_s=latency,
        cached=False,
    )
    parsed = _parse_json(text)  # parse before caching so bad replies are retried next run
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"text": text, "record": asdict(record)}))
    return parsed, record


def _cost(model: ModelConfig, resp: Any, tokens_in: int, tokens_out: int) -> float | None:
    if model.input_price is not None and model.output_price is not None:
        return (tokens_in * model.input_price + tokens_out * model.output_price) / 1e6
    import litellm

    try:
        return float(litellm.completion_cost(completion_response=resp))
    except Exception:
        return None  # unknown model price; shows as n/a rather than a wrong number


def _parse_json(text: str) -> dict[str, Any]:
    text = _THINK_RE.sub("", text).strip()
    match = _JSON_RE.search(text)
    if not match:
        raise LLMError(f"no JSON object in reply: {text[:200]!r}")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise LLMError(f"invalid JSON in reply: {text[:200]!r}") from exc
