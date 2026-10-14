"""One narrow entry point for every LLM call: `complete_json`.

Everything model-specific stays here so the rest of the code never imports LiteLLM. Calls
are cached on disk by (model id, prompt) so re-running an eval costs nothing, and each call
records tokens, cost and latency for the results table.

Two backends:
  - LiteLLM for API models (`anthropic/...`, `ollama_chat/...`, any LiteLLM string).
  - The Claude Code CLI for `claude-cli/<alias>` models, which runs `claude -p` on the
    user's own Claude subscription login instead of an API key. See README, "Running Claude
    on a subscription", for the usage-policy notes and how it differs from a raw API call.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from triage.config import ModelConfig

CACHE_DIR = Path(".cache/llm")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

CLI_PREFIX = "claude-cli/"
CLI_TIMEOUT_S = 180


@dataclass
class CallRecord:
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    latency_s: float
    cached: bool
    resolved_model: str | None = None  # the exact model ID the provider reported


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

    if model.id.startswith(CLI_PREFIX):
        text, record = _complete_via_cli(model, system, user)
    else:
        text, record = _complete_via_litellm(model, system, user, max_tokens)
    parsed = _parse_json(text)  # parse before caching so bad replies are retried next run
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"text": text, "record": asdict(record)}))
    return parsed, record


def _complete_via_litellm(
    model: ModelConfig, system: str, user: str, max_tokens: int
) -> tuple[str, CallRecord]:
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

    usage = resp.usage
    record = CallRecord(
        model=model.key,
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        cost_usd=_cost(model, resp, usage.prompt_tokens, usage.completion_tokens),
        latency_s=latency,
        cached=False,
        resolved_model=getattr(resp, "model", None),
    )
    return resp.choices[0].message.content or "", record


def claude_bin() -> str | None:
    return os.environ.get("CLAUDE_BIN") or shutil.which("claude")


def cli_env() -> dict[str, str]:
    """Environment for `claude` subprocesses.

    ANTHROPIC_API_KEY is removed: Claude Code prefers an API key over the subscription
    login, and our .env puts the key in the environment, so leaving it in would silently
    bill the API instead of the subscription.
    """
    return {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}


def _complete_via_cli(model: ModelConfig, system: str, user: str) -> tuple[str, CallRecord]:
    """One `claude -p` call: our system prompt replaces Claude Code's, no tools, one turn.

    The CLI cannot set max_tokens or temperature. The cost it reports is its own estimate at
    API prices; on a subscription nothing is billed per call.
    """
    binary = claude_bin()
    if not binary:
        raise LLMError(f"{model.key}: `claude` CLI not found (install it or set CLAUDE_BIN)")
    cmd = [
        binary,
        "-p",
        "--output-format",
        "json",
        "--model",
        model.id.removeprefix(CLI_PREFIX),
        "--system-prompt",
        system,
        "--tools",
        "",
        "--disallowedTools",
        "mcp__*",
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--max-turns",
        "1",
    ]
    # An empty directory outside the repo, so no project CLAUDE.md, hooks or .mcp.json load.
    with tempfile.TemporaryDirectory(prefix="triage-cli-") as cwd:
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                input=user,
                capture_output=True,
                text=True,
                cwd=cwd,
                env=cli_env(),
                timeout=CLI_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as exc:
            raise LLMError(f"{model.key}: claude -p timed out after {CLI_TIMEOUT_S}s") from exc
        wall = time.perf_counter() - start

    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        detail = (proc.stderr or proc.stdout)[:300]
        raise LLMError(f"{model.key}: claude -p exit {proc.returncode}: {detail!r}") from exc
    if proc.returncode != 0 or out.get("is_error"):
        raise LLMError(f"{model.key}: claude -p failed: {str(out.get('result'))[:300]!r}")

    usage = out.get("usage") or {}
    input_tokens = sum(
        int(usage.get(k) or 0)
        for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    )
    model_usage = out.get("modelUsage") or {}
    # API time only, so latency is comparable with the LiteLLM path (CLI startup excluded).
    api_ms = out.get("duration_api_ms")
    record = CallRecord(
        model=model.key,
        input_tokens=input_tokens,
        output_tokens=int(usage.get("output_tokens") or 0),
        cost_usd=out.get("total_cost_usd"),
        latency_s=api_ms / 1000 if api_ms else wall,
        cached=False,
        resolved_model=",".join(sorted(model_usage)) or None,
    )
    return out.get("result") or "", record


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
