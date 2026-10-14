"""The `claude -p` backend, tested against a fake `claude` executable (no login, offline)."""

import json
import stat
import sys

import pytest

from triage import llm
from triage.config import ModelConfig
from triage.eval import run as run_mod

FAKE_CLAUDE = """#!{python}
import json, os, sys

args = sys.argv[1:]
if args[:2] == ["auth", "status"]:
    print(json.dumps({{"authMethod": os.environ.get("FAKE_AUTH", "claude.ai")}}))
    sys.exit(0)

prompt = sys.stdin.read()
system = args[args.index("--system-prompt") + 1]
reply = {{
    "saw_api_key": "ANTHROPIC_API_KEY" in os.environ,
    "model_flag": args[args.index("--model") + 1],
    "tools_flag": args[args.index("--tools") + 1],
    "system": system,
    "prompt": prompt,
}}
print(json.dumps({{
    "type": "result",
    "is_error": os.environ.get("FAKE_MODE") == "error",
    "result": "oops" if os.environ.get("FAKE_MODE") == "error" else json.dumps(reply),
    "total_cost_usd": 0.0012,
    "duration_api_ms": 850,
    "usage": {{"input_tokens": 100, "cache_read_input_tokens": 20, "output_tokens": 7}},
    "modelUsage": {{"claude-haiku-5-5": {{}}}},
}}))
"""


@pytest.fixture
def fake_claude(tmp_path, monkeypatch):
    path = tmp_path / "claude"
    path.write_text(FAKE_CLAUDE.format(python=sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("CLAUDE_BIN", str(path))
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "cache")
    return path


MODEL = ModelConfig(key="haiku-sub", id="claude-cli/haiku")


def test_cli_call_parses_reply_and_usage(fake_claude):
    reply, record = llm.complete_json(MODEL, "be terse", "classify this")
    assert reply["model_flag"] == "haiku"
    assert reply["tools_flag"] == ""
    assert reply["system"] == "be terse"
    assert reply["prompt"] == "classify this"
    assert record.input_tokens == 120  # input + cache reads
    assert record.output_tokens == 7
    assert record.cost_usd == pytest.approx(0.0012)
    assert record.latency_s == pytest.approx(0.85)  # API time, not CLI startup
    assert record.resolved_model == "claude-haiku-5-5"
    assert not record.cached


def test_cli_never_sees_api_key(fake_claude, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    reply, _ = llm.complete_json(MODEL, "s", "u")
    assert reply["saw_api_key"] is False


def test_cli_error_raises(fake_claude, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "error")
    with pytest.raises(llm.LLMError, match="claude -p failed"):
        llm.complete_json(MODEL, "s", "u")


def test_cli_second_call_is_cached(fake_claude):
    llm.complete_json(MODEL, "s", "same prompt")
    fake_claude.unlink()  # a cache hit must not need the binary
    reply, record = llm.complete_json(MODEL, "s", "same prompt")
    assert record.cached
    assert reply["prompt"] == "same prompt"


def test_cli_missing_binary(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDE_BIN", "")
    monkeypatch.setattr(llm.shutil, "which", lambda name: None)
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "cache")
    with pytest.raises(llm.LLMError, match="not found"):
        llm.complete_json(MODEL, "s", "u")


def test_availability_requires_subscription_login(fake_claude, monkeypatch):
    assert run_mod._model_available(MODEL)
    monkeypatch.setenv("FAKE_AUTH", "api_key")
    assert not run_mod._model_available(MODEL)


def test_models_config_has_subscription_entries():
    from triage.config import load_models

    models = load_models()
    assert models["haiku-sub"].id == "claude-cli/haiku"
    assert json.dumps(models["sonnet-sub"].params) == "{}"
