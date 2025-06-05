import pytest
import sys
import types
import importlib.util
from pathlib import Path

sys.modules.setdefault(
    "openai_function_call",
    types.SimpleNamespace(
        openai_function=lambda f: setattr(f, "openai_schema", {}) or f
    ),
)
sys.modules.setdefault(
    "tenacity",
    types.SimpleNamespace(
        retry=lambda *a, **k: (lambda f: f),
        stop_after_attempt=lambda *a, **k: None,
        wait_random_exponential=lambda *a, **k: None,
    ),
)

ROOT = Path(__file__).resolve().parents[1]
LLM_PATH = ROOT / "smol_dev" / "llm.py"
PROMPTS_PATH = ROOT / "smol_dev" / "prompts.py"

llm_spec = importlib.util.spec_from_file_location("smol_dev.llm", LLM_PATH)
llm = importlib.util.module_from_spec(llm_spec)
llm_spec.loader.exec_module(llm)
sys.modules.setdefault("smol_dev", types.ModuleType("smol_dev"))
sys.modules["smol_dev.llm"] = llm

prompts_spec = importlib.util.spec_from_file_location("smol_dev.prompts", PROMPTS_PATH)
prompts = importlib.util.module_from_spec(prompts_spec)
prompts_spec.loader.exec_module(prompts)


def test_specify_file_paths_returns_list(monkeypatch):
    def fake_generate_chat(messages, model, backend, **kwargs):
        return '["main.py", "utils.py"]'
    monkeypatch.setattr(prompts, "generate_chat", fake_generate_chat)
    result = prompts.specify_file_paths("prompt", "plan")
    assert result == ["main.py", "utils.py"]


def test_generate_code_sync_strips_fences(monkeypatch):
    def fake_generate_chat(messages, model, backend, **kwargs):
        return "```python\nprint('hi')\n```"
    monkeypatch.setattr(prompts, "generate_chat", fake_generate_chat)
    code = prompts.generate_code_sync("prompt", "plan", "main.py")
    assert "```" not in code
    assert code.strip() == "print('hi')"
