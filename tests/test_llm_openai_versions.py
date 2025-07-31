import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import smol_dev.llm as llm


messages = [{"role": "user", "content": "hello"}]


def test_generate_chat_openai_version_detection(tmp_path, monkeypatch):
    cache1 = tmp_path / "cache1"
    monkeypatch.setattr(llm, "_cache_path", str(cache1))

    called = {"v1": False, "v2": False}

    def create_v1(**kwargs):
        called["v1"] = True
        return {"choices": [{"message": {"content": "one"}}]}

    fake_openai_v1 = SimpleNamespace(
        api_key="key",
        api_base="base",
        ChatCompletion=SimpleNamespace(create=create_v1),
    )

    with patch.object(llm, "openai", fake_openai_v1):
        result1 = llm.generate_chat(messages, "model")
    assert result1 == "one"
    assert called["v1"]

    cache2 = tmp_path / "cache2"
    monkeypatch.setattr(llm, "_cache_path", str(cache2))

    def create_v2(**kwargs):
        called["v2"] = True
        return {"choices": [{"message": {"content": "two"}}]}

    fake_openai_v2 = SimpleNamespace(
        api_key="key",
        api_base="base",
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_v2)),
    )

    with patch.object(llm, "openai", fake_openai_v2):
        result2 = llm.generate_chat(messages, "model")
    assert result2 == "two"
    assert called["v2"]
