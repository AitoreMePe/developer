import os
import sys
import shelve
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import smol_dev.llm as llm


messages = [{"role": "user", "content": "hi"}]


def test_generate_chat_caches(tmp_path, monkeypatch):
    cache_file = tmp_path / "cache"
    monkeypatch.setenv("SMOL_DEV_CACHE_PATH", str(cache_file))
    monkeypatch.setattr(llm, "_cache_path", str(cache_file))

    call_count = {"n": 0}

    def fake_create(**kwargs):
        call_count["n"] += 1
        return {"choices": [{"message": {"content": "hello"}}]}

    fake_openai = SimpleNamespace(ChatCompletion=SimpleNamespace(create=fake_create))

    with patch.object(llm, "openai", fake_openai):
        result1 = llm.generate_chat(messages, "test-model")
        assert result1 == "hello"
        assert call_count["n"] == 1

    with shelve.open(str(cache_file)) as cache:
        assert len(cache) == 1

    def should_not_call(**kwargs):
        raise AssertionError("backend called")

    fake_openai2 = SimpleNamespace(ChatCompletion=SimpleNamespace(create=should_not_call))

    with patch.object(llm, "openai", fake_openai2):
        result2 = llm.generate_chat(messages, "test-model")
        assert result2 == "hello"

    with shelve.open(str(cache_file)) as cache:
        assert len(cache) == 1
