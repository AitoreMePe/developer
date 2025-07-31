import os
import sys
import shelve
import time
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

    fake_openai = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )

    with patch.object(llm, "openai", fake_openai):
        result1 = llm.generate_chat(messages, "test-model")
        assert result1 == "hello"
        assert call_count["n"] == 1

    with shelve.open(str(cache_file)) as cache:
        assert len(cache) == 1

    def should_not_call(**kwargs):
        raise AssertionError("backend called")

    fake_openai2 = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=should_not_call))
    )

    with patch.object(llm, "openai", fake_openai2):
        result2 = llm.generate_chat(messages, "test-model")
        assert result2 == "hello"

    with shelve.open(str(cache_file)) as cache:
        assert len(cache) == 1


def test_generate_chat_cache_expires(tmp_path, monkeypatch):
    cache_file = tmp_path / "cache2"
    monkeypatch.setenv("SMOL_DEV_CACHE_PATH", str(cache_file))
    monkeypatch.setattr(llm, "_cache_path", str(cache_file))
    monkeypatch.setenv("SMOL_DEV_CACHE_TTL", "0.1")

    call_count = {"n": 0}

    def fake_create(**kwargs):
        call_count["n"] += 1
        return {"choices": [{"message": {"content": "bye"}}]}

    fake_openai = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )

    with patch.object(llm, "openai", fake_openai):
        result1 = llm.generate_chat(messages, "test-model")
        assert result1 == "bye"
        assert call_count["n"] == 1

    time.sleep(0.2)

    with patch.object(llm, "openai", fake_openai):
        result2 = llm.generate_chat(messages, "test-model")
        assert result2 == "bye"
        assert call_count["n"] == 2

    with shelve.open(str(cache_file)) as cache:
        assert len(cache) == 1


def test_openai_defaults_to_ollama(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    cache_file = tmp_path / "ollama-cache"
    monkeypatch.setattr(llm, "_cache_path", str(cache_file))

    captured = {}

    def fake_create(**kwargs):
        captured["key"] = fake_openai.api_key
        captured["base"] = fake_openai.api_base
        return {"choices": [{"message": {"content": "ok"}}]}

    fake_openai = SimpleNamespace(
        api_key=None,
        api_base=None,
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)),
    )

    with patch.object(llm, "openai", fake_openai):
        llm.generate_chat(messages, "test-model")

    assert captured["key"] == "ollama"
    assert captured["base"] == "http://localhost:11434/v1"


def test_hf_caching(monkeypatch, tmp_path):
    cache_file = tmp_path / "hf-cache"
    monkeypatch.setenv("SMOL_DEV_CACHE_PATH", str(cache_file))
    monkeypatch.setattr(llm, "_cache_path", str(cache_file))
    monkeypatch.setattr(llm, "_hf_models", {})

    calls = {"load": 0, "generate": 0}

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, name):
            calls["load"] += 1
            return cls()

        def encode(self, text, return_tensors=None):
            return text

        def decode(self, tokens, skip_special_tokens=True):
            if isinstance(tokens, list):
                return tokens[0]
            return tokens

    class FakeModel:
        @classmethod
        def from_pretrained(cls, name):
            calls["load"] += 1
            return cls()

        def generate(self, input_ids, max_new_tokens=256):
            calls["generate"] += 1
            return [input_ids + " there"]

    monkeypatch.setattr(llm, "AutoTokenizer", FakeTokenizer, raising=False)
    monkeypatch.setattr(llm, "AutoModelForCausalLM", FakeModel, raising=False)

    result1 = llm.generate_chat(messages, "fake-model", backend="hf")
    assert result1 == "there"
    assert calls["load"] == 2
    assert calls["generate"] == 1

    result2 = llm.generate_chat(messages, "fake-model", backend="hf")
    assert result2 == "there"
    assert calls["load"] == 2
    assert calls["generate"] == 1

    with shelve.open(str(cache_file)) as cache:
        assert len(cache) == 1
