from typing import List, Dict, Tuple
import json
import os
import hashlib
import shelve
import time

try:
    import openai
except Exception:  # pragma: no cover - openai optional
    openai = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover - transformers optional
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore

_hf_models: Dict[str, Tuple[object, object]] = {}

# path to persistent cache used to store chat responses
_cache_path = os.environ.get("SMOL_DEV_CACHE_PATH", os.path.expanduser("~/.smol_dev_cache"))


def _get_cache_ttl() -> float:
    """Return cache TTL in seconds, 0 for no expiration."""
    try:
        return float(os.environ.get("SMOL_DEV_CACHE_TTL", "0"))
    except ValueError:  # pragma: no cover - bad env var
        return 0.0


def generate_chat(messages: List[Dict[str, str]], model: str, backend: str = "openai", **kwargs) -> str:
    """Generate chat completion text from either OpenAI or HuggingFace backend with caching."""
    key_data = json.dumps({"messages": messages, "model": model, "backend": backend, "kwargs": kwargs}, sort_keys=True)
    key = hashlib.sha256(key_data.encode("utf-8")).hexdigest()

    cache_ttl = _get_cache_ttl()

    with shelve.open(_cache_path) as cache:
        if key in cache:
            entry = cache[key]
            # handle legacy string caches
            if isinstance(entry, dict) and "value" in entry and "time" in entry:
                if cache_ttl <= 0 or time.time() - entry["time"] <= cache_ttl:
                    return entry["value"]
            elif cache_ttl <= 0:
                return entry

        if backend == "openai":
            if openai is None:
                raise ImportError("openai package not available")
            response = openai.ChatCompletion.create(model=model, messages=messages, **kwargs)
            msg = response["choices"][0]["message"]
            if msg.get("content") is not None:
                result = msg["content"]
            elif msg.get("function_call"):
                result = msg["function_call"].get("arguments", "")
            else:
                result = ""
        elif backend == "hf":
            if AutoModelForCausalLM is None or AutoTokenizer is None:
                raise ImportError("transformers package not available")

            tokenizer, model_obj = _hf_models.get(model, (None, None))
            if tokenizer is None or model_obj is None:
                tokenizer = AutoTokenizer.from_pretrained(model)
                model_obj = AutoModelForCausalLM.from_pretrained(model)
                _hf_models[model] = (tokenizer, model_obj)

            prompt_text = "".join(f"{m['role']}: {m['content']}\n" for m in messages)
            input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
            max_new_tokens = kwargs.get("max_tokens", 256)
            output = model_obj.generate(input_ids, max_new_tokens=max_new_tokens)
            text = tokenizer.decode(output[0], skip_special_tokens=True)
            result = text[len(prompt_text) :].strip()
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        cache[key] = {"value": result, "time": time.time()}
        return result
