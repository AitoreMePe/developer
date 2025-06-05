from typing import List, Dict, Tuple

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


def generate_chat(messages: List[Dict[str, str]], model: str, backend: str = "openai", **kwargs) -> str:
    """Generate chat completion text from either OpenAI or HuggingFace backend."""
    if backend == "openai":
        if openai is None:
            raise ImportError("openai package not available")
        response = openai.ChatCompletion.create(model=model, messages=messages, **kwargs)
        msg = response["choices"][0]["message"]
        if msg.get("content") is not None:
            return msg["content"]
        if msg.get("function_call"):
            return msg["function_call"].get("arguments", "")
        return ""
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
        return text[len(prompt_text) :].strip()
    else:
        raise ValueError(f"Unsupported backend: {backend}")
