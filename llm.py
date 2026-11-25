import json
import re

import groq

from .config import CFG

_client = None


class TruncatedError(ValueError):
    """The reply hit the token limit, so it is incomplete."""


def _get_client() -> groq.Groq:
    global _client
    if _client is None:
        # The SDK waits and retries on 429; low-tier limits make those likely.
        _client = groq.Groq(api_key=CFG.groq_api_key, max_retries=4)
    return _client


def complete(prompt: str, system: str = "", model: str | None = None,
             max_tokens: int = CFG.max_output_tokens, json_mode: bool = False,
             temperature: float = 0) -> str:
    model = model or CFG.llm_model
    messages = [{"role": "user", "content": prompt}]
    if system:
        messages.insert(0, {"role": "system", "content": system})

    extra: dict = {}
    if model.startswith("openai/gpt-oss"):   # these parameters are model-specific
        extra.update(reasoning_effort="low", include_reasoning=False)
    if json_mode:
        extra["response_format"] = {"type": "json_object"}

    try:
        resp = _get_client().chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            **extra,
        )
    except groq.BadRequestError as e:
        if json_mode and "json_validate_failed" in str(e):
            raise ValueError(f"Model produced invalid JSON: {e}") from e
        raise

    choice = resp.choices[0]
    if choice.finish_reason == "length":
        raise TruncatedError(f"The model's reply was cut off at {max_tokens} tokens.")
    return (choice.message.content or "").strip()


def complete_json(prompt: str, system: str = "", model: str | None = None,
                  max_tokens: int = CFG.max_output_tokens) -> dict:
    """Ask for a JSON object and parse it tolerantly (handles code fences / preamble)."""
    raw = complete(prompt, system=system, model=model, max_tokens=max_tokens, json_mode=True)
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        raise ValueError(f"No JSON object in model output: {raw[:200]}")
    return json.loads(match.group(0))
