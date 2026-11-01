"""LLM access in ONE place: a free API (Groq, OpenAI-compatible) or local Ollama.
Chosen from .env:
    LLM_API_KEY=...          (required for the API backend)
    LLM_MODEL=openai/gpt-oss-120b
    LLM_BASE_URL=...         (optional, default = Groq)
    LLM_BACKEND=ollama       (optional: force the local model)
    LOCAL_MODEL=qwen2.5:3b   (optional: model used when the backend is ollama)
"""
import os

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY", "").strip().strip("\"'")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
API_MODEL = os.getenv("LLM_MODEL", "").strip()
BACKEND = os.getenv("LLM_BACKEND", "api" if (API_KEY and API_MODEL) else "ollama").lower()
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "qwen2.5:3b")

KEEP_ALIVE = "30m"
OLLAMA_OPTIONS = {"temperature": 0, "num_ctx": 4096, "num_predict": 500, "repeat_penalty": 1.15}

_api = None
_ollama = None
if BACKEND == "api":
    from openai import AsyncOpenAI

    _api = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
else:
    from ollama import AsyncClient

    _ollama = AsyncClient()


def describe() -> str:
    """Which backend is active (never includes the key)."""
    if BACKEND == "api":
        return f"API {API_MODEL} @ {BASE_URL}"
    return f"local Ollama {LOCAL_MODEL}"


async def warm_up() -> None:
    """Local model only: load it into memory now so the first question is fast."""
    if _ollama is not None:
        await _ollama.chat(
            model=LOCAL_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            options={**OLLAMA_OPTIONS, "num_predict": 1},
            keep_alive=KEEP_ALIVE,
        )


async def generate(prompt: str, max_tokens: int = 2000) -> str:
    """One prompt in, one text answer out."""
    if _api is not None:
        extra = {}
        if "gpt-oss" in API_MODEL:
            # reasoning models spend tokens on "thinking": keep it short
            extra["extra_body"] = {"reasoning_effort": "low"}
        resp = await _api.chat.completions.create(
            model=API_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens,
            **extra,
        )
        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        if not text and choice.finish_reason == "length":
            raise RuntimeError("The model used all max_tokens for reasoning and returned no answer. "
                               "Increase max_tokens in llm.generate().")
        return text

    resp = await _ollama.chat(
        model=LOCAL_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options=OLLAMA_OPTIONS,
        keep_alive=KEEP_ALIVE,
    )
    return (resp.message.content or "").strip()
