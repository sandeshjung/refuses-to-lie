"""Thin, retrying, disk-cached clients for the two LLM providers this
project uses: Google Gemini (generator) and Groq (verifier).

Both free tiers rate-limit. Every call here backs off exponentially on
429/5xx, and when given a cache_key, writes its result to disk the moment
it returns and skips the call entirely if that key is already on disk —
the "resumable, not restartable" property an overnight eval run needs to
survive a rate limit without losing progress.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

load_dotenv()

DEFAULT_CACHE_DIR = Path(".cache/llm")
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

_google_client: genai.Client | None = None


def _get_google_client() -> genai.Client:
    global _google_client
    if _google_client is None:
        _google_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    return _google_client


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, genai_errors.APIError):
        return exc.code in _RETRYABLE_STATUS_CODES
    if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
        return exc.response.status_code in _RETRYABLE_STATUS_CODES
    return isinstance(exc, requests.exceptions.ConnectionError | requests.exceptions.Timeout)


def with_backoff(fn, max_retries: int = 5, base_delay: float = 1.0):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            if attempt == max_retries - 1 or not _is_retryable(exc):
                raise
            delay = base_delay * (2**attempt) + random.uniform(0, 0.5)
            time.sleep(delay)
    raise AssertionError("unreachable")


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key.replace('/', '_')}.json"


def _read_cache(cache_dir: Path, key: str) -> str | None:
    path = _cache_path(cache_dir, key)
    if not path.exists():
        return None
    result: str = json.loads(path.read_text())["text"]
    return result


def _write_cache(cache_dir: Path, key: str, text: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    _cache_path(cache_dir, key).write_text(json.dumps({"text": text}))


def call_gemini(
    prompt: str,
    model: str,
    temperature: float = 0.2,
    cache_key: str | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> str:
    if cache_key is not None:
        cached = _read_cache(cache_dir, cache_key)
        if cached is not None:
            return cached

    def _call() -> str:
        response = _get_google_client().models.generate_content(
            model=model,
            contents=prompt,
            config={"temperature": temperature},
        )
        return response.text or ""

    text = with_backoff(_call)
    if cache_key is not None:
        _write_cache(cache_dir, cache_key, text)
    return text


def call_groq(
    prompt: str,
    model: str,
    temperature: float = 0.2,
    cache_key: str | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> str:
    if cache_key is not None:
        cached = _read_cache(cache_dir, cache_key)
        if cached is not None:
            return cached

    def _call() -> str:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
            },
            timeout=60,
        )
        response.raise_for_status()
        result: str = response.json()["choices"][0]["message"]["content"]
        return result

    text = with_backoff(_call)
    if cache_key is not None:
        _write_cache(cache_dir, cache_key, text)
    return text
