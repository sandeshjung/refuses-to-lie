from unittest.mock import Mock

import pytest
import requests

from refuses_to_lie.llm_client import call_gemini, call_groq, with_backoff


def _http_error(status_code: int) -> requests.exceptions.HTTPError:
    response = Mock(spec=requests.Response)
    response.status_code = status_code
    return requests.exceptions.HTTPError(response=response)


def test_with_backoff_retries_transient_failures_then_succeeds():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise _http_error(429)
        return "ok"

    result = with_backoff(flaky, max_retries=5, base_delay=0.001)

    assert result == "ok"
    assert len(calls) == 3


def test_with_backoff_gives_up_after_max_retries():
    calls = []

    def always_fails():
        calls.append(1)
        raise _http_error(503)

    with pytest.raises(requests.exceptions.HTTPError):
        with_backoff(always_fails, max_retries=3, base_delay=0.001)

    assert len(calls) == 3


def test_with_backoff_does_not_retry_non_retryable_errors():
    calls = []

    def bad_request():
        calls.append(1)
        raise _http_error(400)

    with pytest.raises(requests.exceptions.HTTPError):
        with_backoff(bad_request, max_retries=5, base_delay=0.001)

    # A 400 (bad request) is never going to succeed on retry, unlike a 429 or
    # 503 — retrying it would just burn through the rate-limit budget.
    assert len(calls) == 1


def test_call_gemini_writes_and_reuses_cache(tmp_path):
    cache_dir = tmp_path / "llm_cache"
    result = call_gemini(
        "Reply with exactly the single word: pineapple",
        model="gemini-3.6-flash",
        cache_key="test-gemini-pineapple",
        cache_dir=cache_dir,
    )
    assert "pineapple" in result.lower()
    assert (cache_dir / "test-gemini-pineapple.json").exists()

    # Second call with the same key must come from disk, not the network —
    # corrupt the cache dir's write path to prove no live call happens.
    cached_result = call_gemini(
        "Reply with exactly the single word: pineapple",
        model="nonexistent-model-would-error",
        cache_key="test-gemini-pineapple",
        cache_dir=cache_dir,
    )
    assert cached_result == result


def test_call_groq_writes_and_reuses_cache(tmp_path):
    cache_dir = tmp_path / "llm_cache"
    result = call_groq(
        "Reply with exactly the single word: mango",
        model="openai/gpt-oss-120b",
        cache_key="test-groq-mango",
        cache_dir=cache_dir,
    )
    assert "mango" in result.lower()
    assert (cache_dir / "test-groq-mango.json").exists()

    cached_result = call_groq(
        "Reply with exactly the single word: mango",
        model="nonexistent-model-would-error",
        cache_key="test-groq-mango",
        cache_dir=cache_dir,
    )
    assert cached_result == result
