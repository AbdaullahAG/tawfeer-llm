"""Unit tests for ar_tokenwise.provider_counters.

No real API credentials are available in CI/dev environments for these
tests, so real network calls are never made here. Instead:
- The ImportError path (SDK not installed) is tested unconditionally.
- The SDK-wiring logic (do we call the right method and extract the
  right response field?) is tested by installing the REAL provider SDK
  and monkeypatching only the specific network-calling method, so the
  test exercises our actual code path rather than a fully-mocked stand-in.
"""

import builtins

import pytest

from ar_tokenwise import provider_counters


# --- ImportError paths: no SDK required to run these -----------------


def test_anthropic_counter_raises_clear_error_without_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "anthropic":
            raise ImportError("simulated missing anthropic")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="ar-tokenwise\\[providers\\]"):
        provider_counters.get_anthropic_counter(model="claude-opus-5")


def test_gemini_counter_raises_clear_error_without_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "google.genai" or name == "google":
            raise ImportError("simulated missing google-genai")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="ar-tokenwise\\[providers\\]"):
        provider_counters.get_gemini_counter(model="gemini-3.7-flash")


# --- Real-SDK wiring, network call monkeypatched --------------------------


def test_anthropic_counter_extracts_input_tokens_from_response() -> None:
    pytest.importorskip("anthropic", reason="optional dependency not installed")
    import anthropic

    class FakeCountTokensResponse:
        input_tokens = 42

    class FakeMessages:
        def count_tokens(self, model: str, messages: list[dict[str, str]]) -> object:
            assert model == "claude-opus-5"
            assert messages == [{"role": "user", "content": "hello"}]
            return FakeCountTokensResponse()

    class FakeClient:
        def __init__(self, api_key: str | None = None) -> None:
            self.messages = FakeMessages()

    original_client = anthropic.Anthropic
    anthropic.Anthropic = FakeClient  # type: ignore[misc]
    try:
        counter = provider_counters.get_anthropic_counter(model="claude-opus-5")
        assert counter("hello") == 42
    finally:
        anthropic.Anthropic = original_client  # type: ignore[misc]


def test_gemini_counter_extracts_total_tokens_from_response() -> None:
    pytest.importorskip("google.genai", reason="optional dependency not installed")
    from google import genai

    class FakeCountTokensResponse:
        total_tokens = 17

    class FakeModels:
        def count_tokens(self, model: str, contents: str) -> object:
            assert model == "gemini-3.7-flash"
            assert contents == "hello"
            return FakeCountTokensResponse()

    class FakeClient:
        def __init__(self, api_key: str | None = None) -> None:
            self.models = FakeModels()

    original_client = genai.Client
    genai.Client = FakeClient  # type: ignore[misc]
    try:
        counter = provider_counters.get_gemini_counter(model="gemini-3.7-flash")
        assert counter("hello") == 17
    finally:
        genai.Client = original_client  # type: ignore[misc]