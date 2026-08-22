# src/ar_tokenwise/provider_counters.py
"""Real, provider-exact token counters for Anthropic Claude and Google Gemini.

report.py's default counter (tiktoken's o200k_base) is a PROXY for
GPT-4o-family tokenization -- it is not the tokenizer Claude or Gemini
actually use, so numbers reported against it are not exact for those
providers (see report.py's module docstring for that limitation). This
module provides counters backed by each provider's own official
token-counting API, so a report can be exact for the specific provider
you actually target, not an approximation.

TRADE-OFF, stated plainly: these counters make a REAL NETWORK CALL to
the provider's API on EVERY SINGLE invocation -- there is no local/
offline mode, unlike tiktoken which only needs network on first use and
then computes locally. This means:
- They require network access and valid credentials for that provider
  (ANTHROPIC_API_KEY / GEMINI_API_KEY environment variables by default).
- Each call has real network latency. Do NOT use these as the
  TokenCounter inside a per-request hot path (e.g.
  integrations/litellm_plugin.py's async_pre_call_hook, which is
  explicitly documented there as requiring no I/O) -- they are
  appropriate for one-off reports, benchmarks, or batch/offline
  analysis, not high-frequency inline use.
- Both providers' token-counting endpoints are documented as free of
  charge as of this writing, but this library does not track pricing
  (see report.py's docstring for the same "no hardcoded pricing"
  principle) -- check current provider documentation yourself.

Neither provider SDK is imported eagerly -- each is an optional
dependency (`pip install ar-tokenwise[providers]`), matching the
try/except-ImportError pattern used by integrations/*.py.
"""

from __future__ import annotations

from ar_tokenwise.report import TokenCounter


def get_anthropic_counter(model: str, api_key: str | None = None) -> TokenCounter:
    """Build a token counter backed by Anthropic's official count_tokens API.

    Makes a real network call to Anthropic on every invocation of the
    returned counter -- see module docstring for this trade-off.

    Args:
        model: Anthropic model name (e.g. "claude-opus-5") to count
            tokens for -- required because tokenization can differ
            between model generations.
        api_key: Anthropic API key. If omitted, the underlying SDK
            reads the ``ANTHROPIC_API_KEY`` environment variable.

    Returns:
        A callable mapping text -> exact input token count for
        ``model``, per Anthropic's own count_tokens endpoint.

    Raises:
        ImportError: if the ``anthropic`` package is not installed.
    """
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "get_anthropic_counter() requires the anthropic package. "
            "Install it with `pip install ar-tokenwise[providers]`."
        ) from exc

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def _count(text: str) -> int:
        response = client.messages.count_tokens(
            model=model,
            messages=[{"role": "user", "content": text}],
        )
        return response.input_tokens

    return _count


def get_gemini_counter(model: str, api_key: str | None = None) -> TokenCounter:
    """Build a token counter backed by Google's official Gemini count_tokens API.

    Makes a real network call to Google on every invocation of the
    returned counter -- see module docstring for this trade-off.

    Args:
        model: Gemini model name (e.g. "gemini-3.7-flash") to count
            tokens for.
        api_key: Gemini API key. If omitted, the underlying SDK reads
            the ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) environment
            variable.

    Returns:
        A callable mapping text -> exact token count for ``model``, per
        Google's own count_tokens endpoint.

    Raises:
        ImportError: if the ``google-genai`` package is not installed.
    """
    try:
        from google import genai
    except ImportError as exc:
        raise ImportError(
            "get_gemini_counter() requires the google-genai package. "
            "Install it with `pip install ar-tokenwise[providers]`."
        ) from exc

    client = genai.Client(api_key=api_key) if api_key else genai.Client()

    def _count(text: str) -> int:
        response = client.models.count_tokens(model=model, contents=text)
        return response.total_tokens

    return _count