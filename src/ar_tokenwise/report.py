"""Transparent before/after token-usage reporting for LLM API calls.

This module deliberately avoids embedding any built-in token-count
heuristic or model pricing table: token counting must come from a real
tokenizer (caller-supplied or tiktoken via the optional `tokenizers`
extra), and cost figures must come from a price the caller provides.
Guessing either would produce numbers that look precise but are not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ar_tokenwise._internal import compute_fertility, count_words

TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class TokenReport:
    """Result of comparing token usage before and after optimization."""

    original_tokens: int
    optimized_tokens: int
    tokens_saved: int
    percent_saved: float
    original_fertility: float  # tokens per word, original text
    optimized_fertility: float  # tokens per word, optimized text
    estimated_cost_savings_usd: float | None


def get_default_counter(encoding_name: str = "o200k_base") -> TokenCounter:
    """Build a token counter backed by tiktoken.

    Args:
        encoding_name: tiktoken encoding to use. ``o200k_base`` matches
            recent GPT-4o-family models; pass another encoding name for
            other model families.

    Returns:
        A callable mapping text -> token count.

    Raises:
        ImportError: if tiktoken is not installed, with instructions to
            either install the optional extra or supply a custom counter.
    """
    try:
        import tiktoken
    except ImportError as exc:
        raise ImportError(
            "get_default_counter() requires tiktoken. Install it with "
            "`pip install ar-tokenwise[tokenizers]`, or pass your own "
            "`counter` callable to report_savings() instead."
        ) from exc

    encoding = tiktoken.get_encoding(encoding_name)
    return lambda text: len(encoding.encode(text))


def report_savings(
    original: str,
    optimized: str,
    counter: TokenCounter | None = None,
    cost_per_million_tokens: float | None = None,
) -> TokenReport:
    """Compare token usage between original and optimized text.

    Args:
        original: Text before optimization.
        optimized: Text after optimization (e.g. via ``normalize()``).
        counter: Token-counting function. Defaults to a tiktoken-backed
            counter (requires the ``tokenizers`` extra) if not provided.
        cost_per_million_tokens: Optional price the caller supplies, used
            to estimate dollar savings. No built-in pricing table is used,
            since vendor prices change independently of this library.

    Returns:
        A :class:`TokenReport` with token counts, savings, and fertility.

    Raises:
        TypeError: if ``original`` or ``optimized`` is not a string.
        ImportError: if ``counter`` is omitted and tiktoken is not installed.
    """
    if not isinstance(original, str) or not isinstance(optimized, str):
        raise TypeError("report_savings() expects str for original and optimized")

    if counter is None:
        counter = get_default_counter()

    original_tokens = counter(original)
    optimized_tokens = counter(optimized)
    tokens_saved = original_tokens - optimized_tokens

    percent_saved = (
        (tokens_saved / original_tokens) * 100.0 if original_tokens > 0 else 0.0
    )

    original_fertility = compute_fertility(original_tokens, count_words(original))
    optimized_fertility = compute_fertility(optimized_tokens, count_words(optimized))

    estimated_cost_savings_usd = (
        (tokens_saved / 1_000_000) * cost_per_million_tokens
        if cost_per_million_tokens is not None
        else None
    )

    return TokenReport(
        original_tokens=original_tokens,
        optimized_tokens=optimized_tokens,
        tokens_saved=tokens_saved,
        percent_saved=percent_saved,
        original_fertility=original_fertility,
        optimized_fertility=optimized_fertility,
        estimated_cost_savings_usd=estimated_cost_savings_usd,
    )