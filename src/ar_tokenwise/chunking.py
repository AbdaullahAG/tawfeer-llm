"""Sentence-boundary-aware text chunking for RAG pipelines.

Splits text into token-budgeted chunks along sentence boundaries rather
than a fixed character/token cut, which avoids severing sentences mid-way.
Falls back to word-level splitting only for a single sentence that alone
exceeds max_tokens, and merges an undersized trailing chunk into its
neighbor when possible.

Token counts used during accumulation are the SUM of per-unit counts, not
a re-tokenization of the merged text -- this is an approximation, not an
exact count. In practice this makes the max_tokens guard conservative
(actual merged token count is usually <= the summed estimate), never the
reverse, so chunks should not exceed max_tokens by a meaningful margin.
"""

from __future__ import annotations

import re

from ar_tokenwise.normalize import DEFAULT_MAX_LENGTH
from ar_tokenwise.report import TokenCounter

# Matches a run of whitespace immediately after a standard Arabic/Latin
# sentence-ending punctuation mark. See module docstring / chat explanation
# for why this pattern is not ReDoS-prone: fixed-width lookbehind over a
# single character class, followed by a single simple quantifier, no
# nesting or alternation -- matching is linear in input length.
_SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?؟؛])\s+")

# Plain whitespace splitter for the word-level fallback. Intentionally
# duplicated here (rather than importing report.py's private
# _WHITESPACE_PATTERN) to keep this module self-contained and not depend
# on another module's private implementation detail.
_WHITESPACE_SPLIT_PATTERN = re.compile(r"\s+")

DEFAULT_MIN_TOKENS = 100
DEFAULT_MAX_TOKENS = 512


def _validate_input(text: str, max_length: int, min_tokens: int, max_tokens: int) -> None:
    """Validate inputs before any processing.

    Raises:
        TypeError: if ``text`` is not a ``str``.
        ValueError: if ``text`` exceeds ``max_length``, or if the
            min/max token bounds are invalid.
    """
    if not isinstance(text, str):
        raise TypeError(f"chunk_text() expects str, got {type(text).__name__}")
    if len(text) > max_length:
        raise ValueError(
            f"Input length {len(text)} exceeds max_length={max_length}. "
            "Split the text or raise max_length explicitly if intentional."
        )
    if min_tokens <= 0 or max_tokens <= 0:
        raise ValueError("min_tokens and max_tokens must be positive integers")
    if min_tokens > max_tokens:
        raise ValueError(
            f"min_tokens ({min_tokens}) cannot exceed max_tokens ({max_tokens})"
        )


def _split_sentences(text: str) -> list[str]:
    """Split text into non-empty, stripped sentence strings."""
    parts = _SENTENCE_BOUNDARY_PATTERN.split(text.strip())
    return [part.strip() for part in parts if part.strip()]


def _split_oversized_sentence(
    sentence: str, counter: TokenCounter, max_tokens: int
) -> list[str]:
    """Split a single sentence that alone exceeds max_tokens, by words.

    This is an unavoidable fallback: a sentence longer than the chunk
    budget must be cut somewhere, and word boundaries are the least
    disruptive option short of leaving it as one oversized chunk.
    """
    words = [w for w in _WHITESPACE_SPLIT_PATTERN.split(sentence.strip()) if w]

    units: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = current + [word]
        if current and counter(" ".join(candidate)) > max_tokens:
            units.append(" ".join(current))
            current = [word]
        else:
            current = candidate
    if current:
        units.append(" ".join(current))
    return units


def chunk_text(
    text: str,
    counter: TokenCounter,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> list[str]:
    """Split text into sentence-boundary-aware, token-budgeted chunks.

    Args:
        text: Input text. Empty string returns an empty list.
        counter: Token-counting function (see ``ar_tokenwise.report``).
        min_tokens: Soft lower bound -- an undersized trailing chunk is
            merged into its predecessor when the merge still fits under
            max_tokens. Not a hard guarantee for every chunk (e.g. a lone
            oversized word cannot be padded to reach it).
        max_tokens: Hard upper bound target. Chunk sizing is based on
            summed per-unit token estimates (see module docstring), so
            actual re-tokenized size is usually at or under this value,
            not over it -- except for a single word that alone exceeds
            max_tokens, which becomes its own chunk since it cannot be
            split further.
        max_length: Maximum accepted input length in characters, used as
            a size-based safety guard. Raises if exceeded.

    Returns:
        List of chunk strings, in original order. Empty input returns [].

    Raises:
        TypeError: if ``text`` is not a string.
        ValueError: if ``text`` exceeds ``max_length``, or if
            ``min_tokens``/``max_tokens`` are invalid.
    """
    _validate_input(text, max_length, min_tokens, max_tokens)

    if text == "":
        return []

    sentences = _split_sentences(text)

    # Expand any sentence that alone exceeds max_tokens into smaller units.
    units: list[str] = []
    for sentence in sentences:
        if counter(sentence) > max_tokens:
            units.extend(_split_oversized_sentence(sentence, counter, max_tokens))
        else:
            units.append(sentence)

    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0
    for unit in units:
        unit_tokens = counter(unit)
        if current_parts and current_tokens + unit_tokens > max_tokens:
            chunks.append(" ".join(current_parts))
            current_parts = [unit]
            current_tokens = unit_tokens
        else:
            current_parts.append(unit)
            current_tokens += unit_tokens
    if current_parts:
        chunks.append(" ".join(current_parts))

    # Merge an undersized trailing chunk into its predecessor when it fits.
    if len(chunks) >= 2 and counter(chunks[-1]) < min_tokens:
        merged = chunks[-2] + " " + chunks[-1]
        if counter(merged) <= max_tokens:
            chunks[-2] = merged
            chunks.pop()

    return chunks