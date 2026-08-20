"""Deterministic cache/embedding keys for Arabic text.

Diacritized and undiacritized versions of the same Arabic text (or text
using different alef/yeh orthographic variants) are semantically the same
content, but naive caching/embedding-store keys (e.g. the raw string, or
a hash of the raw string) treat them as entirely different entries -- a
documented real-world cache-miss problem for Arabic LLM pipelines.

This module provides a MORE AGGRESSIVE, cache-key-specific canonicalization
than normalize()'s LIGHT/MEDIUM levels, and a SHA-256 hash of that
canonical form suitable as a dict key, cache key, or vector-store lookup
key.

CRITICAL: the output of this module is for KEY GENERATION ONLY. Never use
canonicalize_for_cache_key()'s output as text sent to an LLM prompt --
it is more aggressive (casefolded, whitespace-collapsed) than what's
appropriate for model input. Use normalize() for that instead.

This also does not replace the guidance in SKILL.md's "RAG / retrieval
consistency warning": if you index content with one normalization level
and query with another, retrieval quality can still degrade -- this
module solves the exact-duplicate cache-key problem specifically, not
general embedding consistency.
"""

from __future__ import annotations

import hashlib
import re

from ar_tokenwise.normalize import DEFAULT_MAX_LENGTH, NormalizationLevel, normalize

# Collapses any run of whitespace to a single space, so purely
# presentational spacing differences don't produce different cache keys.
# Single quantifier, no nesting -- linear in input length, no ReDoS risk.
_WHITESPACE_COLLAPSE_PATTERN = re.compile(r"\s+")


def _validate_input(text: str, max_length: int) -> None:
    """Validate input before any processing.

    Raises:
        TypeError: if ``text`` is not a ``str``.
        ValueError: if ``text`` exceeds ``max_length`` characters.
    """
    if not isinstance(text, str):
        raise TypeError(f"generate_cache_key() expects str, got {type(text).__name__}")
    if len(text) > max_length:
        raise ValueError(
            f"Input length {len(text)} exceeds max_length={max_length}. "
            "Split the text or raise max_length explicitly if intentional."
        )


def canonicalize_for_cache_key(text: str, max_length: int = DEFAULT_MAX_LENGTH) -> str:
    """Produce an aggressively canonical form of text, for key generation only.

    Applies MEDIUM-level normalization (tashkeel/tatweel/digit unification
    plus alef/yeh orthographic unification), then collapses whitespace and
    casefolds (lowercases Latin script; a no-op on Arabic script).

    DO NOT send this output to an LLM prompt -- it is more aggressive than
    normalize()'s output and is intended purely for deterministic key
    matching. Use normalize() for actual model input.

    Args:
        text: Input text. Empty string returns an empty string.
        max_length: Maximum accepted input length in characters, used as
            a size-based safety guard. Raises if exceeded.

    Returns:
        The canonicalized string.

    Raises:
        TypeError: if ``text`` is not a string.
        ValueError: if ``text`` exceeds ``max_length``.
    """
    _validate_input(text, max_length)

    normalized = normalize(text, level=NormalizationLevel.MEDIUM, max_length=max_length)
    collapsed = _WHITESPACE_COLLAPSE_PATTERN.sub(" ", normalized).strip()
    return collapsed.casefold()


def generate_cache_key(text: str, max_length: int = DEFAULT_MAX_LENGTH) -> str:
    """Generate a deterministic SHA-256 cache/embedding key for Arabic text.

    Semantically-equivalent Arabic text -- with or without diacritics,
    with different alef/yeh spelling variants, or with different
    whitespace -- maps to the SAME key. This solves exact-duplicate
    cache misses; it does not guarantee general embedding/retrieval
    consistency (see module docstring).

    This key is one-way (a hash): the original text cannot be recovered
    from it. It is a lookup key, not encrypted storage or a security
    primitive.

    Args:
        text: Input text. Empty string is valid and returns a fixed,
            deterministic hash (the hash of an empty canonical string).
        max_length: Maximum accepted input length in characters, used as
            a size-based safety guard. Raises if exceeded.

    Returns:
        A 64-character lowercase hex SHA-256 digest string.

    Raises:
        TypeError: if ``text`` is not a string.
        ValueError: if ``text`` exceeds ``max_length``.
    """
    canonical = canonicalize_for_cache_key(text, max_length)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()