"""Prompt structuring for LLM provider prompt caching (Anthropic, OpenAI, ...).

Provider prompt caching (Anthropic's explicit cache_control breakpoints,
OpenAI's automatic prefix caching) only produces cache hits when the
cached prefix is BYTE-IDENTICAL across requests, and caching applies to
a PREFIX -- never the middle or end of a prompt (verified against
Anthropic's prompt-caching documentation, mid-2026). This means prompt
STRUCTURE matters as much as content: the same long system prompt/
context placed at the START of every request (identical bytes each
time), with only the request-specific/varying text at the END, is what
actually earns the cache discount. A stable block placed after a
varying block, or interleaved with one, breaks the shared prefix and
the cache is never hit.

This module does NOT understand semantics or meaning -- it only
reorders segments YOU label as "stable" (identical across calls) vs
"dynamic" (varies per call), moving all stable segments before all
dynamic ones while preserving each group's original relative order (a
stable sort, not a semantic rewrite). If reordering your segments would
change the prompt's meaning, do not mark them for reordering -- this
tool trusts your labels completely and has no way to verify them.

WHY NO HARDCODED TOKEN THRESHOLDS OR PRICING: Anthropic's minimum
cacheable prompt length varies by model (512 to 4,096 tokens as of this
writing) and per-token cache pricing multipliers change independently
of this library's release cycle. Baking either into this module would
go stale exactly like a hardcoded model-pricing table would (see
report.py's docstring for the same principle) -- check your provider's
current documentation for the numbers that apply to your model.
"""

from __future__ import annotations

from dataclasses import dataclass

from ar_tokenwise._internal import validate_text_input
from ar_tokenwise.normalize import DEFAULT_MAX_LENGTH


@dataclass(frozen=True)
class PromptSegment:
    """One piece of a prompt, labeled by the caller as stable or dynamic.

    Attributes:
        content: The segment's text.
        stable: True if this content is IDENTICAL across repeated calls
            (e.g. a system prompt, a fixed set of few-shot examples, a
            document that doesn't change between requests). False if it
            varies per call (e.g. the current user message, a
            timestamp, retrieved context that changes per query).
    """

    content: str
    stable: bool


@dataclass(frozen=True)
class CacheOptimizedPrompt:
    """Result of reordering prompt segments for provider prompt caching.

    Attributes:
        segments: The input segments reordered -- all stable segments
            first (in their original relative order), then all dynamic
            segments (in their original relative order).
        text: ``segments`` joined with the separator used.
        stable_prefix_length: Character length of the stable portion of
            ``text`` -- guaranteed that ``text[:stable_prefix_length]``
            equals the stable segments joined alone, so this is a safe
            cut point for e.g. deciding where to place a cache boundary
            marker in a provider-specific format.
    """

    segments: list[PromptSegment]
    text: str
    stable_prefix_length: int


def _validate_segments(segments: list[PromptSegment], max_length: int) -> None:
    """Validate the segments list before any processing.

    Raises:
        TypeError: if ``segments`` is not a list, or any element is not
            a :class:`PromptSegment`.
        ValueError: if any segment's content exceeds ``max_length``.
    """
    if not isinstance(segments, list):
        raise TypeError(f"optimize_for_caching() expects list, got {type(segments).__name__}")
    for segment in segments:
        if not isinstance(segment, PromptSegment):
            raise TypeError(
                f"each segment must be a PromptSegment, got {type(segment).__name__}"
            )
        validate_text_input(segment.content, max_length, caller_name="optimize_for_caching")


def optimize_for_caching(
    segments: list[PromptSegment],
    separator: str = "\n\n",
    max_length: int = DEFAULT_MAX_LENGTH,
) -> CacheOptimizedPrompt:
    """Reorder prompt segments so all stable content precedes all dynamic content.

    Uses a stable sort: segments within the "stable" group and within
    the "dynamic" group each keep their original relative order --
    only the boundary between the two groups is what changes.

    Args:
        segments: Prompt pieces, each labeled stable/dynamic by the
            caller. An empty list is valid and returns an empty result.
        separator: String joined between segments to build ``text``.
        max_length: Maximum accepted length in characters for any
            single segment's content, used as a size-based safety guard.

    Returns:
        A :class:`CacheOptimizedPrompt`.

    Raises:
        TypeError: if ``segments`` is not a list of :class:`PromptSegment`.
        ValueError: if any segment's content exceeds ``max_length``.
    """
    _validate_segments(segments, max_length)

    if not segments:
        return CacheOptimizedPrompt(segments=[], text="", stable_prefix_length=0)

    # Stable sort: False (0) for stable segments sorts before True (1)
    # for dynamic ones, so stable segments end up first. Python's sort
    # is guaranteed stable, so relative order within each group survives.
    reordered = sorted(segments, key=lambda s: not s.stable)

    text = separator.join(s.content for s in reordered)

    stable_segments = [s for s in reordered if s.stable]
    stable_prefix_length = (
        len(separator.join(s.content for s in stable_segments)) if stable_segments else 0
    )

    return CacheOptimizedPrompt(
        segments=reordered,
        text=text,
        stable_prefix_length=stable_prefix_length,
    )


def to_anthropic_cache_blocks(
    optimized: CacheOptimizedPrompt,
    ttl: str = "5m",
) -> list[dict[str, object]]:
    """Build Anthropic-style content blocks with a cache_control breakpoint.

    Produces a list of ``{"type": "text", "text": ...}`` blocks matching
    Anthropic's content-block format, with ``cache_control`` placed on
    the LAST stable block -- Anthropic caches everything up to and
    including the marked block, so the breakpoint must sit at the end
    of the shared prefix, never on a block that varies per request (see
    module docstring). If there are no stable segments, no block gets a
    cache_control marker, since there is nothing safe to cache.

    This builds plain dicts, not SDK objects -- no ``anthropic`` package
    dependency, use these as the ``content`` list for a system or user
    message with whichever client you already have.

    Args:
        optimized: Result of :func:`optimize_for_caching`.
        ttl: ``"5m"`` (default, omits the ``ttl`` key entirely, matching
            Anthropic's documented default-TTL shape) or ``"1h"``
            (adds ``"ttl": "1h"`` to the marker, per Anthropic's
            extended-cache format).

    Returns:
        A list of content-block dicts.

    Raises:
        ValueError: if ``ttl`` is not ``"5m"`` or ``"1h"``.
    """
    if ttl not in ("5m", "1h"):
        raise ValueError(f"ttl must be '5m' or '1h', got {ttl!r}")

    last_stable_index = None
    for index, segment in enumerate(optimized.segments):
        if segment.stable:
            last_stable_index = index

    blocks: list[dict[str, object]] = []
    for index, segment in enumerate(optimized.segments):
        block: dict[str, object] = {"type": "text", "text": segment.content}
        if index == last_stable_index:
            cache_control: dict[str, str] = {"type": "ephemeral"}
            if ttl == "1h":
                cache_control["ttl"] = "1h"
            block["cache_control"] = cache_control
        blocks.append(block)

    return blocks