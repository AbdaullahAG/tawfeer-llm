"""Shared internal utilities used across multiple ar_tokenwise modules.

Not part of the public API (no leading underscore convention applies to
the whole module, not just symbols within it) -- these are implementation
details shared by dialect.py and safety_modes.py, and may change without
notice. External code should not import from here.

This module exists to fix a real bug that existed in two places at once:
dialect.py and safety_modes.py both used plain substring matching
(`marker in text`) to check for marker phrases, which matches inside
unrelated words -- e.g. the Egyptian marker "مش" matched inside "لمشوار"
("for an errand"), and the legal marker "بند" matched inside "بندر"
(a personal name). Both modules now share ONE word-boundary-aware
matching implementation, so this class of bug can only exist in one
place, not silently re-introduced in either module independently.
"""

from __future__ import annotations

import re


def compile_marker_pattern(marker: str) -> re.Pattern[str]:
    """Compile a word-boundary-aware regex for a single marker phrase.

    The pattern is the marker text escaped literally (so it can never be
    interpreted as regex syntax) wrapped in \\b anchors. Python's `re`
    module treats \\w -- and therefore \\b -- as Unicode-aware by default
    for str patterns, so this correctly treats Arabic letters as "word"
    characters (verified directly: "مش" no longer matches inside
    "لمشوار", but still matches the standalone word "مش").

    Args:
        marker: The literal marker phrase (single word or short phrase).

    Returns:
        A compiled pattern matching ``marker`` only at word boundaries.
    """
    return re.compile(rf"\b{re.escape(marker)}\b")


def compile_markers_by_category(
    markers_by_category: dict[object, list[str]],
) -> dict[object, list[re.Pattern[str]]]:
    """Compile every marker in a category->markers mapping, once.

    Intended to be called at module import time (not per-call) so the
    per-call matching cost is just running already-compiled patterns.

    Args:
        markers_by_category: Mapping of category key (any hashable, e.g.
            an Enum member) to a list of raw marker strings.

    Returns:
        The same mapping shape, with each marker string replaced by its
        compiled word-boundary pattern.
    """
    return {
        category: [compile_marker_pattern(marker) for marker in markers]
        for category, markers in markers_by_category.items()
    }


def count_marker_matches(
    text: str,
    compiled_markers_by_category: dict[object, list[re.Pattern[str]]],
) -> dict[object, int]:
    """Count word-boundary marker matches per category in ``text``.

    Args:
        text: Text to search (callers are expected to have already
            applied any normalization, e.g. LIGHT-level diacritic
            stripping, before calling this).
        compiled_markers_by_category: Output of
            :func:`compile_markers_by_category`.

    Returns:
        A mapping of category -> number of distinct markers in that
        category that matched at least once in ``text``.
    """
    return {
        category: sum(1 for pattern in patterns if pattern.search(text))
        for category, patterns in compiled_markers_by_category.items()
    }


def validate_text_input(text: str, max_length: int, caller_name: str) -> None:
    """Shared input validation used by every public text-processing function.

    Args:
        text: The value to validate.
        max_length: Maximum accepted length in characters.
        caller_name: Name of the calling public function, used only to
            make the error message point at the right place (e.g.
            "normalize" -> "normalize() expects str, got ...").

    Raises:
        TypeError: if ``text`` is not a ``str``.
        ValueError: if ``text`` exceeds ``max_length`` characters.
    """
    if not isinstance(text, str):
        raise TypeError(f"{caller_name}() expects str, got {type(text).__name__}")
    if len(text) > max_length:
        raise ValueError(
            f"Input length {len(text)} exceeds max_length={max_length}. "
            "Split the text or raise max_length explicitly if intentional."
        )