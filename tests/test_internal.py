"""Unit tests for ar_tokenwise._internal."""

import re

import pytest

from ar_tokenwise._internal import (
    compile_marker_pattern,
    compile_markers_by_category,
    count_marker_matches,
    validate_text_input,
)


# --- compile_marker_pattern: the actual bug fix ---------------------------


def test_marker_does_not_match_inside_unrelated_word() -> None:
    # The exact false positive from the audit: "مش" (Egyptian marker)
    # must not match inside "لمشوار" ("for an errand").
    pattern = compile_marker_pattern("مش")
    assert pattern.search("لمشوار طويل جدا") is None


def test_marker_still_matches_as_standalone_word() -> None:
    pattern = compile_marker_pattern("مش")
    assert pattern.search("انا مش رايح") is not None


def test_legal_marker_does_not_match_inside_a_name() -> None:
    # The exact false positive from the audit: "بند" (legal marker) must
    # not match inside "بندر" (a personal name).
    pattern = compile_marker_pattern("بند")
    assert pattern.search("التقيت بالسيد بندر في المطار") is None


def test_legal_marker_still_matches_as_standalone_word() -> None:
    pattern = compile_marker_pattern("بند")
    assert pattern.search("هذا بند مهم بالعقد") is not None


def test_multi_word_marker_still_matches() -> None:
    pattern = compile_marker_pattern("لو سمحت")
    assert pattern.search("لو سمحت ممكن تساعدني") is not None


def test_marker_with_regex_special_characters_is_escaped() -> None:
    # A marker containing a character with regex meaning must be treated
    # literally, not as regex syntax.
    pattern = compile_marker_pattern("a.b")
    assert pattern.search("a.b test") is not None
    assert pattern.search("axb test") is None  # "." must not mean "any char"


# --- compile_markers_by_category / count_marker_matches -------------------


def test_compile_markers_by_category_preserves_structure() -> None:
    raw = {"cat_a": ["one", "two"], "cat_b": ["three"]}
    compiled = compile_markers_by_category(raw)

    assert set(compiled.keys()) == {"cat_a", "cat_b"}
    assert len(compiled["cat_a"]) == 2
    assert all(isinstance(p, re.Pattern) for p in compiled["cat_a"])


def test_count_marker_matches_counts_distinct_markers_per_category() -> None:
    compiled = compile_markers_by_category({"cat_a": ["hello", "world"], "cat_b": ["foo"]})
    counts = count_marker_matches("hello there, hello again, world", compiled)

    assert counts["cat_a"] == 2  # "hello" and "world" both matched (once each, distinct)
    assert counts["cat_b"] == 0


def test_count_marker_matches_zero_for_no_match() -> None:
    compiled = compile_markers_by_category({"cat_a": ["xyz"]})
    counts = count_marker_matches("completely unrelated text", compiled)
    assert counts["cat_a"] == 0


# --- validate_text_input: required minimum cases -------------------------


def test_validate_accepts_valid_string_within_length() -> None:
    validate_text_input("hello", max_length=100, caller_name="test_fn")  # no raise


def test_validate_non_str_raises_type_error_with_caller_name() -> None:
    with pytest.raises(TypeError, match="test_fn"):
        validate_text_input(123, max_length=100, caller_name="test_fn")  # type: ignore[arg-type]


def test_validate_oversized_raises_value_error() -> None:
    with pytest.raises(ValueError, match="exceeds max_length"):
        validate_text_input("x" * 10, max_length=5, caller_name="test_fn")


def test_validate_empty_string_within_length_does_not_raise() -> None:
    validate_text_input("", max_length=100, caller_name="test_fn")  # no raise