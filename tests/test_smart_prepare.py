"""Unit tests for ar_tokenwise.smart_prepare."""

import pytest

from ar_tokenwise.smart_prepare import smart_prepare


# --- required minimum cases: empty / oversized / wrong type -----------


def test_smart_prepare_empty_text() -> None:
    result = smart_prepare("")
    assert result.text == ""
    assert result.warnings == []
    assert result.was_normalized is True


def test_smart_prepare_oversized_input_raises_value_error() -> None:
    with pytest.raises(ValueError, match="exceeds max_length"):
        smart_prepare("كلمة " * 10, max_length=5)


def test_smart_prepare_non_str_input_raises_type_error() -> None:
    with pytest.raises(TypeError):
        smart_prepare(123)  # type: ignore[arg-type]


# --- core policy -----------------------------------------------------


def test_smart_prepare_normalizes_ordinary_text() -> None:
    result = smart_prepare("مَرْحَـبًا بكم")
    assert result.text == "مرحبا بكم"
    assert result.warnings == []
    assert result.was_normalized is True


def test_smart_prepare_skips_normalization_when_warning_fires() -> None:
    # "بند" (legal marker) with diacritics -- normalize() would strip
    # them, but smart_prepare must return the ORIGINAL text unchanged
    # here since a warning fires.
    text = "هَذا بَنْدٌ مُهِمٌّ بِالنَّصِّ"
    result = smart_prepare(text)

    assert result.text == text  # unchanged, diacritics intact
    assert result.was_normalized is False
    assert len(result.warnings) == 1
    assert result.warnings[0].category.value == "legal"


def test_smart_prepare_respects_custom_normalization_level() -> None:
    result = smart_prepare("أحمد", level=__import__("ar_tokenwise").NormalizationLevel.MEDIUM)
    assert result.text == "احمد"  # alef variant unified, MEDIUM-only behavior
    assert result.was_normalized is True


def test_smart_prepare_result_is_never_none_for_warnings_list() -> None:
    # Sanity check on the dataclass contract: warnings is always a list,
    # never None, so callers can safely do `if result.warnings:`.
    result = smart_prepare("نص عادي بدون أي إشارات")
    assert isinstance(result.warnings, list)