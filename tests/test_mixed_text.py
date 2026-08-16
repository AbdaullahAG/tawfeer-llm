"""Unit tests for ar_tokenwise.mixed_text."""

import pytest

from ar_tokenwise.mixed_text import (
    WordCategory,
    classify_word,
    report_mixed_fertility,
)


def _word_count_counter(text: str) -> int:
    """Deterministic fake counter: 1 'token' per whitespace-separated word."""
    return len(text.split())


# --- required minimum cases: empty / oversized / wrong type -----------


def test_report_empty_text_returns_all_zeros() -> None:
    report = report_mixed_fertility("", counter=_word_count_counter)
    assert report.total_words == 0
    assert report.arabic_word_count == 0
    assert report.non_arabic_word_count == 0
    assert report.mixed_word_count == 0
    assert report.arabic_fertility == 0.0


def test_report_oversized_input_raises_value_error() -> None:
    with pytest.raises(ValueError, match="exceeds max_length"):
        report_mixed_fertility("نص " * 10, counter=_word_count_counter, max_length=5)


def test_report_non_str_input_raises_type_error() -> None:
    with pytest.raises(TypeError):
        report_mixed_fertility(123, counter=_word_count_counter)  # type: ignore[arg-type]


# --- classify_word: script detection, digits ignored --------------------


def test_classify_pure_arabic_word() -> None:
    assert classify_word("مرحبا") == WordCategory.ARABIC


def test_classify_pure_latin_word() -> None:
    assert classify_word("hello") == WordCategory.NON_ARABIC


def test_classify_latin_word_with_attached_latin_digits() -> None:
    # Digits must not cause a false ARABIC/MIXED classification.
    assert classify_word("iPhone15") == WordCategory.NON_ARABIC


def test_classify_arabic_word_with_attached_arabic_indic_digits() -> None:
    # Arabic-indic digits are excluded from the letter check too.
    assert classify_word("سنة٢٠٢٦") == WordCategory.ARABIC


def test_classify_pure_digits_is_non_arabic() -> None:
    assert classify_word("2026") == WordCategory.NON_ARABIC
    assert classify_word("٢٠٢٦") == WordCategory.NON_ARABIC


def test_classify_genuinely_mixed_word() -> None:
    # Arabic prefix glued directly to a Latin term, no space.
    assert classify_word("بـCOVID") == WordCategory.MIXED


def test_classify_punctuation_only_is_non_arabic() -> None:
    assert classify_word("...") == WordCategory.NON_ARABIC


# --- report_mixed_fertility: bucketing and fertility ---------------------


def test_report_separates_arabic_and_non_arabic_fertility() -> None:
    text = "هذا نص عربي جميل and this is English text"
    report = report_mixed_fertility(text, counter=_word_count_counter)

    assert report.total_words == 9
    assert report.arabic_word_count == 4
    assert report.non_arabic_word_count == 5
    assert report.mixed_word_count == 0
    # Fake counter: 1 token per word -> fertility exactly 1.0 per bucket.
    assert report.arabic_fertility == pytest.approx(1.0)
    assert report.non_arabic_fertility == pytest.approx(1.0)


def test_report_zero_words_in_a_category_gives_zero_fertility_not_error() -> None:
    report = report_mixed_fertility("مرحبا بكم جميعا", counter=_word_count_counter)
    assert report.non_arabic_word_count == 0
    assert report.non_arabic_fertility == 0.0
    assert report.mixed_word_count == 0
    assert report.mixed_fertility == 0.0


def test_report_counts_mixed_words_separately() -> None:
    text = "استخدمنا بـCOVID اختبار"
    report = report_mixed_fertility(text, counter=_word_count_counter)
    assert report.mixed_word_count == 1
    assert report.arabic_word_count == 2
    assert report.non_arabic_word_count == 0


def test_report_digits_do_not_inflate_non_arabic_count_incorrectly() -> None:
    # A pure Arabic sentence with a trailing Latin-digit year should not
    # misclassify the digit token or the Arabic words around it.
    text = "بدأ المشروع سنة 2026 بنجاح"
    report = report_mixed_fertility(text, counter=_word_count_counter)
    assert report.arabic_word_count == 4  # بدأ, المشروع, سنة, بنجاح
    assert report.non_arabic_word_count == 1  # 2026
    assert report.mixed_word_count == 0