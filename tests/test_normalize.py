"""Unit tests for ar_tokenwise.normalize."""

import pytest

from ar_tokenwise.normalize import NormalizationLevel, normalize


def test_empty_string_returns_empty() -> None:
    assert normalize("") == ""


def test_non_str_input_raises_type_error() -> None:
    with pytest.raises(TypeError):
        normalize(123)  # type: ignore[arg-type]


def test_oversized_input_raises_value_error() -> None:
    with pytest.raises(ValueError):
        normalize("ا" * 10, max_length=5)


def test_light_strips_tashkeel() -> None:
    # مَرْحَبًا (with diacritics) -> مرحبا
    assert normalize("مَرْحَبًا", NormalizationLevel.LIGHT) == "مرحبا"


def test_light_strips_tatweel() -> None:
    assert normalize("مرحـــبا", NormalizationLevel.LIGHT) == "مرحبا"


def test_light_normalizes_arabic_indic_digits() -> None:
    assert normalize("٢٠٢٦", NormalizationLevel.LIGHT) == "2026"


def test_light_normalizes_persian_digits() -> None:
    assert normalize("۲۰۲۶", NormalizationLevel.LIGHT) == "2026"


def test_light_does_not_unify_alef_variants() -> None:
    # LIGHT must leave orthographic alef distinctions untouched.
    assert normalize("أحمد", NormalizationLevel.LIGHT) == "أحمد"


def test_medium_unifies_alef_variants() -> None:
    for variant in ["أحمد", "إحمد", "آحمد", "ٱحمد"]:
        assert normalize(variant, NormalizationLevel.MEDIUM) == "احمد"


def test_medium_unifies_alef_maksura_to_yeh() -> None:
    assert normalize("مستشفى", NormalizationLevel.MEDIUM) == "مستشفي"


def test_teh_marbuta_never_touched() -> None:
    # Meaning-bearing distinction (feminine marker) must survive both levels.
    assert normalize("مدرسة", NormalizationLevel.LIGHT) == "مدرسة"
    assert normalize("مدرسة", NormalizationLevel.MEDIUM) == "مدرسة"


def test_non_arabic_text_passes_through() -> None:
    assert normalize("Hello 123", NormalizationLevel.LIGHT) == "Hello 123"


def test_idempotent() -> None:
    text = "مَرْحَـبًا ٢٠٢٦ أحمد"
    once = normalize(text, NormalizationLevel.MEDIUM)
    twice = normalize(once, NormalizationLevel.MEDIUM)
    assert once == twice