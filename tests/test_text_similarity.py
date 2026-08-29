"""Unit tests for benchmark/_text_similarity.py."""

import importlib.util
from pathlib import Path

import pytest

BENCHMARK_DIR = Path(__file__).parent.parent / "benchmark"


def _load_module(filename: str):
    module_path = BENCHMARK_DIR / filename
    spec = importlib.util.spec_from_file_location(filename, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


ts = _load_module("_text_similarity.py")


# --- normalize_answer_for_comparison --------------------------------------


def test_normalize_strips_punctuation() -> None:
    assert ts.normalize_answer_for_comparison("مرحبا، كيف حالك؟") == "مرحبا كيف حالك"


def test_normalize_strips_diacritics() -> None:
    assert ts.normalize_answer_for_comparison("مَرْحَـبًا") == "مرحبا"


def test_normalize_collapses_whitespace() -> None:
    assert ts.normalize_answer_for_comparison("كلمة   كلمة") == "كلمة كلمة"


# --- f1_score: required minimum cases + real logic -----------------------


def test_f1_identical_strings_is_one() -> None:
    assert ts.f1_score("القاهرة", "القاهرة") == 1.0


def test_f1_completely_different_is_zero() -> None:
    assert ts.f1_score("القاهرة", "دمشق") == 0.0


def test_f1_both_empty_is_one() -> None:
    assert ts.f1_score("", "") == 1.0


def test_f1_one_empty_one_not_is_zero() -> None:
    assert ts.f1_score("القاهرة", "") == 0.0
    assert ts.f1_score("", "القاهرة") == 0.0


def test_f1_partial_overlap() -> None:
    # predicted "مدينة القاهرة الكبرى" (3 tokens) vs gold "القاهرة" (1 token)
    # overlap=1, precision=1/3, recall=1/1, F1 = 2*(1/3*1)/(1/3+1) = 0.5
    score = ts.f1_score("مدينة القاهرة الكبرى", "القاهرة")
    assert score == pytest.approx(0.5)


def test_f1_ignores_diacritic_differences() -> None:
    # Same content, one diacritized -- should score as if identical.
    assert ts.f1_score("مَرْحَـبًا بكم", "مرحبا بكم") == 1.0


def test_f1_ignores_punctuation_differences() -> None:
    assert ts.f1_score("القاهرة.", "القاهرة") == 1.0


# --- exact_match -----------------------------------------------------


def test_exact_match_true_for_identical_after_normalization() -> None:
    assert ts.exact_match("مَرْحَـبًا،", "مرحبا") is True


def test_exact_match_false_for_different_content() -> None:
    assert ts.exact_match("القاهرة", "دمشق") is False