# tests/test_report.py
"""Unit tests for ar_tokenwise.report."""

import builtins

import pytest

from ar_tokenwise.report import get_default_counter, report_savings


def _word_count_counter(text: str) -> int:
    """Deterministic fake counter: 1 'token' per whitespace-separated word."""
    return len(text.split())


def test_report_computes_savings_and_percent() -> None:
    original = "هذا نص طويل جدا يحتاج تحسين"
    optimized = "هذا نص طويل يحتاج تحسين"
    report = report_savings(original, optimized, counter=_word_count_counter)

    assert report.original_tokens == 6
    assert report.optimized_tokens == 5
    assert report.tokens_saved == 1
    assert report.percent_saved == pytest.approx(16.666, rel=1e-2)


def test_report_zero_original_tokens_no_division_error() -> None:
    report = report_savings("", "", counter=_word_count_counter)
    assert report.original_tokens == 0
    assert report.percent_saved == 0.0
    assert report.original_fertility == 0.0


def test_report_fertility_calculation() -> None:
    # 4 words, fake counter returns 1 token/word -> fertility 1.0
    report = report_savings(
        "كلمة كلمة كلمة كلمة", "كلمة كلمة كلمة", counter=_word_count_counter
    )
    assert report.original_fertility == pytest.approx(1.0)
    assert report.optimized_fertility == pytest.approx(1.0)


def test_report_estimated_cost_savings_when_price_given() -> None:
    report = report_savings(
        "واحد اثنين ثلاثة اربعة",
        "واحد اثنين ثلاثة",
        counter=_word_count_counter,
        cost_per_million_tokens=10.0,
    )
    # 1 token saved out of 1,000,000 * $10 price
    assert report.estimated_cost_savings_usd == pytest.approx(0.00001)


def test_report_no_cost_when_price_not_given() -> None:
    report = report_savings("نص", "نص", counter=_word_count_counter)
    assert report.estimated_cost_savings_usd is None


def test_report_rejects_non_str_input() -> None:
    with pytest.raises(TypeError):
        report_savings(123, "نص", counter=_word_count_counter)  # type: ignore[arg-type]


def test_get_default_counter_raises_clear_error_without_tiktoken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate tiktoken missing regardless of the test environment."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "tiktoken":
            raise ImportError("simulated missing tiktoken")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="ar-tokenwise\\[tokenizers\\]"):
        get_default_counter()


def test_get_default_counter_works_when_tiktoken_installed() -> None:
    pytest.importorskip("tiktoken", reason="optional dependency not installed")
    counter = get_default_counter()
    assert counter("hello") > 0