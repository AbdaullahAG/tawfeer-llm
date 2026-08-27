# tests/test_run_real_world_benchmark.py
"""Unit tests for benchmark/run_real_world_benchmark.py's non-network logic.

This script lives outside the installed package (benchmark/, not
src/), so it's loaded dynamically via importlib -- same pattern as
tests/test_check_corpus_quality.py.

What's NOT tested here (and can't be, in this environment): actually
downloading FLORES-200 via the `datasets` library, and real Anthropic/
Gemini API calls. Those need network access this sandbox doesn't have
and real API credentials. Everything else -- the statistics, the
skip-reason messaging, the table rendering -- is tested against real
logic, not mocked away.
"""

import importlib.util
from pathlib import Path

import pytest

from ar_tokenwise.normalize import NormalizationLevel

BENCHMARK_DIR = Path(__file__).parent.parent / "benchmark"


def _load_module(filename: str):
    module_path = BENCHMARK_DIR / filename
    spec = importlib.util.spec_from_file_location(filename, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


rwb = _load_module("run_real_world_benchmark.py")


def _word_count_counter(text: str) -> int:
    """Deterministic fake counter: 1 'token' per whitespace-separated word."""
    return len(text.split())


# --- mean_and_ci95 ---------------------------------------------------


def test_mean_and_ci95_single_value_returns_degenerate_interval() -> None:
    mean, lo, hi = rwb.mean_and_ci95([5.0])
    assert mean == lo == hi == 5.0


def test_mean_and_ci95_computes_correct_mean() -> None:
    mean, lo, hi = rwb.mean_and_ci95([1.0, 2.0, 3.0, 4.0, 5.0])
    assert mean == pytest.approx(3.0)
    assert lo < mean < hi


def test_mean_and_ci95_zero_variance_gives_zero_width_interval() -> None:
    mean, lo, hi = rwb.mean_and_ci95([2.0, 2.0, 2.0, 2.0])
    assert mean == lo == hi == 2.0


def test_mean_and_ci95_wider_spread_gives_wider_interval() -> None:
    _, lo_tight, hi_tight = rwb.mean_and_ci95([2.0, 2.1, 1.9, 2.0, 2.05])
    _, lo_wide, hi_wide = rwb.mean_and_ci95([0.0, 4.0, 1.0, 3.0, 2.0])
    assert (hi_wide - lo_wide) > (hi_tight - lo_tight)


# --- compute_fertility_series --------------------------------------------


def test_compute_fertility_series_matches_word_count_counter() -> None:
    sentences = ["كلمة كلمة كلمة"]  # 3 words, fake counter -> 3 tokens
    original, optimized = rwb.compute_fertility_series(
        sentences, _word_count_counter, NormalizationLevel.LIGHT
    )
    assert original == [pytest.approx(1.0)]
    assert optimized == [pytest.approx(1.0)]


def test_compute_fertility_series_skips_empty_sentences() -> None:
    original, optimized = rwb.compute_fertility_series(
        ["", "كلمة واحدة هنا"], _word_count_counter, NormalizationLevel.LIGHT
    )
    assert len(original) == 1
    assert len(optimized) == 1


def test_compute_fertility_series_same_length_lists() -> None:
    sentences = ["نص أول هنا", "نص ثاني هنا كمان"]
    original, optimized = rwb.compute_fertility_series(
        sentences, _word_count_counter, NormalizationLevel.LIGHT
    )
    assert len(original) == len(optimized) == 2


# --- summarize_group / render_markdown_table ------------------------------


def test_summarize_group_computes_percent_saved_correctly() -> None:
    # original fertility mean 2.0, optimized mean 1.0 -> 50% saved
    stats = rwb.summarize_group("test", [2.0, 2.0], [1.0, 1.0])
    assert stats.percent_saved == pytest.approx(50.0)
    assert stats.n == 2


def test_summarize_group_zero_original_mean_no_division_error() -> None:
    stats = rwb.summarize_group("test", [0.0, 0.0], [0.0, 0.0])
    assert stats.percent_saved == 0.0


def test_render_markdown_table_contains_all_groups() -> None:
    groups = [
        rwb.summarize_group("group_a", [2.0, 2.0], [1.5, 1.5]),
        rwb.summarize_group("group_b", [3.0, 3.0], [3.0, 3.0]),
    ]
    table = rwb.render_markdown_table(groups)
    assert "group_a" in table
    assert "group_b" in table
    assert "|" in table


# --- build_available_counters: skip-reason messaging ----------------------


def test_build_available_counters_reports_missing_anthropic_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    counters = rwb.build_available_counters(anthropic_model="claude-opus-4-8", gemini_model=None)
    _, note = counters["anthropic"]
    assert "ANTHROPIC_API_KEY not set" in note


def test_build_available_counters_reports_missing_model_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    counters = rwb.build_available_counters(anthropic_model=None, gemini_model=None)
    _, note = counters["anthropic"]
    assert "--anthropic-model not given" in note


def test_build_available_counters_reports_missing_gemini_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    counters = rwb.build_available_counters(anthropic_model=None, gemini_model="gemini-3-flash")
    _, note = counters["gemini"]
    assert "GEMINI_API_KEY/GOOGLE_API_KEY not set" in note


def test_build_available_counters_always_includes_tiktoken_entry() -> None:
    counters = rwb.build_available_counters(anthropic_model=None, gemini_model=None)
    assert "tiktoken_o200k_base" in counters


# --- load_flores_sentences: only the ImportError path is testable here ----


def test_load_flores_sentences_raises_clear_error_without_datasets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "datasets":
            raise ImportError("simulated missing datasets")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="pip install datasets"):
        rwb.load_flores_sentences("arb_Arab")