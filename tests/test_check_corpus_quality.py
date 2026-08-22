"""Unit tests for benchmark/check_corpus_quality.py's core logic.

This script lives outside the installed package (benchmark/, not
src/), so it's loaded dynamically via importlib -- same pattern as
tests/test_skill_scripts.py for skill/scripts/.
"""

import importlib.util
from pathlib import Path

from ar_tokenwise.benchmark import BenchmarkCategory, CorpusEntry

BENCHMARK_DIR = Path(__file__).parent.parent / "benchmark"


def _load_module(filename: str):
    module_path = BENCHMARK_DIR / filename
    spec = importlib.util.spec_from_file_location(filename, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


check_corpus_quality = _load_module("check_corpus_quality.py")


def test_jaccard_similarity_identical_sentences_is_one() -> None:
    assert check_corpus_quality._jaccard_similarity("hello world", "hello world") == 1.0


def test_jaccard_similarity_completely_different_is_zero() -> None:
    assert check_corpus_quality._jaccard_similarity("hello world", "foo bar") == 0.0


def test_jaccard_similarity_partial_overlap() -> None:
    # "a b c" vs "a b d": intersection {a, b} = 2, union {a, b, c, d} = 4
    assert check_corpus_quality._jaccard_similarity("a b c", "a b d") == 0.5


def test_find_near_duplicates_flags_similar_pair_within_same_group() -> None:
    entries = [
        CorpusEntry(id="a", category=BenchmarkCategory.MSA, text="نص طويل هنا اليوم"),
        CorpusEntry(id="b", category=BenchmarkCategory.MSA, text="نص طويل هنا غدا"),
    ]
    flagged = check_corpus_quality.find_near_duplicates(entries, threshold=0.5)
    assert len(flagged) == 1
    assert flagged[0][0] == "a"
    assert flagged[0][1] == "b"


def test_find_near_duplicates_does_not_cross_groups() -> None:
    entries = [
        CorpusEntry(id="a", category=BenchmarkCategory.MSA, text="نص طويل هنا اليوم"),
        CorpusEntry(
            id="b", category=BenchmarkCategory.DIALECT, region="gulf",
            text="نص طويل هنا اليوم",
        ),
    ]
    # Identical text, but different (category, region) groups -- should
    # not be flagged by find_near_duplicates (that's what
    # find_cross_dialect_duplicates is for, tested separately below).
    flagged = check_corpus_quality.find_near_duplicates(entries, threshold=0.5)
    assert flagged == []


def test_find_cross_dialect_duplicates_flags_translated_content() -> None:
    entries = [
        CorpusEntry(
            id="gulf-1", category=BenchmarkCategory.DIALECT, region="gulf",
            text="السيارة تعطلت اليوم بالطريق",
        ),
        CorpusEntry(
            id="egy-1", category=BenchmarkCategory.DIALECT, region="egyptian",
            text="السيارة عطلت النهارده بالطريق",
        ),
    ]
    # Literal shared words here: "السيارة" and "بالطريق".
    flagged = check_corpus_quality.find_cross_dialect_duplicates(entries, threshold=0.2)
    assert len(flagged) == 1


def test_find_cross_dialect_duplicates_ignores_same_region_pairs() -> None:
    entries = [
        CorpusEntry(
            id="gulf-1", category=BenchmarkCategory.DIALECT, region="gulf", text="نفس النص",
        ),
        CorpusEntry(
            id="gulf-2", category=BenchmarkCategory.DIALECT, region="gulf", text="نفس النص",
        ),
    ]
    # Same region -- already covered by find_near_duplicates, must not
    # be double-flagged here.
    flagged = check_corpus_quality.find_cross_dialect_duplicates(entries, threshold=0.5)
    assert flagged == []


def test_find_cross_dialect_duplicates_ignores_non_dialect_categories() -> None:
    entries = [
        CorpusEntry(id="a", category=BenchmarkCategory.MSA, text="نص مشترك هنا"),
        CorpusEntry(id="b", category=BenchmarkCategory.FORMAL, text="نص مشترك هنا"),
    ]
    flagged = check_corpus_quality.find_cross_dialect_duplicates(entries, threshold=0.5)
    assert flagged == []


def test_report_length_distribution_computes_correct_stats() -> None:
    entries = [
        CorpusEntry(id="a", category=BenchmarkCategory.MSA, text="one two three"),
        CorpusEntry(id="b", category=BenchmarkCategory.MSA, text="one two three four five"),
    ]
    report = check_corpus_quality.report_length_distribution(entries)
    assert "3" in report  # min
    assert "5" in report  # max
    assert "4.0" in report  # avg