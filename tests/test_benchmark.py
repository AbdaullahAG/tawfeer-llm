"""Unit tests for ar_tokenwise.benchmark."""

from pathlib import Path

import pytest

from ar_tokenwise.benchmark import (
    BenchmarkCategory,
    CorpusEntry,
    load_corpus,
    render_markdown_table,
    run_benchmark,
)


def _word_count_counter(text: str) -> int:
    """Deterministic fake counter: 1 'token' per whitespace-separated word."""
    return len(text.split())


def test_load_corpus_parses_valid_jsonl(tmp_path: Path) -> None:
    corpus_file = tmp_path / "corpus.jsonl"
    corpus_file.write_text(
        '{"id": "a", "category": "msa", "text": "مرحبا بالعالم"}\n'
        '{"id": "b", "category": "dialect", "region": "gulf", "text": "شلونك"}\n',
        encoding="utf-8",
    )
    entries = load_corpus(corpus_file)

    assert len(entries) == 2
    assert entries[0].category == BenchmarkCategory.MSA
    assert entries[1].region == "gulf"


def test_load_corpus_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_corpus("does/not/exist.jsonl")


def test_load_corpus_invalid_json_raises(tmp_path: Path) -> None:
    corpus_file = tmp_path / "corpus.jsonl"
    corpus_file.write_text("not valid json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_corpus(corpus_file)


def test_load_corpus_missing_field_raises(tmp_path: Path) -> None:
    corpus_file = tmp_path / "corpus.jsonl"
    corpus_file.write_text('{"id": "a", "text": "نص"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing field 'category'"):
        load_corpus(corpus_file)


def test_load_corpus_invalid_category_raises(tmp_path: Path) -> None:
    corpus_file = tmp_path / "corpus.jsonl"
    corpus_file.write_text(
        '{"id": "a", "category": "klingon", "text": "نص"}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="invalid category"):
        load_corpus(corpus_file)


def test_load_corpus_empty_text_raises(tmp_path: Path) -> None:
    corpus_file = tmp_path / "corpus.jsonl"
    corpus_file.write_text('{"id": "a", "category": "msa", "text": "  "}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="empty text"):
        load_corpus(corpus_file)


def test_load_corpus_respects_max_entries_guard(tmp_path: Path) -> None:
    corpus_file = tmp_path / "corpus.jsonl"
    line = '{"id": "a", "category": "msa", "text": "نص"}\n'
    corpus_file.write_text(line * 5, encoding="utf-8")
    with pytest.raises(ValueError, match="exceeding max_entries"):
        load_corpus(corpus_file, max_entries=3)


def test_run_benchmark_groups_dialects_by_region() -> None:
    entries = [
        CorpusEntry(id="1", category=BenchmarkCategory.DIALECT, region="gulf", text="شلونك اليوم"),
        CorpusEntry(id="2", category=BenchmarkCategory.DIALECT, region="egyptian", text="إزيك يا صاحبي"),
        CorpusEntry(id="3", category=BenchmarkCategory.MSA, text="مرحبا بكم جميعا"),
    ]
    results = run_benchmark(entries, counter=_word_count_counter)
    groups = {r.group for r in results}

    assert groups == {"dialect:gulf", "dialect:egyptian", "msa"}


def test_run_benchmark_sorted_deterministically() -> None:
    entries = [
        CorpusEntry(id="1", category=BenchmarkCategory.MIXED, text="update النظام"),
        CorpusEntry(id="2", category=BenchmarkCategory.FORMAL, text="يلتزم الطرف الأول"),
        CorpusEntry(id="3", category=BenchmarkCategory.MSA, text="نص فصيح"),
    ]
    results = run_benchmark(entries, counter=_word_count_counter)
    assert [r.group for r in results] == ["formal", "mixed", "msa"]


def test_render_markdown_table_contains_all_groups() -> None:
    entries = [CorpusEntry(id="1", category=BenchmarkCategory.MSA, text="نص فصيح هنا")]
    results = run_benchmark(entries, counter=_word_count_counter)
    table = render_markdown_table(results)

    assert "msa" in table
    assert "|" in table  # basic markdown table sanity check