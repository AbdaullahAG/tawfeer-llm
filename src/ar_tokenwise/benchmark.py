"""Reproducible fertility benchmarking across Arabic text registers.

Given a labeled corpus (MSA, dialect, mixed, formal/legal register), this
module measures tokens-per-word (fertility) before and after normalization,
grouped by category and dialect region. No hardcoded corpus or results are
embedded here -- this module is pure measurement logic, reusable against
any caller-supplied corpus file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ar_tokenwise.normalize import NormalizationLevel, normalize
from ar_tokenwise.report import TokenCounter, _count_words, _fertility

# Guard against accidentally loading a pathological number of corpus lines.
DEFAULT_MAX_ENTRIES = 10_000


class BenchmarkCategory(str, Enum):
    """Text register categories tracked by the benchmark."""

    MSA = "msa"
    DIALECT = "dialect"
    MIXED = "mixed"
    FORMAL = "formal"


@dataclass(frozen=True)
class CorpusEntry:
    """A single labeled benchmark sentence."""

    id: str
    category: BenchmarkCategory
    text: str
    region: str | None = None  # e.g. "gulf", "egyptian", "levantine", "maghrebi"


@dataclass(frozen=True)
class BenchmarkResult:
    """Aggregated fertility stats for one category/region group."""

    group: str  # e.g. "msa" or "dialect:gulf"
    entry_count: int
    avg_original_fertility: float
    avg_optimized_fertility: float
    avg_percent_saved: float


def load_corpus(path: str | Path, max_entries: int = DEFAULT_MAX_ENTRIES) -> list[CorpusEntry]:
    """Load a JSONL corpus file into validated entries.

    Each line must be a JSON object with at least "id", "category", "text",
    and an optional "region". Invalid lines raise immediately rather than
    being silently skipped, so corpus errors are caught early.

    Args:
        path: Path to a .jsonl corpus file.
        max_entries: Safety guard against accidentally huge corpus files.

    Returns:
        List of parsed :class:`CorpusEntry` objects, in file order.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if the file exceeds ``max_entries`` lines, or a line
            is malformed (bad JSON, missing field, invalid category,
            empty text).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Corpus file not found: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    non_empty_lines = [line for line in lines if line.strip()]

    if len(non_empty_lines) > max_entries:
        raise ValueError(
            f"Corpus has {len(non_empty_lines)} entries, exceeding "
            f"max_entries={max_entries}."
        )

    entries: list[CorpusEntry] = []
    for line_number, line in enumerate(non_empty_lines, start=1):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corpus line {line_number}: invalid JSON") from exc

        for field in ("id", "category", "text"):
            if field not in raw:
                raise ValueError(f"Corpus line {line_number}: missing field '{field}'")

        if not raw["text"].strip():
            raise ValueError(f"Corpus line {line_number}: empty text")

        try:
            category = BenchmarkCategory(raw["category"])
        except ValueError as exc:
            raise ValueError(
                f"Corpus line {line_number}: invalid category '{raw['category']}'"
            ) from exc

        entries.append(
            CorpusEntry(
                id=raw["id"],
                category=category,
                text=raw["text"],
                region=raw.get("region"),
            )
        )

    return entries


def _group_key(entry: CorpusEntry) -> str:
    """Group dialect entries by region when available, others by category alone."""
    if entry.region:
        return f"{entry.category.value}:{entry.region}"
    return entry.category.value


def run_benchmark(
    entries: list[CorpusEntry],
    counter: TokenCounter,
    level: NormalizationLevel = NormalizationLevel.LIGHT,
) -> list[BenchmarkResult]:
    """Run the fertility benchmark over a corpus.

    Args:
        entries: Corpus entries, typically from :func:`load_corpus`.
        counter: Token-counting function (see ``ar_tokenwise.report``).
        level: Normalization level applied before re-measuring fertility.

    Returns:
        One :class:`BenchmarkResult` per category/region group, sorted by
        group name for deterministic, diffable output.
    """
    groups: dict[str, list[CorpusEntry]] = {}
    for entry in entries:
        groups.setdefault(_group_key(entry), []).append(entry)

    results: list[BenchmarkResult] = []
    for group, group_entries in sorted(groups.items()):
        original_fertilities = []
        optimized_fertilities = []

        for entry in group_entries:
            words = _count_words(entry.text)
            original_tokens = counter(entry.text)
            original_fertilities.append(_fertility(original_tokens, words))

            optimized_text = normalize(entry.text, level=level)
            optimized_words = _count_words(optimized_text)
            optimized_tokens = counter(optimized_text)
            optimized_fertilities.append(_fertility(optimized_tokens, optimized_words))

        avg_original = sum(original_fertilities) / len(original_fertilities)
        avg_optimized = sum(optimized_fertilities) / len(optimized_fertilities)
        percent_saved = (
            ((avg_original - avg_optimized) / avg_original) * 100.0
            if avg_original > 0
            else 0.0
        )

        results.append(
            BenchmarkResult(
                group=group,
                entry_count=len(group_entries),
                avg_original_fertility=avg_original,
                avg_optimized_fertility=avg_optimized,
                avg_percent_saved=percent_saved,
            )
        )

    return results


def render_markdown_table(results: list[BenchmarkResult]) -> str:
    """Render benchmark results as a Markdown table for README/results publishing."""
    header = (
        "| Group | Entries | Avg Fertility (before) | Avg Fertility (after) | Saved % |\n"
        "|---|---|---|---|---|\n"
    )
    rows = "\n".join(
        f"| {r.group} | {r.entry_count} | {r.avg_original_fertility:.2f} | "
        f"{r.avg_optimized_fertility:.2f} | {r.avg_percent_saved:.1f}% |"
        for r in results
    )
    return header + rows