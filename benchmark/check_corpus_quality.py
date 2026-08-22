"""Detect near-duplicate sentences and report length distribution.

Not part of the installable package -- a repo-level QA tool for anyone
(including future contributors via PR) adding sentences to
corpus/seed_corpus.jsonl or corpus/dialect_validation.jsonl, to catch
accidental repetition/templating that isn't obvious from eyeballing a
large file.

Method: within each (category, region) group, computes Jaccard word
overlap between every pair of sentences (word sets, not exact strings,
so near-duplicates with minor rewording are still caught). Any pair
above --threshold is flagged for manual review -- this is a heuristic
screen, not an automatic rejection; some legitimately similar sentences
(e.g. testing a specific grammatical construction) may be fine on
inspection.
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

from ar_tokenwise.benchmark import load_corpus

DEFAULT_SIMILARITY_THRESHOLD = 0.5


def _jaccard_similarity(a: str, b: str) -> float:
    """Word-set Jaccard similarity between two sentences (0.0-1.0)."""
    words_a = set(a.split())
    words_b = set(b.split())
    if not words_a and not words_b:
        return 1.0
    union = words_a | words_b
    if not union:
        return 0.0
    return len(words_a & words_b) / len(union)


def find_near_duplicates(
    entries: list, threshold: float = DEFAULT_SIMILARITY_THRESHOLD
) -> list[tuple[str, str, float]]:
    """Find sentence pairs above the similarity threshold, within each group.

    Grouped by (category, region) since cross-dialect comparisons of
    intentionally-similar content (if ever done deliberately) shouldn't
    be flagged the same way as accidental within-dialect repetition.

    Returns:
        List of (id_a, id_b, similarity) tuples, sorted by similarity
        descending (most suspicious first).
    """
    groups: dict[tuple, list] = {}
    for entry in entries:
        key = (entry.category, entry.region)
        groups.setdefault(key, []).append(entry)

    flagged: list[tuple[str, str, float]] = []
    for group_entries in groups.values():
        for entry_a, entry_b in combinations(group_entries, 2):
            similarity = _jaccard_similarity(entry_a.text, entry_b.text)
            if similarity >= threshold:
                flagged.append((entry_a.id, entry_b.id, similarity))

    return sorted(flagged, key=lambda t: t[2], reverse=True)


def find_cross_dialect_duplicates(
    entries: list, threshold: float = DEFAULT_SIMILARITY_THRESHOLD
) -> list[tuple[str, str, float]]:
    """Find sentence pairs above the threshold ACROSS different dialect regions.

    This catches a specific, easy-to-fall-into mistake: writing the same
    content once and translating it into each dialect, which produces
    superficially "different" sentences (different words) but zero real
    topical diversity across the corpus as a whole. Compares every
    dialect-category entry against every other dialect-category entry
    regardless of region, unlike find_near_duplicates() which only
    compares within the same region.

    Returns:
        List of (id_a, id_b, similarity) tuples, sorted by similarity
        descending.
    """
    dialect_entries = [e for e in entries if e.category.value == "dialect"]

    flagged: list[tuple[str, str, float]] = []
    for entry_a, entry_b in combinations(dialect_entries, 2):
        if entry_a.region == entry_b.region:
            continue  # already covered by find_near_duplicates()
        similarity = _jaccard_similarity(entry_a.text, entry_b.text)
        if similarity >= threshold:
            flagged.append((entry_a.id, entry_b.id, similarity))

    return sorted(flagged, key=lambda t: t[2], reverse=True)


def report_length_distribution(entries: list) -> str:
    """Render a word-count distribution table by (category, region) group."""
    groups: dict[tuple, list[int]] = {}
    for entry in entries:
        key = (entry.category.value, entry.region)
        groups.setdefault(key, []).append(len(entry.text.split()))

    lines = ["| Group | Min | Max | Avg |", "|---|---|---|---|"]
    for (category, region), lengths in sorted(groups.items(), key=lambda kv: str(kv[0])):
        label = f"{category}:{region}" if region else category
        avg = sum(lengths) / len(lengths)
        lines.append(f"| {label} | {min(lengths)} | {max(lengths)} | {avg:.1f} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check corpus quality.")
    parser.add_argument(
        "corpus_path",
        nargs="?",
        default=str(Path(__file__).parent / "corpus" / "seed_corpus.jsonl"),
        help="Path to a corpus JSONL file (default: corpus/seed_corpus.jsonl).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help=f"Jaccard similarity threshold to flag (default: {DEFAULT_SIMILARITY_THRESHOLD}).",
    )
    args = parser.parse_args()

    entries = load_corpus(args.corpus_path)
    print(f"Loaded {len(entries)} entries from {args.corpus_path}\n")

    print("## Length distribution (words)\n")
    print(report_length_distribution(entries))
    print()

    duplicates = find_near_duplicates(entries, threshold=args.threshold)
    print(f"\n## Within-group near-duplicate check (threshold={args.threshold})\n")
    if not duplicates:
        print("No near-duplicate pairs found.")
    else:
        print(f"{len(duplicates)} suspicious pair(s) found:\n")
        for id_a, id_b, similarity in duplicates:
            print(f"  {similarity:.2f}  {id_a}  <->  {id_b}")

    cross_dialect_duplicates = find_cross_dialect_duplicates(entries, threshold=args.threshold)
    print(f"\n## Cross-dialect content-reuse check (threshold={args.threshold})\n")
    print(
        "Catches the same content translated across dialects instead of "
        "genuinely distinct topics -- a real issue this project hit once "
        "before, worth checking every time.\n"
    )
    if not cross_dialect_duplicates:
        print("No cross-dialect content reuse found.")
        return 0 if not duplicates else 1

    print(f"{len(cross_dialect_duplicates)} suspicious pair(s) found:\n")
    for id_a, id_b, similarity in cross_dialect_duplicates:
        print(f"  {similarity:.2f}  {id_a}  <->  {id_b}")
    return 1


if __name__ == "__main__":
    sys.exit(main())