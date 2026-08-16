"""Separate Arabic/non-Arabic fertility reporting for code-switched text.

Real-world Arabic text often mixes Arabic words with Latin script (product
names, technical terms) and digits (either script). Reporting a single
blended fertility number for such text can be misleading -- this module
splits text into words, classifies each by script content, and reports
token-per-word fertility separately for the Arabic and non-Arabic portions.

Word classification ignores digits (Arabic-indic and Latin) entirely, so a
word like "iPhone15" or "COVID-19" is not misclassified as mixed just
because it contains digits -- only actual Arabic vs Latin *letters* in the
same word count as "mixed".

This is descriptive/measurement only: it does not modify or split text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from ar_tokenwise.normalize import DEFAULT_MAX_LENGTH
from ar_tokenwise.report import TokenCounter, _fertility

# Arabic letters (main block), excluding tatweel, diacritics, and
# Arabic-indic digits -- see chat explanation for why this pattern is not
# ReDoS-prone: a single character class, no nested/repeated groups.
_ARABIC_LETTER_PATTERN = re.compile(r"[\u0621-\u063A\u0641-\u064A\u066E\u066F\u0671-\u06D3]")

# Latin letters only -- digits and punctuation deliberately excluded so
# they don't trigger a false "mixed" classification on their own.
_LATIN_LETTER_PATTERN = re.compile(r"[A-Za-z]")

# Whitespace splitter. Duplicated from report.py/chunking.py's private
# pattern rather than imported, to keep this module self-contained and
# not depend on another module's private implementation detail.
_WHITESPACE_SPLIT_PATTERN = re.compile(r"\s+")


class WordCategory(str, Enum):
    """Script classification for a single whitespace-delimited word.

    ARABIC: contains Arabic letters, no Latin letters.
    NON_ARABIC: contains no Arabic letters (Latin text, digits-only,
        punctuation-only all fall here).
    MIXED: contains both Arabic and Latin letters in the same word
        (e.g. an Arabic prefix glued to a Latin term with no space).
    """

    ARABIC = "arabic"
    NON_ARABIC = "non_arabic"
    MIXED = "mixed"


@dataclass(frozen=True)
class MixedTextReport:
    """Per-script-category word counts and fertility for a text.

    Fertility values are ESTIMATES derived from the caller-supplied
    counter re-tokenizing each category's reconstructed substring in
    isolation -- not an exact per-word token attribution from the
    original combined tokenization. Treat them as representative, not
    an exact accounting that sums to the whole text's true token count.
    """

    total_words: int
    arabic_word_count: int
    non_arabic_word_count: int
    mixed_word_count: int
    arabic_fertility: float
    non_arabic_fertility: float
    mixed_fertility: float


def classify_word(word: str) -> WordCategory:
    """Classify a single word by Arabic/Latin letter content.

    Digits (any script) and punctuation are ignored for classification;
    only the presence of actual Arabic vs. Latin letters matters.

    Args:
        word: A single word (no internal whitespace expected, but not
            enforced -- classification still works on any string).

    Returns:
        The word's :class:`WordCategory`.
    """
    has_arabic = bool(_ARABIC_LETTER_PATTERN.search(word))
    has_latin = bool(_LATIN_LETTER_PATTERN.search(word))

    if has_arabic and has_latin:
        return WordCategory.MIXED
    if has_arabic:
        return WordCategory.ARABIC
    return WordCategory.NON_ARABIC


def _validate_input(text: str, max_length: int) -> None:
    """Validate input before any processing.

    Raises:
        TypeError: if ``text`` is not a ``str``.
        ValueError: if ``text`` exceeds ``max_length`` characters.
    """
    if not isinstance(text, str):
        raise TypeError(f"report_mixed_fertility() expects str, got {type(text).__name__}")
    if len(text) > max_length:
        raise ValueError(
            f"Input length {len(text)} exceeds max_length={max_length}. "
            "Split the text or raise max_length explicitly if intentional."
        )


def report_mixed_fertility(
    text: str,
    counter: TokenCounter,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> MixedTextReport:
    """Report Arabic/non-Arabic fertility separately for code-switched text.

    Args:
        text: Input text. Empty string returns a report of all zeros.
        counter: Token-counting function (see ``ar_tokenwise.report``).
        max_length: Maximum accepted input length in characters, used as
            a size-based safety guard. Raises if exceeded.

    Returns:
        A :class:`MixedTextReport`. Fertility for a category with zero
        words in it is reported as 0.0 (not an error, not NaN).

    Raises:
        TypeError: if ``text`` is not a string.
        ValueError: if ``text`` exceeds ``max_length``.
    """
    _validate_input(text, max_length)

    if text == "":
        return MixedTextReport(
            total_words=0,
            arabic_word_count=0,
            non_arabic_word_count=0,
            mixed_word_count=0,
            arabic_fertility=0.0,
            non_arabic_fertility=0.0,
            mixed_fertility=0.0,
        )

    words = [w for w in _WHITESPACE_SPLIT_PATTERN.split(text.strip()) if w]

    buckets: dict[WordCategory, list[str]] = {
        WordCategory.ARABIC: [],
        WordCategory.NON_ARABIC: [],
        WordCategory.MIXED: [],
    }
    for word in words:
        buckets[classify_word(word)].append(word)

    def _bucket_fertility(bucket_words: list[str]) -> float:
        if not bucket_words:
            return 0.0
        reconstructed = " ".join(bucket_words)
        return _fertility(counter(reconstructed), len(bucket_words))

    return MixedTextReport(
        total_words=len(words),
        arabic_word_count=len(buckets[WordCategory.ARABIC]),
        non_arabic_word_count=len(buckets[WordCategory.NON_ARABIC]),
        mixed_word_count=len(buckets[WordCategory.MIXED]),
        arabic_fertility=_bucket_fertility(buckets[WordCategory.ARABIC]),
        non_arabic_fertility=_bucket_fertility(buckets[WordCategory.NON_ARABIC]),
        mixed_fertility=_bucket_fertility(buckets[WordCategory.MIXED]),
    )