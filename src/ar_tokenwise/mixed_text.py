"""Separate Arabic/non-Arabic/Arabizi fertility reporting for code-switched text.

Real-world Arabic text often mixes Arabic words with Latin script (product
names, technical terms) and digits (either script). It also very commonly
uses "Arabizi" -- Arabic written phonetically in Latin letters, with
digits standing in for Arabic sounds that have no Latin equivalent
(2=hamza, 3=ain, 5=kha, 6=ta, 7=ha, 8=ghain/qaf, 9=sad -- e.g. "7abibi"
for "habibi", "3andi" for "عندي"). This is extremely common in everyday
Arabic chat, SMS, and social media, and was previously misclassified as
plain non-Arabic text since it uses no Arabic-script characters at all.

Word classification ignores plain digit runs (Arabic-indic or Latin, 2+
digits together, e.g. product model numbers or years) entirely, so a word
like "iPhone15" or "COVID19" is not misclassified as Arabizi just because
it ends in digits -- only an ISOLATED single digit from the Arabizi set,
adjacent to a Latin letter, counts as an Arabizi signal. This is a
heuristic, not a guarantee: rare words like "3M" or "G7" can be
misclassified as Arabizi. The trade-off is accepted because Arabizi is a
widespread, well-documented phenomenon and the false-positive rate on
ordinary alphanumeric product/company names is low in practice.

This is descriptive/measurement only: it does not modify or split text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from ar_tokenwise._internal import compute_fertility
from ar_tokenwise.normalize import DEFAULT_MAX_LENGTH
from ar_tokenwise.report import TokenCounter

_ARABIC_LETTER_PATTERN = re.compile(r"[\u0621-\u063A\u0641-\u064A\u066E\u066F\u0671-\u06D3]")
_LATIN_LETTER_PATTERN = re.compile(r"[A-Za-z]")
_WHITESPACE_SPLIT_PATTERN = re.compile(r"\s+")

_ARABIZI_DIGITS = frozenset("2356789")


class WordCategory(str, Enum):
    """Script classification for a single whitespace-delimited word.

    ARABIC: contains Arabic letters, no Latin letters.
    NON_ARABIC: contains no Arabic letters and no Arabizi digit signal.
    MIXED: contains both Arabic and Latin letters in the same word.
    ARABIZI: Latin letters with an isolated Arabizi-convention digit
        adjacent to a letter.
    """

    ARABIC = "arabic"
    NON_ARABIC = "non_arabic"
    MIXED = "mixed"
    ARABIZI = "arabizi"


@dataclass(frozen=True)
class MixedTextReport:
    """Per-script-category word counts and fertility for a text."""

    total_words: int
    arabic_word_count: int
    non_arabic_word_count: int
    mixed_word_count: int
    arabizi_word_count: int
    arabic_fertility: float
    non_arabic_fertility: float
    mixed_fertility: float
    arabizi_fertility: float


def _has_arabizi_digit_signal(word: str) -> bool:
    """Detect an isolated single Arabizi-convention digit next to a letter."""
    i = 0
    n = len(word)
    while i < n:
        if word[i].isdigit():
            run_start = i
            while i < n and word[i].isdigit():
                i += 1
            run_length = i - run_start
            if run_length == 1 and word[run_start] in _ARABIZI_DIGITS:
                letter_before = run_start > 0 and word[run_start - 1].isalpha()
                letter_after = i < n and word[i].isalpha()
                if letter_before or letter_after:
                    return True
        else:
            i += 1
    return False


def classify_word(word: str) -> WordCategory:
    """Classify a single word by Arabic/Latin/Arabizi content."""
    has_arabic = bool(_ARABIC_LETTER_PATTERN.search(word))
    has_latin = bool(_LATIN_LETTER_PATTERN.search(word))

    if has_arabic and has_latin:
        return WordCategory.MIXED
    if has_arabic:
        return WordCategory.ARABIC
    if has_latin and _has_arabizi_digit_signal(word):
        return WordCategory.ARABIZI
    return WordCategory.NON_ARABIC


def _validate_input(text: str, max_length: int) -> None:
    """Validate input before any processing."""
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
    """Report Arabic/non-Arabic/Arabizi fertility separately for mixed text."""
    _validate_input(text, max_length)

    if text == "":
        return MixedTextReport(
            total_words=0,
            arabic_word_count=0,
            non_arabic_word_count=0,
            mixed_word_count=0,
            arabizi_word_count=0,
            arabic_fertility=0.0,
            non_arabic_fertility=0.0,
            mixed_fertility=0.0,
            arabizi_fertility=0.0,
        )

    words = [w for w in _WHITESPACE_SPLIT_PATTERN.split(text.strip()) if w]

    buckets: dict[WordCategory, list[str]] = {
        WordCategory.ARABIC: [],
        WordCategory.NON_ARABIC: [],
        WordCategory.MIXED: [],
        WordCategory.ARABIZI: [],
    }
    for word in words:
        buckets[classify_word(word)].append(word)

    def _bucket_fertility(bucket_words: list[str]) -> float:
        if not bucket_words:
            return 0.0
        reconstructed = " ".join(bucket_words)
        return compute_fertility(counter(reconstructed), len(bucket_words))

    return MixedTextReport(
        total_words=len(words),
        arabic_word_count=len(buckets[WordCategory.ARABIC]),
        non_arabic_word_count=len(buckets[WordCategory.NON_ARABIC]),
        mixed_word_count=len(buckets[WordCategory.MIXED]),
        arabizi_word_count=len(buckets[WordCategory.ARABIZI]),
        arabic_fertility=_bucket_fertility(buckets[WordCategory.ARABIC]),
        non_arabic_fertility=_bucket_fertility(buckets[WordCategory.NON_ARABIC]),
        mixed_fertility=_bucket_fertility(buckets[WordCategory.MIXED]),
        arabizi_fertility=_bucket_fertility(buckets[WordCategory.ARABIZI]),
    )
