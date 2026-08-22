"""Experimental Arabic dialect signal detection (probabilistic, not exact).

IMPORTANT: This is a heuristic based on small, hand-authored marker word
lists -- NOT a trained model. Even state-of-the-art transformer models on
the NADI 2024 shared task (the standard academic benchmark) only reached
50.57% F1 on multi-label dialect identification, and closely related
dialect groups (notably Levantine: Syrian/Jordanian/Palestinian/Lebanese)
have documented overlap even in expert-annotated ground truth. A simple
word-list heuristic should not be expected to exceed that methodological
ceiling.

Design choices driven by that ceiling:
- Returns a PROBABILITY-LIKE DISTRIBUTION across categories, never a single
  confident label -- a text can score for more than one dialect.
- Explicitly reports INSUFFICIENT_TEXT for very short input, rather than
  producing a confident-looking distribution on 2-3 words.
- Explicitly reports NO_SIGNAL when enough words are present but none
  match any marker, rather than silently defaulting to MSA.
- Marker lists are a small, documented "seed" set (see MARKER LISTS below),
  not derived from or validated against any licensed dataset (e.g. NADI's
  Twitter-sourced data, which has restricted redistribution terms).
- Marker matching is WORD-BOUNDARY-AWARE (via ar_tokenwise._internal), not
  plain substring matching. Plain substring matching previously caused a
  real, reproduced false-positive bug: the Egyptian marker "مش" matched
  inside "لمشوار" ("for an errand"), and this module's own MSA marker
  list had a partial, inconsistent workaround (trailing spaces on some
  entries) that was never applied to the other three lists. That
  workaround is gone now that matching is correctly boundary-aware
  everywhere.
- KNOWN LIMITATION: word-boundary matching will miss a marker fused to a
  single-letter Arabic prefix with no space (و/ب/ل/ك/ف -- e.g. "بالحين"
  won't match the Gulf marker "الحين"). This trades some missed matches
  for eliminating the substring false-positive bug above -- see
  safety_modes.py's module docstring for the same trade-off, discussed
  there in more detail.

Measured accuracy: see benchmark/run_dialect_validation.py and its output
in benchmark/results/dialect_validation.md -- run against a hand-labeled
validation corpus kept separate from the marker lists below, to avoid a
circular/self-validating number.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ar_tokenwise._internal import (
    compile_markers_by_category,
    count_marker_matches,
    validate_text_input,
)
from ar_tokenwise.normalize import DEFAULT_MAX_LENGTH, NormalizationLevel, normalize

DEFAULT_MIN_WORDS = 5


class DialectCategory(str, Enum):
    """Categories this module can report a signal for."""

    MSA = "msa"
    GULF = "gulf"
    EGYPTIAN = "egyptian"
    LEVANTINE = "levantine"
    MAGHREBI = "maghrebi"


class DetectionStatus(str, Enum):
    """Outcome kind for a detection call -- check this before reading distribution."""

    INSUFFICIENT_TEXT = "insufficient_text"  # fewer than min_words words
    NO_SIGNAL = "no_signal"  # enough words, but no marker matched at all
    DISTRIBUTION = "distribution"  # a probability-like distribution was computed


@dataclass(frozen=True)
class DialectDetectionResult:
    """Result of a dialect signal detection call.

    ``distribution`` is only populated when status is DISTRIBUTION. It is
    an ESTIMATE derived from marker-match counts, not a calibrated
    probability -- values sum to 1.0 but should not be read as exact
    confidence percentages.
    """

    status: DetectionStatus
    word_count: int
    distribution: dict[DialectCategory, float] | None


_GULF_MARKERS = [
    "شلون", "شلونك", "وش", "وشلون", "أبغى", "ابغى", "أبي", "ابي", "زين",
    "الحين", "هالحين", "وايد", "مب", "شنو", "عاد", "يبيله", "يبغاله",
    "تراه", "شخبارك", "عساك", "خوش", "دشيت", "يا خوي", "قدايش",
    "لو سمحت", "هني", "جان", "ماكو", "شكو", "زينة",
]

_EGYPTIAN_MARKERS = [
    "إزيك", "ازيك", "عايز", "عاوز", "ازاي", "إزاي", "ليه", "دلوقتي",
    "كده", "أهو", "اهو", "يلا بينا", "مش", "فين", "إمتى", "امتى", "بجد",
    "خالص", "أوي", "اوي", "برضه", "لسه", "عشان", "ايه", "إيه", "مفيش",
    "هروح", "هعمل", "عامل ايه", "جدعان",
]

_LEVANTINE_MARKERS = [
    "كيفك", "شو", "هلق", "لهلق", "هيك", "بدي", "بدك", "وين", "هنيك",
    "هلأ", "كتير", "منيح", "يلا", "هاد", "هاي", "هدول", "عم", "رح",
    "ما في", "شغلة", "مبين", "ولك", "شفت", "ليش", "خلص", "بكفي", "عنجد",
]

_MAGHREBI_MARKERS = [
    "واش", "لاباس", "بزاف", "دابا", "غادي", "كيفاش", "خاصني", "بغيت",
    "ديال", "واخا", "صافي", "نتا", "نتي", "حنا", "راه", "زعما", "ماشي",
    "بصح", "دراري", "فلوس", "مزيان", "دراهم", "شحال", "والو", "مزال",
    "بحال",
]

# NOTE: trailing spaces previously used on some entries here ("إن ",
# "أن ", "إذ ", "لكن ", "سوف ", "قد ") as a partial workaround for the
# substring-matching bug have been removed -- word-boundary matching
# (see module docstring) now handles this correctly and consistently
# for every marker list, so the workaround is no longer needed.
_MSA_MARKERS = [
    "إن", "أن", "الذي", "التي", "اللذان", "اللتان", "حيث", "بينما",
    "إذ", "لكن", "سوف", "قد", "ينبغي", "وفقا", "نظرا", "بالإضافة",
    "لذلك", "فإن", "كما أن",
]

_MARKERS_BY_CATEGORY: dict[DialectCategory, list[str]] = {
    DialectCategory.MSA: _MSA_MARKERS,
    DialectCategory.GULF: _GULF_MARKERS,
    DialectCategory.EGYPTIAN: _EGYPTIAN_MARKERS,
    DialectCategory.LEVANTINE: _LEVANTINE_MARKERS,
    DialectCategory.MAGHREBI: _MAGHREBI_MARKERS,
}

# Compiled once at import time, not per-call -- see _internal.py.
_COMPILED_MARKERS_BY_CATEGORY = compile_markers_by_category(_MARKERS_BY_CATEGORY)


def detect_dialect(
    text: str,
    min_words: int = DEFAULT_MIN_WORDS,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> DialectDetectionResult:
    """Detect a rough dialect signal distribution for Arabic text.

    This is a heuristic marker-based estimate, not a trained classifier --
    see the module docstring for its documented accuracy ceiling. Always
    check ``result.status`` before reading ``result.distribution``.

    Args:
        text: Input text. Empty string is treated as insufficient text,
            not an error.
        min_words: Minimum whitespace-delimited word count required to
            attempt detection. Below this, status is INSUFFICIENT_TEXT.
        max_length: Maximum accepted input length in characters, used as
            a size-based safety guard. Raises if exceeded.

    Returns:
        A :class:`DialectDetectionResult`.

    Raises:
        TypeError: if ``text`` is not a string.
        ValueError: if ``text`` exceeds ``max_length``.
    """
    validate_text_input(text, max_length, caller_name="detect_dialect")

    word_count = len(text.split())
    if word_count < min_words:
        return DialectDetectionResult(
            status=DetectionStatus.INSUFFICIENT_TEXT,
            word_count=word_count,
            distribution=None,
        )

    matching_text = normalize(text, level=NormalizationLevel.LIGHT)

    raw_counts = count_marker_matches(matching_text, _COMPILED_MARKERS_BY_CATEGORY)
    total = sum(raw_counts.values())

    if total == 0:
        return DialectDetectionResult(
            status=DetectionStatus.NO_SIGNAL,
            word_count=word_count,
            distribution=None,
        )

    distribution = {
        category: count / total for category, count in raw_counts.items()
    }
    return DialectDetectionResult(
        status=DetectionStatus.DISTRIBUTION,
        word_count=word_count,
        distribution=distribution,
    )
