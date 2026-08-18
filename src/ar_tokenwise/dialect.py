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

Measured accuracy: see benchmark/run_dialect_validation.py and its output
in benchmark/results/dialect_validation.md -- run against a hand-labeled
validation corpus kept separate from the marker lists below, to avoid a
circular/self-validating number.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

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


# --- Marker lists (seed, hand-authored, not derived from any licensed
# dataset) -----------------------------------------------------------------
# These are approximate, non-exhaustive, and grouped as broad regional
# umbrellas (not precise per-country boundaries) -- consistent with the
# documented overlap between closely related dialect groups.

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

_MSA_MARKERS = [
    "إن ", "أن ", "الذي", "التي", "اللذان", "اللتان", "حيث", "بينما",
    "إذ ", "لكن ", "سوف ", "قد ", "ينبغي", "وفقا", "نظرا", "بالإضافة",
    "لذلك", "فإن", "كما أن",
]

_MARKERS_BY_CATEGORY: dict[DialectCategory, list[str]] = {
    DialectCategory.MSA: _MSA_MARKERS,
    DialectCategory.GULF: _GULF_MARKERS,
    DialectCategory.EGYPTIAN: _EGYPTIAN_MARKERS,
    DialectCategory.LEVANTINE: _LEVANTINE_MARKERS,
    DialectCategory.MAGHREBI: _MAGHREBI_MARKERS,
}


def _validate_input(text: str, max_length: int) -> None:
    """Validate input before any processing.

    Raises:
        TypeError: if ``text`` is not a ``str``.
        ValueError: if ``text`` exceeds ``max_length`` characters.
    """
    if not isinstance(text, str):
        raise TypeError(f"detect_dialect() expects str, got {type(text).__name__}")
    if len(text) > max_length:
        raise ValueError(
            f"Input length {len(text)} exceeds max_length={max_length}. "
            "Split the text or raise max_length explicitly if intentional."
        )


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
    _validate_input(text, max_length)

    word_count = len(text.split())
    if word_count < min_words:
        return DialectDetectionResult(
            status=DetectionStatus.INSUFFICIENT_TEXT,
            word_count=word_count,
            distribution=None,
        )

    # Diacritic-insensitive matching via the existing LIGHT normalizer.
    # The result is used only internally for marker matching; the
    # caller's original text is never modified or returned.
    matching_text = normalize(text, level=NormalizationLevel.LIGHT)

    raw_counts: dict[DialectCategory, int] = {
        category: sum(1 for marker in markers if marker in matching_text)
        for category, markers in _MARKERS_BY_CATEGORY.items()
    }
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