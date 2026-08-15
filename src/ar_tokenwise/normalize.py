"""Conservative Arabic text normalization for LLM token efficiency.

This module intentionally avoids any transformation that could change the
meaning of the text (e.g. unifying teh marbuta with heh, or stripping hamza
letters themselves). Only orthographic variation and optional pronunciation
marks are touched, and only at the level the caller explicitly opts into.
"""

from __future__ import annotations

import re
import unicodedata
from enum import Enum

# Default ceiling for a single normalize() call, to avoid pathological
# memory/CPU usage on accidental huge inputs (size-based DoS guard).
DEFAULT_MAX_LENGTH = 1_000_000


class NormalizationLevel(str, Enum):
    """How aggressive the normalization pipeline should be.

    LIGHT: typographic changes (tatweel, digit unification) plus removal of
        optional pronunciation marks (tashkeel). Tashkeel is decorative and
        safe to remove in ordinary prose, but is NOT safe for text where
        diacritics carry grammatical/semantic weight -- Quranic verses,
        Hadith, and other texts where i'rab (case marks) is part of the
        meaning. For such text, do not normalize at all rather than relying
        on LIGHT being universally safe.
    MEDIUM: adds common orthographic unification (alef/yeh forms), which is
        standard practice but can rarely matter for a specific word, so it
        is opt-in rather than default.
    """

    LIGHT = "light"
    MEDIUM = "medium"


# --- Precompiled patterns (module-level, built once) ---------------------

# Arabic optional diacritics (harakat/tashkeel): fatha, damma, kasra,
# tanween forms, shadda, sukun, and the superscript alef (dagger alef).
# These mark pronunciation, not meaning, and are omitted in the vast
# majority of everyday written Arabic.
_TASHKEEL_PATTERN = re.compile(r"[\u064B-\u0652\u0670]")

# Tatweel / kashida: a purely decorative elongation character used to
# stretch words for justification or calligraphy. Never carries meaning.
_TATWEEL_PATTERN = re.compile(r"\u0640")

# Arabic-Indic and Extended Arabic-Indic (Persian) digits -> ASCII digits.
# Mapping is 1:1 and unambiguous.
_DIGIT_MAP = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩" "۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)

# Alef variants unified to the bare alef (ا). This is standard search-time
# normalization; kept out of LIGHT because it collapses orthographic
# distinctions some writers use deliberately.
_ALEF_VARIANTS_PATTERN = re.compile(r"[\u0622\u0623\u0625\u0671]")

# Alef maksura (ى) unified to yeh (ي). Common Egyptian/Gulf spelling
# variation; same rationale as alef unification above.
_ALEF_MAKSURA_PATTERN = re.compile(r"\u0649")


def _validate_input(text: str, max_length: int) -> None:
    """Validate input before any processing.

    Raises:
        TypeError: if ``text`` is not a ``str``.
        ValueError: if ``text`` exceeds ``max_length`` characters.
    """
    if not isinstance(text, str):
        raise TypeError(f"normalize() expects str, got {type(text).__name__}")
    if len(text) > max_length:
        raise ValueError(
            f"Input length {len(text)} exceeds max_length={max_length}. "
            "Split the text or raise max_length explicitly if intentional."
        )


def normalize(
    text: str,
    level: NormalizationLevel = NormalizationLevel.LIGHT,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> str:
    """Normalize Arabic text conservatively for LLM token efficiency.

    Args:
        text: Input text. Non-Arabic characters pass through untouched.
        level: How aggressive the normalization should be. See
            :class:`NormalizationLevel`.
        max_length: Maximum accepted input length in characters, used as a
            size-based safety guard. Raises if exceeded.

    Returns:
        The normalized text. Empty input returns empty output.

    Raises:
        TypeError: if ``text`` is not a string.
        ValueError: if ``text`` is longer than ``max_length``.
    """
    _validate_input(text, max_length)

    if text == "":
        return text

    # Unicode canonical normalization first (NFC) so downstream regex
    # patterns match consistently regardless of how the input was encoded.
    result = unicodedata.normalize("NFC", text)

    result = _TATWEEL_PATTERN.sub("", result)
    result = _TASHKEEL_PATTERN.sub("", result)
    result = result.translate(_DIGIT_MAP)

    if level is NormalizationLevel.MEDIUM:
        result = _ALEF_VARIANTS_PATTERN.sub("\u0627", result)  # -> ا
        result = _ALEF_MAKSURA_PATTERN.sub("\u064A", result)  # -> ي

    return result