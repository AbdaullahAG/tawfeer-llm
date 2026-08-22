"""Heuristic content-sensitivity warnings for Arabic text (stateless, opt-in).

This module is a PURE FUNCTION, deliberately -- it holds no session state,
counters, or memory of prior calls. Each call to check_content_warnings()
is independent and returns only information: it never blocks, modifies,
or refuses to process text.

Why stateless: rate-limiting how often the same warning is shown to a
user requires session state (what was shown before, when). That decision
belongs to the caller (the agent or application using this library), not
to a normalization library -- see SKILL.md for the recommended behavioral
guidance ("don't repeat the same warning category more than once per
session"), which is documented as guidance for the caller, not enforced
in code here.

Why confidence levels, not a single boolean: a single incidental match of
a common word is a weak signal and, if treated as equally alarming as a
strong match, trains users to ignore all warnings (documented as "alert
fatigue" in clinical decision support literature, with override rates of
49-96% for undifferentiated alerts). Marker lists here also deliberately
exclude everyday religious filler phrases (e.g. "ان شاء الله",
"الحمدلله") which are common in ordinary conversational Arabic and are
not, by themselves, an indicator of Quranic/Hadith-sensitive content.

This is a heuristic based on small, hand-authored marker lists -- not a
trained classifier. False negatives (missing an actually sensitive text)
and false positives (flagging ordinary text) are both possible. Treat
results as a hint to review, never as a guarantee of safety.

Marker matching is WORD-BOUNDARY-AWARE (via ar_tokenwise._internal), not
plain substring matching. Plain substring matching previously caused a
real, reproduced false-positive bug: the LEGAL marker "بند" matched
inside "بندر" (a personal name / place name), flagging entirely ordinary
text mentioning someone named Bandar as legally sensitive.

KNOWN LIMITATION of word-boundary matching for Arabic: single-letter
prefixes commonly fused directly onto the next word without a space
(و "and", ب "by/with", ل "for", ك "like", ف "so" -- e.g. "بالعقد" =
"ب" + "العقد" with no space) will NOT match a marker for the base word,
since \b does not recognize a boundary between two Arabic letters even
when one is a grammatical prefix. This trades a small number of missed
matches (false negatives) for eliminating the substring false-positive
bug above -- a deliberate choice, since this module already documents
that it can miss real sensitive content (false negatives are an
accepted, disclosed limitation) but should not cry wolf on ordinary text
(false positives directly cause the alert-fatigue problem discussed
above). Proper handling would need real Arabic morphological analysis,
which is out of scope for this lightweight, dependency-free module.
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


class ContentCategory(str, Enum):
    """Content-sensitivity categories this module can flag."""

    RELIGIOUS = "religious"
    LEGAL = "legal"
    MEDICAL = "medical"


class ConfidenceLevel(str, Enum):
    """How many distinct markers matched -- not a calibrated probability."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class ContentWarning:
    """A single category flagged for a piece of text, with its confidence."""

    category: ContentCategory
    confidence: ConfidenceLevel
    matched_marker_count: int


_RELIGIOUS_MARKERS = [
    "قال تعالى", "صلى الله عليه وسلم", "روى عن", "رواه", "سورة", "آية",
    "حديث شريف", "عليه السلام", "رضي الله عنه", "أخرجه", "باب من أبواب",
    "فقه", "الفقهاء", "أهل العلم", "إسناده",
]

_LEGAL_MARKERS = [
    "الطرف الأول", "الطرف الثاني", "بموجب هذا العقد", "المادة رقم",
    "يلتزم الطرف", "التحكيم", "الفسخ", "بند", "الاتفاقية", "العقد المبرم",
    "الإخلال بأحكام", "سارية المفعول", "إشعار خطي", "النزاعات الناشئة",
    "التوقيع عليها",
]

_MEDICAL_MARKERS = [
    "الجرعة", "التشخيص", "الأعراض", "العلاج", "المريض", "الوصفة الطبية",
    "مضاد حيوي", "الآثار الجانبية", "التحاليل", "الفحص السريري",
    "الحالة الصحية", "الطبيب المعالج", "الدواء", "المستشفى",
]

_MARKERS_BY_CATEGORY: dict[ContentCategory, list[str]] = {
    ContentCategory.RELIGIOUS: _RELIGIOUS_MARKERS,
    ContentCategory.LEGAL: _LEGAL_MARKERS,
    ContentCategory.MEDICAL: _MEDICAL_MARKERS,
}

# Compiled once at import time, not per-call -- see _internal.py.
_COMPILED_MARKERS_BY_CATEGORY = compile_markers_by_category(_MARKERS_BY_CATEGORY)


def _confidence_for_count(count: int) -> ConfidenceLevel:
    """Map a matched-marker count to a confidence level."""
    if count >= 3:
        return ConfidenceLevel.HIGH
    if count == 2:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


def check_content_warnings(
    text: str,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> list[ContentWarning]:
    """Check text for markers of religious/legal/medical sensitivity.

    This is advisory only: it never blocks, modifies, or refuses text. It
    is a heuristic estimate, not a guarantee -- see the module docstring
    for its limitations. Each call is independent; this function holds no
    state across calls (see module docstring for why).

    Args:
        text: Input text. Empty string returns an empty list (no warnings).
        max_length: Maximum accepted input length in characters, used as
            a size-based safety guard. Raises if exceeded.

    Returns:
        A list of :class:`ContentWarning`, one per category with at least
        one marker match. Empty list means no category was flagged --
        this does NOT guarantee the text is safe to process without care.

    Raises:
        TypeError: if ``text`` is not a string.
        ValueError: if ``text`` exceeds ``max_length``.
    """
    validate_text_input(text, max_length, caller_name="check_content_warnings")

    if text == "":
        return []

    matching_text = normalize(text, level=NormalizationLevel.LIGHT)

    counts = count_marker_matches(matching_text, _COMPILED_MARKERS_BY_CATEGORY)

    warnings: list[ContentWarning] = []
    for category, count in counts.items():
        if count > 0:
            warnings.append(
                ContentWarning(
                    category=category,
                    confidence=_confidence_for_count(count),
                    matched_marker_count=count,
                )
            )

    return warnings
