# tests/test_safety_modes.py
"""Unit tests for ar_tokenwise.safety_modes."""

import pytest

from ar_tokenwise.safety_modes import (
    ConfidenceLevel,
    ContentCategory,
    check_content_warnings,
)


def test_check_empty_text_returns_no_warnings() -> None:
    assert check_content_warnings("") == []


def test_check_oversized_input_raises_value_error() -> None:
    with pytest.raises(ValueError, match="exceeds max_length"):
        check_content_warnings("نص " * 10, max_length=5)


def test_check_non_str_input_raises_type_error() -> None:
    with pytest.raises(TypeError):
        check_content_warnings(123)  # type: ignore[arg-type]


def test_check_ordinary_text_returns_no_warnings() -> None:
    result = check_content_warnings("رحت عالسوق اليوم واشتريت بعض الأغراض")
    assert result == []


def test_check_casual_religious_phrase_does_not_trigger_warning() -> None:
    result = check_content_warnings("ان شاء الله بكرة بشوفك، الحمدلله كله تمام")
    assert result == []


def test_check_single_legal_marker_gives_low_confidence() -> None:
    result = check_content_warnings("هذا بند مهم بالنص المرفق.")
    legal = [w for w in result if w.category is ContentCategory.LEGAL]
    assert len(legal) == 1
    assert legal[0].confidence is ConfidenceLevel.LOW
    assert legal[0].matched_marker_count == 1


def test_check_two_legal_markers_gives_medium_confidence() -> None:
    text = "هذا بند مهم بالعقد المبرم بيننا."
    result = check_content_warnings(text)
    legal = [w for w in result if w.category is ContentCategory.LEGAL]
    assert len(legal) == 1
    assert legal[0].confidence is ConfidenceLevel.MEDIUM
    assert legal[0].matched_marker_count == 2


def test_check_three_plus_legal_markers_gives_high_confidence() -> None:
    text = (
        "بموجب هذا العقد يلتزم الطرف الأول بتسليم المستندات، "
        "ويحق للطرف الثاني طلب الفسخ وفقا للاتفاقية المبرمة بينهما، "
        "وتخضع النزاعات الناشئة عن هذا العقد للتحكيم."
    )
    result = check_content_warnings(text)
    legal = [w for w in result if w.category is ContentCategory.LEGAL]
    assert len(legal) == 1
    assert legal[0].confidence is ConfidenceLevel.HIGH
    assert legal[0].matched_marker_count >= 3


def test_check_religious_quotation_markers_trigger_warning() -> None:
    text = "روى عن النبي صلى الله عليه وسلم حديث شريف في هذا الباب."
    result = check_content_warnings(text)
    religious = [w for w in result if w.category is ContentCategory.RELIGIOUS]
    assert len(religious) == 1


def test_check_medical_markers_trigger_warning() -> None:
    text = "يجب مراجعة الجرعة الموصوفة مع الطبيب المعالج بسبب الأعراض الجانبية."
    result = check_content_warnings(text)
    medical = [w for w in result if w.category is ContentCategory.MEDICAL]
    assert len(medical) == 1


def test_check_multiple_categories_can_be_flagged_together() -> None:
    text = (
        "بموجب هذا العقد يلتزم الطبيب المعالج بمراجعة الجرعة الموصوفة "
        "للمريض قبل الفسخ."
    )
    result = check_content_warnings(text)
    categories = {w.category for w in result}
    assert ContentCategory.LEGAL in categories
    assert ContentCategory.MEDICAL in categories


def test_check_diacritics_do_not_prevent_marker_match() -> None:
    text = "هَذا بَنْدٌ مُهِمٌّ بِالنَّصِّ المُرْفَق."
    result = check_content_warnings(text)
    legal = [w for w in result if w.category is ContentCategory.LEGAL]
    assert len(legal) == 1