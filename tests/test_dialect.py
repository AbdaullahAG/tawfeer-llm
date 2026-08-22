"""Unit tests for ar_tokenwise.dialect."""

import pytest

from ar_tokenwise.dialect import DetectionStatus, DialectCategory, detect_dialect


# --- required minimum cases: empty / oversized / wrong type -----------


def test_detect_empty_text_is_insufficient() -> None:
    result = detect_dialect("")
    assert result.status is DetectionStatus.INSUFFICIENT_TEXT
    assert result.word_count == 0
    assert result.distribution is None


def test_detect_oversized_input_raises_value_error() -> None:
    with pytest.raises(ValueError, match="exceeds max_length"):
        detect_dialect("كلمة " * 10, max_length=5)


def test_detect_non_str_input_raises_type_error() -> None:
    with pytest.raises(TypeError):
        detect_dialect(123)  # type: ignore[arg-type]


# --- insufficient_text / no_signal edge cases ---------------------------


def test_detect_below_min_words_is_insufficient() -> None:
    result = detect_dialect("مرحبا بكم", min_words=5)
    assert result.status is DetectionStatus.INSUFFICIENT_TEXT
    assert result.word_count == 2


def test_detect_at_exactly_min_words_is_not_insufficient() -> None:
    # 5 words, no markers -> should reach NO_SIGNAL, not INSUFFICIENT_TEXT.
    result = detect_dialect("كتاب جميل على الطاولة اليوم", min_words=5)
    assert result.status is not DetectionStatus.INSUFFICIENT_TEXT


def test_detect_enough_words_no_markers_is_no_signal() -> None:
    result = detect_dialect("كتاب جميل على الطاولة اليوم هنا", min_words=5)
    assert result.status is DetectionStatus.NO_SIGNAL
    assert result.distribution is None


# --- distribution behavior -----------------------------------------------


def test_detect_gulf_markers_score_highest_for_gulf() -> None:
    text = "شلونك اليوم؟ وش سويت من الصبح الحين؟"
    result = detect_dialect(text)
    assert result.status is DetectionStatus.DISTRIBUTION
    top = max(result.distribution, key=result.distribution.get)
    assert top is DialectCategory.GULF


def test_detect_egyptian_markers_score_highest_for_egyptian() -> None:
    text = "إزيك يا صاحبي؟ عامل ايه في الشغل الجديد ده؟"
    result = detect_dialect(text)
    assert result.status is DetectionStatus.DISTRIBUTION
    top = max(result.distribution, key=result.distribution.get)
    assert top is DialectCategory.EGYPTIAN


def test_detect_msa_markers_score_highest_for_msa() -> None:
    text = "أعلنت الوزارة أن المشروع سيبدأ تنفيذه وفقا للخطة الموضوعة."
    result = detect_dialect(text)
    assert result.status is DetectionStatus.DISTRIBUTION
    top = max(result.distribution, key=result.distribution.get)
    assert top is DialectCategory.MSA


def test_detect_distribution_sums_to_one() -> None:
    text = "شلونك اليوم؟ وش سويت من الصبح الحين؟"
    result = detect_dialect(text)
    assert result.distribution is not None
    assert sum(result.distribution.values()) == pytest.approx(1.0)


def test_detect_distribution_covers_all_categories() -> None:
    text = "شلونك اليوم؟ وش سويت من الصبح الحين؟"
    result = detect_dialect(text)
    assert set(result.distribution.keys()) == set(DialectCategory)


def test_detect_diacritics_do_not_prevent_marker_match() -> None:
    # Same Gulf markers, but with diacritics -- LIGHT normalization inside
    # detect_dialect should still catch them via substring match.
    text = "شَلونَك اليَوم؟ وِش سَوّيت مِن الصُبح الحين؟"
    result = detect_dialect(text)
    assert result.status is DetectionStatus.DISTRIBUTION


def test_detect_mixed_dialect_signals_split_distribution() -> None:
    # Contains both a clear Gulf marker and a clear Egyptian marker.
    text = "شلونك؟ عايز اروح السوق بكرة ان شاء الله معك"
    result = detect_dialect(text)
    assert result.status is DetectionStatus.DISTRIBUTION
    assert result.distribution[DialectCategory.GULF] > 0
    assert result.distribution[DialectCategory.EGYPTIAN] > 0


def test_detect_plain_msa_with_word_containing_marker_substring_is_not_misdetected() -> None:
    # Regression test for a real, reproduced bug: the Egyptian marker
    # "مش" previously matched via plain substring inside "لمشوار"
    # ("for an errand"), causing a neutral MSA sentence with zero real
    # dialect markers to be reported as a confident Egyptian/Levantine
    # distribution. Word-boundary matching must prevent this.
    text = "ذهب الأولاد في نزهة جميلة إلى الحديقة العامة صباح اليوم لمشوار طويل جدا"
    result = detect_dialect(text)
    assert result.status is DetectionStatus.NO_SIGNAL
