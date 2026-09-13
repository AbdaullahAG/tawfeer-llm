"""Tests for reference-backed Quran preservation."""

from ar_tokenwise import NormalizationLevel, QuranGuard, normalize


def _normalizer(text: str) -> str:
    return normalize(text, level=NormalizationLevel.LIGHT)


def test_preserves_a_confirmed_quranic_span_verbatim() -> None:
    text = "مقدمة بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَٰلَمِينَ خاتمة"
    result = normalize(text, preserve_quran=True)

    quote = "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَٰلَمِينَ"
    assert quote in result
    assert result.startswith("مقدمة ")
    assert result.endswith(" خاتمة")


def test_short_phrase_is_not_a_standalone_match() -> None:
    guard = QuranGuard()
    assert guard.find_quran_spans("إن الله غفور") == []


def test_short_tail_is_preserved_only_as_an_extension_of_long_match() -> None:
    text = "قُلْ هُوَ ٱللَّهُ أَحَدٌ ٱللَّهُ ٱلصَّمَدُ لَمْ يَلِدْ"
    spans = QuranGuard().find_quran_spans(text)

    assert len(spans) == 1
    assert text[spans[0].start : spans[0].end] == text


def test_punctuation_does_not_break_a_reference_match() -> None:
    text = "قُلْ هُوَ ٱللَّهُ أَحَدٌ، ٱللَّهُ ٱلصَّمَدُ."
    spans = QuranGuard().find_quran_spans(text)

    assert len(spans) == 1
    assert text[spans[0].start : spans[0].end].startswith("قُلْ")
    assert text[spans[0].start : spans[0].end].endswith("ٱلصَّمَدُ")
