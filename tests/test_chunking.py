"""Unit tests for ar_tokenwise.chunking."""

import pytest

from ar_tokenwise.chunking import chunk_text


def _word_count_counter(text: str) -> int:
    """Deterministic fake counter: 1 'token' per whitespace-separated word."""
    return len(text.split())


# --- required minimum cases: empty / oversized / wrong type -----------


def test_chunk_text_empty_returns_empty_list() -> None:
    assert chunk_text("", counter=_word_count_counter) == []


def test_chunk_text_oversized_input_raises_value_error() -> None:
    with pytest.raises(ValueError, match="exceeds max_length"):
        chunk_text("كلمة " * 10, counter=_word_count_counter, max_length=5)


def test_chunk_text_non_str_input_raises_type_error() -> None:
    with pytest.raises(TypeError):
        chunk_text(123, counter=_word_count_counter)  # type: ignore[arg-type]


# --- min/max validation -------------------------------------------------


def test_chunk_text_min_greater_than_max_raises_value_error() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        chunk_text("نص عادي هنا.", counter=_word_count_counter, min_tokens=10, max_tokens=5)


def test_chunk_text_non_positive_bounds_raise_value_error() -> None:
    with pytest.raises(ValueError, match="positive"):
        chunk_text("نص عادي هنا.", counter=_word_count_counter, min_tokens=0, max_tokens=5)


# --- sentence-boundary chunking behavior --------------------------------


def test_chunk_text_single_short_sentence_is_one_chunk() -> None:
    result = chunk_text(
        "هذا نص قصير جدا.", counter=_word_count_counter, min_tokens=1, max_tokens=10
    )
    assert result == ["هذا نص قصير جدا."]


def test_chunk_text_respects_max_tokens_by_splitting_sentences() -> None:
    # 3 sentences of 3 words each; max_tokens=3 forces one sentence per chunk.
    text = "كلمة كلمة كلمة. كلمة كلمة كلمة. كلمة كلمة كلمة."
    result = chunk_text(text, counter=_word_count_counter, min_tokens=1, max_tokens=3)
    assert len(result) == 3
    for chunk in result:
        assert _word_count_counter(chunk) <= 3


def test_chunk_text_merges_undersized_trailing_chunk() -> None:
    # Two sentences of 2 words each; min_tokens=5 forces a merge attempt.
    text = "كلمة كلمة. كلمة كلمة."
    result = chunk_text(text, counter=_word_count_counter, min_tokens=5, max_tokens=10)
    # Both sentences fit together under max_tokens (4 <= 10), so they merge
    # into a single chunk rather than leaving an undersized trailing one.
    assert len(result) == 1


def test_chunk_text_does_not_merge_when_merge_would_exceed_max() -> None:
    # First sentence (3 tokens) fills the max_tokens=3 budget on its own,
    # so the 2-token trailing sentence starts a new chunk. That trailing
    # chunk (2 tokens) is under min_tokens=3, so a merge is attempted --
    # but 3 + 2 = 5 exceeds max_tokens=3, so it must NOT merge.
    text = "كلمة كلمة كلمة. كلمة كلمة."
    result = chunk_text(text, counter=_word_count_counter, min_tokens=3, max_tokens=3)
    assert len(result) == 2


def test_chunk_text_splits_oversized_single_sentence_by_words() -> None:
    # One long sentence (no punctuation breaks) exceeding max_tokens=3.
    text = "كلمة كلمة كلمة كلمة كلمة كلمة"
    result = chunk_text(text, counter=_word_count_counter, min_tokens=1, max_tokens=3)
    assert len(result) == 2
    for chunk in result:
        assert _word_count_counter(chunk) <= 3


def test_chunk_text_preserves_all_words() -> None:
    text = "الجملة الاولى هنا. الجملة الثانية هنا ايضا. والثالثة اخيرا."
    result = chunk_text(text, counter=_word_count_counter, min_tokens=1, max_tokens=4)
    rejoined_word_count = sum(_word_count_counter(chunk) for chunk in result)
    assert rejoined_word_count == _word_count_counter(text)