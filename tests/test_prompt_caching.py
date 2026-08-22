"""Unit tests for ar_tokenwise.prompt_caching."""

import pytest

from ar_tokenwise.prompt_caching import (
    PromptSegment,
    optimize_for_caching,
    to_anthropic_cache_blocks,
)


# --- required minimum cases: empty / oversized / wrong type -----------


def test_optimize_empty_list_returns_empty_result() -> None:
    result = optimize_for_caching([])
    assert result.segments == []
    assert result.text == ""
    assert result.stable_prefix_length == 0


def test_optimize_oversized_segment_raises_value_error() -> None:
    segments = [PromptSegment(content="x" * 10, stable=True)]
    with pytest.raises(ValueError, match="exceeds max_length"):
        optimize_for_caching(segments, max_length=5)


def test_optimize_non_list_input_raises_type_error() -> None:
    with pytest.raises(TypeError):
        optimize_for_caching("not a list")  # type: ignore[arg-type]


def test_optimize_non_segment_element_raises_type_error() -> None:
    with pytest.raises(TypeError):
        optimize_for_caching(["not a segment"])  # type: ignore[list-item]


# --- reordering behavior -------------------------------------------------


def test_optimize_moves_stable_segments_before_dynamic() -> None:
    segments = [
        PromptSegment(content="dynamic 1", stable=False),
        PromptSegment(content="stable 1", stable=True),
        PromptSegment(content="dynamic 2", stable=False),
        PromptSegment(content="stable 2", stable=True),
    ]
    result = optimize_for_caching(segments)

    assert [s.content for s in result.segments] == [
        "stable 1", "stable 2", "dynamic 1", "dynamic 2",
    ]


def test_optimize_preserves_relative_order_within_each_group() -> None:
    # Stable sort guarantee: original order within stable/dynamic groups
    # must survive even when interleaved in the input.
    segments = [
        PromptSegment(content="s_a", stable=True),
        PromptSegment(content="d_a", stable=False),
        PromptSegment(content="s_b", stable=True),
        PromptSegment(content="d_b", stable=False),
        PromptSegment(content="s_c", stable=True),
    ]
    result = optimize_for_caching(segments)

    stable_order = [s.content for s in result.segments if s.stable]
    dynamic_order = [s.content for s in result.segments if not s.stable]
    assert stable_order == ["s_a", "s_b", "s_c"]
    assert dynamic_order == ["d_a", "d_b"]


def test_optimize_all_stable_segments() -> None:
    segments = [PromptSegment(content="a", stable=True), PromptSegment(content="b", stable=True)]
    result = optimize_for_caching(segments)
    assert result.stable_prefix_length == len(result.text)


def test_optimize_all_dynamic_segments() -> None:
    segments = [PromptSegment(content="a", stable=False), PromptSegment(content="b", stable=False)]
    result = optimize_for_caching(segments)
    assert result.stable_prefix_length == 0


def test_optimize_stable_prefix_length_matches_text_slice() -> None:
    # The documented guarantee: text[:stable_prefix_length] == stable
    # segments joined alone.
    segments = [
        PromptSegment(content="system prompt", stable=True),
        PromptSegment(content="few-shot examples", stable=True),
        PromptSegment(content="user question", stable=False),
    ]
    result = optimize_for_caching(segments)

    expected_stable_text = "system prompt\n\nfew-shot examples"
    assert result.text[: result.stable_prefix_length] == expected_stable_text
    assert result.stable_prefix_length == len(expected_stable_text)


def test_optimize_respects_custom_separator() -> None:
    segments = [
        PromptSegment(content="a", stable=True),
        PromptSegment(content="b", stable=False),
    ]
    result = optimize_for_caching(segments, separator=" | ")
    assert result.text == "a | b"


# --- to_anthropic_cache_blocks() ------------------------------------------


def test_anthropic_blocks_marks_last_stable_block() -> None:
    segments = [
        PromptSegment(content="system prompt", stable=True),
        PromptSegment(content="examples", stable=True),
        PromptSegment(content="user question", stable=False),
    ]
    optimized = optimize_for_caching(segments)
    blocks = to_anthropic_cache_blocks(optimized)

    assert blocks[0] == {"type": "text", "text": "system prompt"}
    assert blocks[1] == {
        "type": "text",
        "text": "examples",
        "cache_control": {"type": "ephemeral"},
    }
    assert blocks[2] == {"type": "text", "text": "user question"}


def test_anthropic_blocks_default_ttl_omits_ttl_key() -> None:
    optimized = optimize_for_caching([PromptSegment(content="a", stable=True)])
    blocks = to_anthropic_cache_blocks(optimized)
    assert "ttl" not in blocks[0]["cache_control"]


def test_anthropic_blocks_1h_ttl_includes_ttl_key() -> None:
    optimized = optimize_for_caching([PromptSegment(content="a", stable=True)])
    blocks = to_anthropic_cache_blocks(optimized, ttl="1h")
    assert blocks[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_anthropic_blocks_no_stable_segments_no_marker() -> None:
    optimized = optimize_for_caching([PromptSegment(content="a", stable=False)])
    blocks = to_anthropic_cache_blocks(optimized)
    assert "cache_control" not in blocks[0]


def test_anthropic_blocks_invalid_ttl_raises_value_error() -> None:
    optimized = optimize_for_caching([PromptSegment(content="a", stable=True)])
    with pytest.raises(ValueError, match="ttl must be"):
        to_anthropic_cache_blocks(optimized, ttl="1d")


def test_anthropic_blocks_empty_segments_returns_empty_list() -> None:
    optimized = optimize_for_caching([])
    assert to_anthropic_cache_blocks(optimized) == []