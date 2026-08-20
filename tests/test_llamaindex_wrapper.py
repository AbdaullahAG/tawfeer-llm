"""Unit tests for integrations/llamaindex_wrapper.py.

This module lives outside the installed package (integrations/, not
src/), so it's loaded dynamically via importlib -- same pattern as
tests/test_langchain_wrapper.py for langchain_wrapper.py.
"""

import importlib.util
from pathlib import Path

import pytest

INTEGRATIONS_DIR = Path(__file__).parent.parent / "integrations"


def _load_module(filename: str):
    """Dynamically load a standalone module as an importable module."""
    module_path = INTEGRATIONS_DIR / filename
    spec = importlib.util.spec_from_file_location(filename, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


llamaindex_wrapper = _load_module("llamaindex_wrapper.py")


# --- normalize_node_texts(): pure function, no llama-index required -----


def test_normalize_node_texts_normalizes_strings() -> None:
    result = llamaindex_wrapper.normalize_node_texts(["مَرْحَـبًا بكم"])
    assert result == ["مرحبا بكم"]


def test_normalize_node_texts_preserves_order_and_length() -> None:
    texts = ["مَرْحَـبًا", "hello", "أَهْلاً"]
    result = llamaindex_wrapper.normalize_node_texts(texts)
    assert len(result) == 3
    assert result == ["مرحبا", "hello", "أهلا"]


def test_normalize_node_texts_empty_list_returns_empty_list() -> None:
    assert llamaindex_wrapper.normalize_node_texts([]) == []


def test_normalize_node_texts_non_list_input_raises_type_error() -> None:
    with pytest.raises(TypeError):
        llamaindex_wrapper.normalize_node_texts("not a list")  # type: ignore[arg-type]


def test_normalize_node_texts_non_str_element_raises_type_error() -> None:
    with pytest.raises(TypeError):
        llamaindex_wrapper.normalize_node_texts([123])  # type: ignore[list-item]


# --- ArabicNodeNormalizer: behavior depends on whether llama-index-core is installed --


def test_normalizer_raises_clear_error_without_llamaindex() -> None:
    if llamaindex_wrapper._LLAMAINDEX_AVAILABLE:
        pytest.skip("llama-index-core is installed in this environment; covered by the class test instead")
    with pytest.raises(ImportError, match="ar-tokenwise\\[integrations\\]"):
        llamaindex_wrapper.ArabicNodeNormalizer()


@pytest.mark.skipif(
    not llamaindex_wrapper._LLAMAINDEX_AVAILABLE,
    reason="llama-index-core not installed in this environment",
)
def test_normalizer_call_normalizes_node_text_in_place() -> None:
    from llama_index.core.schema import TextNode

    nodes = [TextNode(text="مَرْحَـبًا بكم")]
    normalizer = llamaindex_wrapper.ArabicNodeNormalizer()
    result = normalizer(nodes)

    assert result[0].text == "مرحبا بكم"


@pytest.mark.skipif(
    not llamaindex_wrapper._LLAMAINDEX_AVAILABLE,
    reason="llama-index-core not installed in this environment",
)
def test_normalizer_accepts_level_as_pydantic_field() -> None:
    from llama_index.core.schema import TextNode

    nodes = [TextNode(text="أَحْمَد")]
    normalizer = llamaindex_wrapper.ArabicNodeNormalizer(level="medium")
    result = normalizer(nodes)

    assert result[0].text == "احمد"


@pytest.mark.skipif(
    not llamaindex_wrapper._LLAMAINDEX_AVAILABLE,
    reason="llama-index-core not installed in this environment",
)
def test_normalizer_preserves_node_order() -> None:
    from llama_index.core.schema import TextNode

    nodes = [TextNode(text="أَوَّل", id_="1"), TextNode(text="ثانٍ", id_="2")]
    normalizer = llamaindex_wrapper.ArabicNodeNormalizer()
    result = normalizer(nodes)

    assert result[0].id_ == "1"
    assert result[1].id_ == "2"


@pytest.mark.skipif(
    not llamaindex_wrapper._LLAMAINDEX_AVAILABLE,
    reason="llama-index-core not installed in this environment",
)
@pytest.mark.anyio
async def test_normalizer_acall_matches_call() -> None:
    from llama_index.core.schema import TextNode

    nodes = [TextNode(text="مَرْحَـبًا")]
    normalizer = llamaindex_wrapper.ArabicNodeNormalizer()
    result = await normalizer.acall(nodes)

    assert result[0].text == "مرحبا"