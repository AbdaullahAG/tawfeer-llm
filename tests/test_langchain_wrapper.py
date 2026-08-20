"""Unit tests for integrations/langchain_wrapper.py.

This module lives outside the installed package (integrations/, not
src/), so it's loaded dynamically via importlib -- same pattern as
tests/test_litellm_plugin.py for litellm_plugin.py.
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


langchain_wrapper = _load_module("langchain_wrapper.py")


# --- normalize_page_contents(): pure function, no langchain required ----


def test_normalize_page_contents_normalizes_strings() -> None:
    result = langchain_wrapper.normalize_page_contents(["مَرْحَـبًا بكم"])
    assert result == ["مرحبا بكم"]


def test_normalize_page_contents_preserves_order_and_length() -> None:
    contents = ["مَرْحَـبًا", "hello", "أَهْلاً"]
    result = langchain_wrapper.normalize_page_contents(contents)
    assert len(result) == 3
    assert result == ["مرحبا", "hello", "أهلا"]


def test_normalize_page_contents_empty_list_returns_empty_list() -> None:
    assert langchain_wrapper.normalize_page_contents([]) == []


def test_normalize_page_contents_non_list_input_raises_type_error() -> None:
    with pytest.raises(TypeError):
        langchain_wrapper.normalize_page_contents("not a list")  # type: ignore[arg-type]


def test_normalize_page_contents_non_str_element_raises_type_error() -> None:
    with pytest.raises(TypeError):
        langchain_wrapper.normalize_page_contents([123])  # type: ignore[list-item]


# --- ArabicDocumentNormalizer: behavior depends on whether langchain-core is installed --


def test_normalizer_raises_clear_error_without_langchain() -> None:
    if langchain_wrapper._LANGCHAIN_AVAILABLE:
        pytest.skip("langchain-core is installed in this environment; covered by the class test instead")
    with pytest.raises(ImportError, match="ar-tokenwise\\[integrations\\]"):
        langchain_wrapper.ArabicDocumentNormalizer()


@pytest.mark.skipif(
    not langchain_wrapper._LANGCHAIN_AVAILABLE,
    reason="langchain-core not installed in this environment",
)
def test_normalizer_transform_documents_normalizes_content_and_keeps_metadata() -> None:
    from langchain_core.documents import Document

    docs = [Document(page_content="مَرْحَـبًا بكم", metadata={"source": "test.txt"})]
    normalizer = langchain_wrapper.ArabicDocumentNormalizer()
    result = normalizer.transform_documents(docs)

    assert result[0].page_content == "مرحبا بكم"
    assert result[0].metadata == {"source": "test.txt"}


@pytest.mark.skipif(
    not langchain_wrapper._LANGCHAIN_AVAILABLE,
    reason="langchain-core not installed in this environment",
)
def test_normalizer_preserves_document_order() -> None:
    from langchain_core.documents import Document

    docs = [
        Document(page_content="أَوَّل", metadata={"id": 1}),
        Document(page_content="ثانٍ", metadata={"id": 2}),
    ]
    normalizer = langchain_wrapper.ArabicDocumentNormalizer()
    result = normalizer.transform_documents(docs)

    assert result[0].metadata == {"id": 1}
    assert result[1].metadata == {"id": 2}


@pytest.mark.skipif(
    not langchain_wrapper._LANGCHAIN_AVAILABLE,
    reason="langchain-core not installed in this environment",
)
@pytest.mark.anyio
async def test_normalizer_async_matches_sync() -> None:
    from langchain_core.documents import Document

    docs = [Document(page_content="مَرْحَـبًا", metadata={})]
    normalizer = langchain_wrapper.ArabicDocumentNormalizer()
    result = await normalizer.atransform_documents(docs)

    assert result[0].page_content == "مرحبا"