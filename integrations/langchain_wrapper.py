"""LangChain document transformer: normalize Arabic text in RAG pipelines.

Optional integration -- requires `langchain-core` installed separately
(not a hard dependency of ar-tokenwise). Install with:
    pip install ar-tokenwise[integrations]

This targets LangChain's document-loading stage: normalize Document
page_content BEFORE chunking/embedding, addressing the exact scenario
SKILL.md's "RAG / retrieval consistency warning" describes -- as long as
you also normalize queries at retrieval time with the same level, index
and query stay consistent.

Split into two parts, same pattern as litellm_plugin.py:
1. normalize_page_contents() -- pure function operating on plain strings,
   no langchain dependency needed, fully testable in isolation.
2. ArabicDocumentNormalizer -- a thin LangChain BaseDocumentTransformer
   wrapper around part 1, only usable if langchain-core is installed.
"""

from __future__ import annotations

from typing import Any, Sequence

from ar_tokenwise.normalize import NormalizationLevel, normalize

try:
    from langchain_core.documents import BaseDocumentTransformer, Document

    _LANGCHAIN_AVAILABLE = True
except ImportError:
    BaseDocumentTransformer = object  # fallback base so class definition doesn't crash
    Document = None
    _LANGCHAIN_AVAILABLE = False


def normalize_page_contents(
    contents: list[str],
    level: NormalizationLevel = NormalizationLevel.LIGHT,
) -> list[str]:
    """Normalize a list of raw page_content strings.

    Pure and lightweight (regex/lookup only, no I/O, no langchain
    dependency) -- safe to call on every document in a loading pipeline.

    Args:
        contents: A list of page_content strings.
        level: Normalization aggressiveness applied to each string.

    Returns:
        A new list of normalized strings, same length and order.

    Raises:
        TypeError: if ``contents`` is not a list, or any element is not
            a string.
    """
    if not isinstance(contents, list):
        raise TypeError(f"normalize_page_contents() expects list, got {type(contents).__name__}")

    result: list[str] = []
    for content in contents:
        if not isinstance(content, str):
            raise TypeError(
                f"each page_content must be a str, got {type(content).__name__}"
            )
        result.append(normalize(content, level=level))
    return result


class ArabicDocumentNormalizer(BaseDocumentTransformer):  # type: ignore[misc]
    """LangChain document transformer that normalizes Arabic page_content.

    Usage:
        from langchain_core.document_loaders import ...
        from integrations.langchain_wrapper import ArabicDocumentNormalizer

        docs = loader.load()
        normalizer = ArabicDocumentNormalizer()
        docs = normalizer.transform_documents(docs)
        # then chunk/embed docs as usual

    IMPORTANT: apply the same normalization level to queries at retrieval
    time (see SKILL.md's "RAG / retrieval consistency warning") or
    embedding similarity can degrade.

    Requires langchain-core installed: pip install ar-tokenwise[integrations]
    """

    def __init__(self, level: NormalizationLevel = NormalizationLevel.LIGHT) -> None:
        if not _LANGCHAIN_AVAILABLE:
            raise ImportError(
                "ArabicDocumentNormalizer requires langchain-core. Install "
                "it with `pip install ar-tokenwise[integrations]`, or use "
                "normalize_page_contents() directly without langchain."
            )
        self.level = level

    def transform_documents(
        self, documents: Sequence[Any], **kwargs: Any
    ) -> Sequence[Any]:
        """Return new Documents with normalized page_content (metadata preserved)."""
        contents = [doc.page_content for doc in documents]
        normalized_contents = normalize_page_contents(contents, level=self.level)
        return [
            Document(page_content=new_content, metadata=doc.metadata)
            for doc, new_content in zip(documents, normalized_contents)
        ]

    async def atransform_documents(
        self, documents: Sequence[Any], **kwargs: Any
    ) -> Sequence[Any]:
        """Async version -- delegates to the sync path (no I/O to await here)."""
        return self.transform_documents(documents, **kwargs)