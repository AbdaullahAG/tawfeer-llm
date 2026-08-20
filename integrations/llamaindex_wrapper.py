"""LlamaIndex TransformComponent: normalize Arabic text in ingestion pipelines.

Optional integration -- requires `llama-index-core` installed separately
(not a hard dependency of ar-tokenwise). Install with:
    pip install ar-tokenwise[integrations]

NOTE (verified against the installed library, not assumed): unlike
LangChain's BaseDocumentTransformer (a plain ABC), LlamaIndex's
TransformComponent is a Pydantic BaseModel. This means the `level`
option must be declared as a Pydantic field (class-level annotation),
not a plain __init__ parameter -- see ArabicNodeNormalizer below.

IMPORTANT: this file deliberately does NOT use
`from __future__ import annotations`, unlike every other module in this
project. Verified by hitting the failure directly: with deferred
(string) annotations enabled, Pydantic v2 fails to resolve the `level`
field's type at class-definition time and raises
`PydanticUserError: ArabicNodeNormalizer is not fully defined`. This is
a real, reproduced incompatibility between deferred annotations and
Pydantic BaseModel field resolution in this context, not a
precautionary guess -- so this file's imports use real Python 3.9+
built-in generics (`list[str]`) directly instead.

Split into two parts, same pattern as litellm_plugin.py and
langchain_wrapper.py:
1. normalize_node_texts() -- pure function operating on plain strings,
   no llama-index dependency needed, fully testable in isolation.
2. ArabicNodeNormalizer -- a thin LlamaIndex TransformComponent wrapper
   around part 1, only usable if llama-index-core is installed.
"""

from typing import Any, Sequence

from ar_tokenwise.normalize import NormalizationLevel, normalize

try:
    from llama_index.core.schema import TransformComponent

    _LLAMAINDEX_AVAILABLE = True
except ImportError:
    TransformComponent = object  # fallback base so class definition doesn't crash
    _LLAMAINDEX_AVAILABLE = False


def normalize_node_texts(
    texts: list[str],
    level: NormalizationLevel = NormalizationLevel.LIGHT,
) -> list[str]:
    """Normalize a list of raw node text strings.

    Pure and lightweight (regex/lookup only, no I/O, no llama-index
    dependency) -- safe to call on every node in an ingestion pipeline.

    Args:
        texts: A list of node text strings.
        level: Normalization aggressiveness applied to each string.

    Returns:
        A new list of normalized strings, same length and order.

    Raises:
        TypeError: if ``texts`` is not a list, or any element is not a
            string.
    """
    if not isinstance(texts, list):
        raise TypeError(f"normalize_node_texts() expects list, got {type(texts).__name__}")

    result: list[str] = []
    for text in texts:
        if not isinstance(text, str):
            raise TypeError(f"each node text must be a str, got {type(text).__name__}")
        result.append(normalize(text, level=level))
    return result


class ArabicNodeNormalizer(TransformComponent):  # type: ignore[misc]
    """LlamaIndex TransformComponent that normalizes Arabic node text.

    Usage:
        from llama_index.core.ingestion import IngestionPipeline
        from integrations.llamaindex_wrapper import ArabicNodeNormalizer

        pipeline = IngestionPipeline(transformations=[ArabicNodeNormalizer(), ...])
        nodes = pipeline.run(documents=documents)

    IMPORTANT: apply the same normalization level to queries at retrieval
    time (see SKILL.md's "RAG / retrieval consistency warning") or
    embedding similarity can degrade.

    Requires llama-index-core installed: pip install ar-tokenwise[integrations]

    Note: `level` is a Pydantic field (TransformComponent is a Pydantic
    BaseModel), not a plain constructor parameter -- both
    `ArabicNodeNormalizer()` and `ArabicNodeNormalizer(level="medium")`
    work, matching standard Pydantic model construction.
    """

    level: NormalizationLevel = NormalizationLevel.LIGHT

    def __init__(self, **data: Any) -> None:
        if not _LLAMAINDEX_AVAILABLE:
            raise ImportError(
                "ArabicNodeNormalizer requires llama-index-core. Install "
                "it with `pip install ar-tokenwise[integrations]`, or use "
                "normalize_node_texts() directly without llama-index."
            )
        super().__init__(**data)

    def __call__(self, nodes: Sequence[Any], **kwargs: Any) -> Sequence[Any]:
        """Normalize each node's text in place, then return the same nodes."""
        texts = [node.text for node in nodes]
        normalized_texts = normalize_node_texts(texts, level=self.level)
        for node, new_text in zip(nodes, normalized_texts):
            node.text = new_text
        return nodes

    async def acall(self, nodes: Sequence[Any], **kwargs: Any) -> Sequence[Any]:
        """Async version -- delegates to the sync path (no I/O to await here)."""
        return self(nodes, **kwargs)