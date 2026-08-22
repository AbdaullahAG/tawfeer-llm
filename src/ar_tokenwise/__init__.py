# src/ar_tokenwise/__init__.py
"""ar-tokenwise: conservative Arabic text normalization and token-usage
reporting for LLM API calls.
"""

# NOTE: ar_tokenwise.formulaic exists as an explicit stub (all functions
# raise NotImplementedError) and is deliberately NOT imported/exported
# here -- it is blocked on real usage data, not forgotten. See its
# module docstring and README.md's "Roadmap" section.

from ar_tokenwise.benchmark import (
    BenchmarkCategory,
    BenchmarkResult,
    CorpusEntry,
    load_corpus,
    render_markdown_table,
    run_benchmark,
)
from ar_tokenwise.cache_keys import canonicalize_for_cache_key, generate_cache_key
from ar_tokenwise.chunking import chunk_text
from ar_tokenwise.dialect import (
    DetectionStatus,
    DialectCategory,
    DialectDetectionResult,
    detect_dialect,
)
from ar_tokenwise.mixed_text import (
    MixedTextReport,
    WordCategory,
    classify_word,
    report_mixed_fertility,
)
from ar_tokenwise.normalize import NormalizationLevel, normalize
from ar_tokenwise.prompt_caching import (
    CacheOptimizedPrompt,
    PromptSegment,
    optimize_for_caching,
    to_anthropic_cache_blocks,
)
from ar_tokenwise.provider_counters import get_anthropic_counter, get_gemini_counter
from ar_tokenwise.report import TokenReport, get_default_counter, report_savings
from ar_tokenwise.safety_modes import (
    ConfidenceLevel,
    ContentCategory,
    ContentWarning,
    check_content_warnings,
)
from ar_tokenwise.smart_prepare import SmartPrepareResult, smart_prepare

__version__ = "0.1.0"

__all__ = [
    "normalize",
    "NormalizationLevel",
    "report_savings",
    "TokenReport",
    "get_default_counter",
    "load_corpus",
    "run_benchmark",
    "render_markdown_table",
    "BenchmarkCategory",
    "BenchmarkResult",
    "CorpusEntry",
    "chunk_text",
    "report_mixed_fertility",
    "MixedTextReport",
    "WordCategory",
    "classify_word",
    "detect_dialect",
    "DialectDetectionResult",
    "DialectCategory",
    "DetectionStatus",
    "check_content_warnings",
    "ContentWarning",
    "ContentCategory",
    "ConfidenceLevel",
    "generate_cache_key",
    "canonicalize_for_cache_key",
    "PromptSegment",
    "CacheOptimizedPrompt",
    "optimize_for_caching",
    "to_anthropic_cache_blocks",
    "get_anthropic_counter",
    "get_gemini_counter",
    "smart_prepare",
    "SmartPrepareResult",
    "__version__",
]