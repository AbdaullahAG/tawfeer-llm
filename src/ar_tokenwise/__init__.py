"""ar-tokenwise: conservative Arabic text normalization and token-usage
reporting for LLM API calls.
"""

from ar_tokenwise.benchmark import (
    BenchmarkCategory,
    BenchmarkResult,
    CorpusEntry,
    load_corpus,
    render_markdown_table,
    run_benchmark,
)
from ar_tokenwise.normalize import NormalizationLevel, normalize
from ar_tokenwise.report import TokenReport, get_default_counter, report_savings

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
    "__version__",
]