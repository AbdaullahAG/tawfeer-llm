# tawfeer-llm (ar-tokenwise)

Lightweight, conservative Arabic text normalization and transparent
token-usage reporting for LLM API calls — works with any closed API
(Claude, GPT, Gemini, ...), no model access or retraining required.

## Why this exists

Arabic text is measurably more expensive to send to most LLM APIs:
higher tokens-per-word (fertility) than English for the same content,
which means higher cost, faster context exhaustion, and slower
generation. This library reduces that overhead through safe,
meaning-preserving text normalization, and reports the exact token
savings — no estimates, no marketing numbers.

**What this library does NOT claim:** normalization here does not claim
to improve model comprehension or output quality through morphological
awareness. Recent research (LREC 2026) found mixed evidence on whether
tokenizer morphological alignment affects generation quality at all.
This library's value proposition is strictly: lower cost, smaller
context footprint, faster responses — all independently measurable.

## Install

**Not yet published to PyPI.** Install from source:

```bash
git clone <this-repo-url>
cd tawfeer-llm
pip install -e ".[tokenizers]"
```

The `tokenizers` extra pulls in `tiktoken` for real token-count reports
and benchmarks; core normalization works without it.

For the optional LiteLLM/LangChain/LlamaIndex integrations (see
[Integrations](#integrations) below), install:

```bash
pip install -e ".[tokenizers,integrations]"
```

## Quick start

```python
from ar_tokenwise import normalize, NormalizationLevel, report_savings

text = "مَرْحَـبًا بكم، اليوم ٢٠٢٦."
optimized = normalize(text, level=NormalizationLevel.LIGHT)

report = report_savings(text, optimized, cost_per_million_tokens=3.0)
print(report.tokens_saved, report.percent_saved, report.estimated_cost_savings_usd)
```

## Beyond normalization

The core library also includes (all documented in-module and covered by
tests -- see each module's docstring for full details and caveats):

```python
from ar_tokenwise import (
    chunk_text,             # sentence-boundary-aware chunking for RAG
    report_mixed_fertility, # separate fertility for Arabic/Latin/Arabizi text
    detect_dialect,         # EXPERIMENTAL dialect signal (probability, not a fact)
    check_content_warnings, # advisory-only religious/legal/medical flags
    generate_cache_key,     # stable cache/embedding key across diacritic variants
    optimize_for_caching,   # reorder prompt segments for provider prompt caching
)
```

**Prompt caching** (Anthropic/OpenAI etc.) requires an identical, unchanged
prefix across requests — label your prompt pieces as stable/dynamic and
this reorders them so the stable content forms one shared prefix:

```python
from ar_tokenwise import PromptSegment, optimize_for_caching, to_anthropic_cache_blocks

segments = [
    PromptSegment(content=system_prompt, stable=True),
    PromptSegment(content=user_message, stable=False),
]
optimized = optimize_for_caching(segments)
blocks = to_anthropic_cache_blocks(optimized)  # ready for the `content` of a message
```

**Read the module docstrings before using `detect_dialect()` or
`check_content_warnings()`** -- both are heuristic, both document their
accuracy limitations explicitly, and neither should be treated as a
guarantee. `benchmark/README.md` has measured (not estimated) validation
numbers for dialect detection, with an important caveat on their current
reliability.

## Integrations

Optional, thin wrappers around the core library for popular frameworks —
none are required to use `ar-tokenwise` directly. Install with
`pip install ar-tokenwise[integrations]`.

| Framework | File | What it does |
|---|---|---|
| LiteLLM | `integrations/litellm_plugin.py` | `async_pre_call_hook` that normalizes `messages` before the model call |
| LangChain | `integrations/langchain_wrapper.py` | `BaseDocumentTransformer` that normalizes `Document.page_content` |
| LlamaIndex | `integrations/llamaindex_wrapper.py` | `TransformComponent` that normalizes node text in an ingestion pipeline |

Each file works standalone without its target framework installed too —
the core normalization function in each (`normalize_messages()`,
`normalize_page_contents()`, `normalize_node_texts()`) has no framework
dependency; only the class wrapper around it does.

**If you use these for RAG indexing:** apply the same normalization
level to queries at retrieval time, or see `generate_cache_key()` above
for exact-duplicate cache-key matching — see SKILL.md's "RAG / retrieval
consistency warning" for why this matters.

## How this compares

| | Works with closed APIs (no retraining) | Arabic-specific | Reports token savings |
|---|---|---|---|
| **AraToken / aranizer** (alternative tokenizers) | ❌ requires model integration | ✅ | ❌ |
| **PyArabic** (raw text preprocessing) | ✅ | ✅ | ❌ |
| **tawfeer-llm** | ✅ | ✅ | ✅ |

AraToken and aranizer are full tokenizer replacements — they require
retraining or hosting your own model. PyArabic is a mature, general-purpose
Arabic text toolkit but has no concept of LLM token cost or savings
reporting. tawfeer-llm sits specifically at the middleware layer: drop-in
normalization plus measurable, transparent reporting for any API you
already call.

## Benchmark

See [`benchmark/`](./benchmark) for the reproducible fertility benchmark
across MSA, regional dialects, mixed Arabic-English text, and formal/legal
register. **Current corpus is a 242-sentence hand-authored set** — still
curated, not large-scale/naturally-occurring, so treat numbers as
directional, not final claims. Run it yourself:

```bash
python benchmark/run_benchmark.py
```

`benchmark/` also has a separate dialect-detection validation corpus and
runner — see `benchmark/README.md` for both, including an important
caveat on the dialect-detection accuracy number's current reliability.

## Roadmap

- **v1 (shipped):** conservative normalization, transparent token-delta
  reporting, reproducible fertility benchmark.
- **v2 (shipped):** dialect detection (experimental), mixed-text and
  Arabizi handling, advisory content-sensitivity warnings, RAG chunking
  helper.
- **v3 (shipped, partial):** cache-key generation, LiteLLM/LangChain/
  LlamaIndex integrations.
  - **Not yet built: formulaic-expression compression.** This needs
    real usage data (common stock phrases/openings actually seen in
    production Arabic text) to build a marker dictionary responsibly --
    building it now, without that data, would mean guessing at patterns
    instead of measuring them, which conflicts with this project's
    "real numbers, not assumptions" approach used everywhere else. It
    stays deliberately unbuilt until real usage data exists, not
    forgotten.
- **Not yet done (non-code):** publish to PyPI. Currently install-from-
  source only (see Install above).

## Limitations & honest expectations

- **Diacritized religious/liturgical text** (Quranic verses, Hadith)
  should not be normalized at all — diacritics there carry grammatical
  meaning, not decoration. See `skill/SKILL.md` for the full list of
  text types this library should not be applied to.
- **Realistic savings on everyday text are often smaller** than a
  diacritized example suggests — most modern Arabic writing has no
  diacritics to remove to begin with.
- **Reported token counts are tokenizer-specific.** The default counter
  uses tiktoken's `o200k_base` encoding (GPT-4o family) as a proxy — it
  is not the exact tokenizer for every model.
- **`detect_dialect()` is a heuristic with a low methodological ceiling**
  — even state-of-the-art academic systems (NADI 2024) only reach ~50%
  F1 on this task. Treat any result as a rough signal, not a fact.
- **`check_content_warnings()` is advisory only** and never blocks or
  modifies text — a false negative (missing sensitive content) is
  possible. It is opt-in and stateless by design; see SKILL.md for
  guidance on avoiding repeated-warning fatigue in your own session
  logic.

## License

Apache License 2.0 — see [LICENSE](./LICENSE).