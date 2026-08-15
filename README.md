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

## Quick start

```python
from ar_tokenwise import normalize, NormalizationLevel, report_savings

text = "مَرْحَـبًا بكم، اليوم ٢٠٢٦."
optimized = normalize(text, level=NormalizationLevel.LIGHT)

report = report_savings(text, optimized, cost_per_million_tokens=3.0)
print(report.tokens_saved, report.percent_saved, report.estimated_cost_savings_usd)
```

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
register. **Current corpus is a 27-sentence seed set** — early numbers are
directional, not final claims. Run it yourself:

```bash
python benchmark/run_benchmark.py
```

## Roadmap

- **v1 (current):** conservative normalization, transparent token-delta
  reporting, reproducible fertility benchmark.
- **v2:** dialect detection (experimental), mixed-text handling, safety
  modes for sensitive registers (legal/religious/medical), RAG chunking helper.
- **v3:** formulaic-expression compression, cache-key normalization,
  LiteLLM/LangChain/LlamaIndex integrations.
  
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
- **No mixed-text or dialect-aware handling in v1** — planned for v2,
  see the Roadmap above.

## License

Apache License 2.0 — see [LICENSE](./LICENSE).