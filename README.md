<div align="center">

# tawfeer-llm

**Arabic text is expensive to tokenize. This library measures exactly how expensive — and safely trims the part that isn't necessary.**

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](./LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Dependencies: none](https://img.shields.io/badge/core%20dependencies-zero-brightgreen.svg)](#install)

`pip install -e .` from source · works with Claude, GPT, Gemini, or any LLM API · no model access needed

</div>

---

## Why

Arabic uses more tokens per word than English for the same content on
most tokenizers — higher cost, faster context exhaustion, slower
responses. `tawfeer-llm` normalizes Arabic conservatively (never
changes meaning) and tells you the **exact, measured** token savings —
not a marketing estimate.

**What it doesn't claim:** normalizing Arabic does not make a model
*understand* it better. This library's entire value case is cost,
context size, and speed — each independently measurable, none of it
guesswork.

## Install

```bash
git clone https://github.com/AbdaullahAG/tawfeer-llm.git && cd tawfeer-llm
pip install -e ".[tokenizers]"
```

<details>
<summary>Optional extras (provider counters, framework integrations)</summary>

```bash
pip install -e ".[tokenizers,providers]"     # exact Anthropic/Gemini token counts
pip install -e ".[tokenizers,integrations]"  # LiteLLM / LangChain / LlamaIndex wrappers
```

</details>

> **First-time network note:** the first call to `report_savings()` (or
> anything using `get_default_counter()`) downloads tiktoken's encoding
> file once, then caches it locally — no network needed after that. On
> a restrictive network, this first call can fail; see
> [`CONTRIBUTING.md`](./CONTRIBUTING.md#a-note-on-tiktokens-first-use-network-call)
> for what that looks like and how to work around it.

## Quick start

```python
from ar_tokenwise import normalize, NormalizationLevel, report_savings

text = "مَرْحَـبًا بكم، اليوم ٢٠٢٦."
optimized = normalize(text, level=NormalizationLevel.LIGHT)

report = report_savings(text, optimized, cost_per_million_tokens=3.0)
print(report.tokens_saved, report.percent_saved, report.estimated_cost_savings_usd)
```

> **Before you normalize anything:** Quranic/Hadith text, embedded ID
> numbers, and text displayed verbatim to a user should **not** be
> normalized — see [`skill/SKILL.md`](./skill/SKILL.md) for the full,
> non-negotiable list.

## What's included

| | |
|---|---|
| 🧹 **Conservative normalization** | Strips decorative elongation & optional diacritics, unifies digit forms — never changes meaning |
| 📊 **Transparent reporting** | Real before/after token counts against an actual tokenizer, never estimated |
| ✂️ **RAG chunking** | Sentence-boundary-aware, token-budgeted `chunk_text()` |
| 🔤 **Mixed-text & Arabizi handling** | Separate fertility for Arabic / Latin / Arabizi ("7abibi") words in the same text |
| 🗣️ **Dialect signal** *(experimental)* | Probability distribution across MSA/Gulf/Egyptian/Levantine/Maghrebi — never a confident single label |
| ⚠️ **Content-sensitivity warnings** | Advisory-only religious/legal/medical flags, opt-in, never automatic |
| 🔑 **Cache-key generation** | Diacritized & undiacritized text hash to the same cache/embedding key |
| ⚡ **Prompt-caching optimizer** | Reorders labeled prompt segments for Anthropic/OpenAI prompt-cache hits |
| 🎯 **Exact provider token counts** | Real counts via Anthropic's and Gemini's own APIs, not an approximation |
| 🔌 **Framework integrations** | Thin, optional wrappers for LiteLLM, LangChain, LlamaIndex |

Every item above is a heuristic where a heuristic is involved, and says
so in its own docstring — see [`skill/REFERENCE.md`](./skill/REFERENCE.md)
for full usage and every documented limitation before relying on one in
production.

## How this compares

| | Works with closed APIs | Arabic-specific | Reports token savings |
|---|:---:|:---:|:---:|
| AraToken / aranizer *(alt. tokenizers)* | ❌ | ✅ | ❌ |
| PyArabic *(text preprocessing)* | ✅ | ✅ | ❌ |
| **tawfeer-llm** | ✅ | ✅ | ✅ |

AraToken/aranizer require retraining or hosting your own model. PyArabic
has no concept of LLM token cost. `tawfeer-llm` is middleware: drop-in
normalization plus measurable reporting for any API you already call.

## Benchmark

242 hand-authored sentences across MSA, four dialect regions, mixed
Arabic-English, and formal/legal register — run it yourself:

```bash
python benchmark/run_benchmark.py
```

Full corpus methodology, a reusable quality-check tool, and honest
caveats on every measured number: [`benchmark/README.md`](./benchmark/README.md).

## Docs

| | |
|---|---|
| [`skill/SKILL.md`](./skill/SKILL.md) | Core usage rules — read this before normalizing anything sensitive |
| [`skill/REFERENCE.md`](./skill/REFERENCE.md) | Every advanced module, in full detail |
| [`benchmark/README.md`](./benchmark/README.md) | Corpus methodology & measured results |
| [`integrations/README.md`](./integrations/README.md) | Notes for extending a framework integration |
| [`PUBLISHING.md`](./PUBLISHING.md) | PyPI release checklist (not yet published) |
| [`CONTRIBUTING.md`](./CONTRIBUTING.md) | Dev setup, running tests, code conventions, PR process |

## License

Apache License 2.0 — see [LICENSE](./LICENSE).