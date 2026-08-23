# Arabic Token Optimizer — Advanced Reference

This file is NOT loaded automatically with the skill — read it only
when a task actually needs one of the functions below (none of them
are wired into `normalize_cli.py`/`report_cli.py`; they're Python
library functions used via `from ar_tokenwise import ...` or a short
inline script). See `SKILL.md` for the core, always-relevant usage
(normalization levels, safety exclusions, the two CLI scripts).

## v2/v3 library modules

- **`chunk_text()`** — sentence-boundary-aware chunking for RAG pipelines,
  with a token budget (`min_tokens`/`max_tokens`).
- **`report_mixed_fertility()`** — separate token-per-word fertility for
  the Arabic/non-Arabic/Arabizi portions of code-switched text (Arabizi =
  Arabic written phonetically in Latin letters with digit substitutions,
  e.g. "7abibi"), so a report on a mostly-English sentence with a few
  Arabic words isn't misleading.
- **`detect_dialect()`** — EXPERIMENTAL. Returns a probability-like
  distribution across MSA/Gulf/Egyptian/Levantine/Maghrebi, or an
  explicit `insufficient_text`/`no_signal` status — never a single
  confident label. Low accuracy ceiling even for trained models on this
  task (see `src/ar_tokenwise/dialect.py`'s module docstring for the
  academic citation and reasoning); treat any result as a rough signal,
  not a fact. See `benchmark/README.md` for this project's own measured
  validation accuracy and an important caveat about that number's
  current reliability (it reflects internal consistency, not
  independent real-world accuracy).
- **`check_content_warnings()`** — advisory-only heuristic flags for
  religious/legal/medical content sensitivity. **Never call this
  automatically as part of a normalization pipeline** — it's opt-in by
  design. If you use it in a multi-turn session, **do not repeat the
  same warning category to the user more than once per session**: this
  function is stateless (no built-in repeat-suppression) precisely so
  that decision stays with you, the caller — repeating the same warning
  on every call trains users to ignore it (documented "alert fatigue"
  effect in studied domains). Track what you've already shown in your
  own session state.
- **`smart_prepare()`** — opt-in convenience wrapper combining
  `normalize()` and `check_content_warnings()`: if any warning fires, it
  returns the **original, unnormalized text** plus the warnings instead
  of silently normalizing; you decide what to do next. This does not
  change `check_content_warnings()`'s opt-in nature — you still have to
  call `smart_prepare()` yourself, nothing calls it for you.
- **`generate_cache_key()`** — deterministic SHA-256 key so
  diacritized/undiacritized/alef-variant versions of the same Arabic
  text hash to the same cache/embedding-store key. One-way (not
  reversible, not encryption) — a lookup key, not a security primitive.
  Solves exact-duplicate cache misses specifically; does not by itself
  guarantee general embedding/retrieval consistency (see SKILL.md's RAG
  warning).
- **`optimize_for_caching()`** / **`to_anthropic_cache_blocks()`** —
  reorders prompt segments you label `stable=True/False` so all stable
  (identical-across-calls) content forms one shared prefix, which is
  what providers' prompt-caching (Anthropic explicit breakpoints, OpenAI
  automatic) actually requires to produce a cache hit. This tool trusts
  your `stable`/`dynamic` labels completely — mislabeling can reorder
  content in a way that changes meaning, so only mark segments stable if
  reordering them is genuinely safe.
- **`get_anthropic_counter()`** / **`get_gemini_counter()`** — exact
  (not approximated) token counts via each provider's own official
  count_tokens API. **Makes a real network call on every single
  invocation** (no local caching like `get_default_counter()`'s
  tiktoken) and needs that provider's API key — appropriate for one-off
  reports/benchmarks, never for a per-request hot path. Requires
  `pip install ar-tokenwise[providers]`.

## Framework integrations

Thin, optional wrappers for LiteLLM, LangChain, and LlamaIndex live in
`integrations/` (outside `src/ar_tokenwise`). See the root
[README.md](../README.md)'s Integrations table for what each does, and
`integrations/README.md` for base-class differences between the three
frameworks if you're extending one. Require
`pip install ar-tokenwise[integrations]`.

## Extra installs for the functions above

The base `pip install -e ".[tokenizers]"` in SKILL.md does NOT cover
everything on this page:

```bash
pip install -e ".[tokenizers,providers]"     # for get_anthropic_counter/get_gemini_counter
pip install -e ".[tokenizers,integrations]"  # for LiteLLM/LangChain/LlamaIndex wrappers
pip install -e ".[tokenizers,providers,integrations]"  # for both
```