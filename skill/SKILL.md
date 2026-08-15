---
name: arabic-token-optimizer
description: Use this skill when preparing, summarizing, chunking, or forwarding Arabic text (MSA or dialectal) as part of an LLM prompt, and token cost, context-window usage, or latency matters. It conservatively normalizes Arabic text to reduce token count without changing meaning, and reports exact before/after token savings. Trigger on any task involving long Arabic documents, Arabic chat transcripts, Arabic RAG context, or explicit requests to reduce Arabic prompt/token cost.
---

# Arabic Token Optimizer

## What this does

Arabic text is measurably more expensive to tokenize than English for
equivalent content (more tokens per word). This skill wraps the
`ar-tokenwise` Python library to:

1. Normalize Arabic text conservatively (remove decorative elongation,
   optional pronunciation marks, unify digit forms — and optionally unify
   alef/yeh orthographic variants).
2. Report exact token savings (before/after count, percent saved, and
   estimated dollar cost saved if a price is supplied).

This does **not** claim to improve model comprehension — only to reduce
token count, cost, and context footprint. Treat it as a pre-processing
step, not a rewriting or summarization step.

**Note on this library's status:** `ar-tokenwise` currently lives in this
same repository (`src/ar_tokenwise`) and is not yet published to PyPI.
Install it locally from the repo root:

```bash
pip install -e ".[tokenizers]"
```

This single command covers both scripts below — the `tokenizers` extra
is required for `report_cli.py`'s real token counts; `normalize_cli.py`
works without it too, since it's included either way.

## Do NOT normalize this text at all

- **Quranic, Hadith, or other liturgical text where diacritics (tashkeel)
  carry grammatical meaning (i'rab).** Even the `light` level removes
  tashkeel, which is safe for ordinary prose but not for text where
  vowel marks are part of the meaning, not decoration. Skip normalization
  entirely for such text rather than assuming any level is safe.
- **Numeric identifiers embedded in text** (ID numbers, phone numbers,
  license plates, product codes) — digit-form unification (e.g. `٣٢١` ->
  `321`) changes the literal character representation. If a downstream
  system matches or deduplicates on the exact original string, this can
  break that matching. If your text mixes prose and identifiers, review
  before normalizing, or extract identifiers first.
- **Text that will be displayed verbatim to an end user** (e.g. a legal
  quote rendered in a UI). This skill is for text going *into* a model
  prompt, not text coming back out for display. Normalizing user-facing
  text risks silently altering a quote's original spelling.

## Use `light` only (never `medium`) for

Legal, religious, or medical text in general — `medium` additionally
unifies alef/yeh orthographic variants, which is unnecessary risk when
exact spelling may matter to the reader, on top of the tashkeel caveat
above.

## Known v1 limitations

- **No automatic content-type detection.** This skill does not scan text
  for religious/legal/medical markers — the caller (you, the agent) must
  decide whether normalization is appropriate based on the guidance
  above. Dialect-aware and content-aware safety modes are planned for
  v2 — see the "Roadmap" section in the repository's
  [README](../README.md).
- **No mixed Arabic-English-numeral handling.** Non-Arabic characters
  pass through untouched, but there is no special logic yet for
  code-switched text (e.g. product names, technical terms). This is
  planned for v2 — see the "Roadmap" section in the repository's
  [README](../README.md).
- **Realistic savings vary widely.** The example below shows ~17%
  savings, but that example text is diacritized. Most everyday Arabic
  text (chat messages, articles, casual writing) has no diacritics to
  begin with, so the tatweel/tashkeel-removal savings on such text will
  be smaller — often close to zero — and most of the benefit will come
  from digit unification when present. Always run `report_cli.py` on
  your actual text rather than assuming the example's percentage.

## Token counts are tokenizer-specific

`report_cli.py`'s default counter uses tiktoken's `o200k_base` encoding,
which approximates GPT-4o-family tokenization. It is **not** the exact
tokenizer used by Claude, Gemini, or other models — actual token counts
and savings on those APIs may differ. Treat the reported numbers as a
representative estimate for the encoding used, not an exact count for
every provider.

## Reversibility

`normalize()` is a pure function: it returns a new string and never
modifies the input. Still, the CLI scripts here only print the
normalized output — they do not persist the original for you. If you
need to keep the original for audit or rollback, save it yourself
(e.g. keep the source file, or capture `--input`'s content) before
overwriting anything with the normalized version.

## Idempotency

Running normalization on already-normalized text is safe and does not
degrade it further — `normalize(normalize(text)) == normalize(text)` at
the same level (covered by the library's test suite). Re-running this
skill on previously-processed text will not compound changes.

## RAG / retrieval consistency warning

If you normalize text before embedding/indexing it, you must apply the
**same** normalization to queries at retrieval time, or embedding
similarity may degrade due to a mismatch between how indexed content and
queries were processed. If a piece of text is stored as a source of
truth (e.g. the canonical version of a document), consider not
normalizing it at all, and only normalizing a transient copy prepared
for a specific prompt.

## When to use it

- Before sending a long Arabic document, chat transcript, or RAG context
  chunk to an LLM API call, when the user cares about cost or context
  size — subject to the exclusions above.
- When the user explicitly asks to reduce token usage or cost for Arabic
  text.
- When preparing Arabic text for repeated/batched API calls where token
  savings compound.

## How to use it

Two scripts, both plain CLI wrappers around the `ar_tokenwise` library.
Run them with the bash tool. Both read text from a file path or stdin,
never require `eval`/`exec`, and cap input size to avoid accidentally
processing an oversized file.

Both scripts reject (not truncate) input files larger than 20 MB with a
clear error message — they never silently process a partial file. Stdin
input has no size cap; pipe large text through a file if you need the
guard. If a file is not valid UTF-8, both scripts fail immediately with
an explicit "not valid UTF-8" error — they never guess an encoding or
process corrupted bytes silently.

### 1. Normalize text

```bash
python skill/scripts/normalize_cli.py --level light --input /path/to/text.txt
# or via stdin:
echo "مَرْحَـبًا بكم" | python skill/scripts/normalize_cli.py --level light
```

Prints the normalized text to stdout. Default level is `light`. See
"Do NOT normalize this text at all" above before running this on
sensitive registers.

### 2. Get a token savings report

```bash
python skill/scripts/report_cli.py --input /path/to/text.txt --level light --cost-per-million 3.0
```

Prints a JSON report to stdout:

```json
{
  "original_tokens": 42,
  "optimized_tokens": 35,
  "tokens_saved": 7,
  "percent_saved": 16.67,
  "original_fertility": 2.1,
  "optimized_fertility": 1.75,
  "estimated_cost_savings_usd": 0.000021
}
```

Requires the `tokenizers` extra, installed above via
`pip install -e ".[tokenizers]"`, for real token counts measured against
`o200k_base` (see "Token counts are tokenizer-specific" above). If it
wasn't installed, the script fails with a clear error rather than
guessing token counts.

## Typical workflow for an agent

1. Read the Arabic source text. Keep the original around (see
   "Reversibility") if you might need to revert.
2. Check it against "Do NOT normalize this text at all" above.
3. Run `normalize_cli.py` to get the optimized version.
4. Use the optimized text in the actual LLM prompt/context you're
   building — never for text that will be displayed verbatim to a user.
5. Optionally run `report_cli.py` to tell the user exactly how many
   tokens/cost were saved — never state an estimated savings number
   without running the report, and note it's specific to the `o200k_base`
   encoding.