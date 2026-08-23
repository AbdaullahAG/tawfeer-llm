---
name: arabic-token-optimizer
description: Use this skill when preparing, summarizing, chunking, or forwarding Arabic text (MSA or dialectal) as part of an LLM prompt, and token cost, context-window usage, or latency matters. It conservatively normalizes Arabic text to reduce token count without changing meaning, and reports exact before/after token savings. Trigger on any task involving long Arabic documents, Arabic chat transcripts, Arabic RAG context, or explicit requests to reduce Arabic prompt/token cost. Does NOT apply to Quranic/Hadith/liturgical text, embedded numeric IDs, or text displayed verbatim to a user -- see "Do NOT normalize" below before using.
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

This covers both CLI scripts below. `normalize_cli.py` does not use
`tiktoken` at all (pure text transformation, no token counting); only
`report_cli.py` needs the `tokenizers` extra, for its real token counts.
Functions covered in `REFERENCE.md` need additional extras — see that
file, not this one.

## Do NOT normalize this text at all

- **Quranic, Hadith, or other LITURGICAL text where diacritics (tashkeel)
  carry grammatical meaning (i'rab).** Even the `light` level removes
  tashkeel, which is safe for ordinary prose but not for text where
  vowel marks are part of the meaning, not decoration. Skip normalization
  entirely for such text rather than assuming any level is safe. This is
  narrower than "religious text in general" below -- e.g. a news article
  mentioning a mosque opening is ordinary prose (the `light` rule below
  applies); an actual Quranic verse or Hadith quotation, anywhere it
  appears, is not (this rule applies, full stop, regardless of context).
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

Legal, religious (non-liturgical -- see above for the stricter,
do-not-touch-at-all case), or medical text in general — `medium`
additionally unifies alef/yeh orthographic variants, which is
unnecessary risk when exact spelling may matter to the reader, on top
of the tashkeel caveat above. **This is guidance for you to apply, not
an automatic check**: `normalize_cli.py --level medium` will run on any
text you give it, including text that should never have been passed to
it at `medium` -- nothing in the script itself detects or blocks this.

## Known v1 limitations (resolved in v2/v3 — see REFERENCE.md)

- ~~No automatic content-type detection~~ / ~~No mixed
  Arabic-English-numeral handling~~ — both resolved; see
  `REFERENCE.md`'s "v2/v3 library modules" for `safety_modes.py` and
  `mixed_text.py`.
- **Realistic savings vary widely.** Most everyday Arabic text (chat
  messages, articles, casual writing) has no diacritics to begin with,
  so tatweel/tashkeel-removal savings on such text are often close to
  zero, and most benefit comes from digit unification when present.
  Always run `report_cli.py` on your actual text (see its JSON output
  below) rather than assuming any fixed percentage.

## Advanced library modules

Dialect detection, content warnings, prompt caching, exact provider
counters, framework integrations — **not covered here on purpose**.
None are wired into the two CLI scripts below, so documenting them
fully here would add fixed context cost to every skill invocation, even
the common case that only needs the two scripts. See
`skill/REFERENCE.md`, read only if your task actually needs one.

## Token counts are tokenizer-specific

`report_cli.py`'s default counter uses tiktoken's `o200k_base` encoding,
which approximates GPT-4o-family tokenization. It is **not** the exact
tokenizer used by Claude, Gemini, or other models — actual token counts
and savings on those APIs may differ. Treat the reported numbers as a
representative estimate for the encoding used, not an exact count for
every provider. (An exact, provider-specific counter exists too — see
`REFERENCE.md`'s `get_anthropic_counter()`/`get_gemini_counter()` — but
it calls that provider's API over the network on every use, so it's not
part of the default CLI path here.)

## Reversibility & idempotency

`normalize()` is pure: returns a new string, never modifies the input,
and re-normalizing already-normalized text at the same level is a
no-op (`normalize(normalize(text)) == normalize(text)` —
see `tests/test_normalize.py::test_idempotent`). The CLI scripts only
print the normalized output, though — they don't save the original for
you. Keep it yourself (e.g. the source file, or captured `--input`
content) before overwriting anything, if you might need to revert.

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
never require `eval`/`exec`.

**Size guard is file-input-only, not a general limit:** both scripts
reject (not truncate) `--input` files larger than 20 MB with a clear
error message. **Stdin has no size limit at all** — piping a large file
via stdin does not go through this guard. If you need the 20 MB check
to actually apply, pass the text via `--input <path>`, not stdin; do
not treat "pipe via stdin instead" as a way to bypass a limit you
actually want enforced.

If a file is not valid UTF-8, both scripts fail immediately with an
explicit "not valid UTF-8" error — they never guess an encoding or
process corrupted bytes silently.

### 1. Normalize text

```bash
python skill/scripts/normalize_cli.py --level light --input /path/to/text.txt
# or via stdin (no size guard, see above):
echo "مَرْحَـبًا بكم" | python skill/scripts/normalize_cli.py --level light
```

Prints the normalized text to stdout. Default level is `light`. See
"Do NOT normalize this text at all" above before running this on
sensitive registers.

### 2. Get a token savings report

```bash
python skill/scripts/report_cli.py --input /path/to/text.txt --level light --cost-per-million 3.0
```

Prints a JSON report to stdout. Example, run against the diacritized
input `"مَرْحَـبًا بكم، اليوم ٢٠٢٦."` specifically (the ~17% figure
below is tied to THIS exact input, not a general claim — see "Known v1
limitations" above for why undiacritized text saves much less):

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
6. Only if the task needs dialect detection, content warnings, prompt
   caching, exact provider token counts, or a framework integration:
   read `skill/REFERENCE.md` then.