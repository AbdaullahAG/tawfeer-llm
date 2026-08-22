# tawfeer-llm Benchmark

## Corpus status: expanded seed (v3)

`corpus/seed_corpus.jsonl` currently has **242 hand-authored sentences**
across 4 registers (MSA, dialect x4 regions, mixed Arabic-English,
formal/legal register) — expanded from an initial 27-sentence seed,
through 120, then 207, to the current count. This is still a curated,
hand-written set, not a large-scale or naturally-occurring corpus —
treat results as directional and useful for regression-testing changes
to this library, not as a publication-grade academic benchmark.
Publishing this as an independent dataset (e.g. on HuggingFace) is a
reasonable future step once the corpus grows further through real
community contributions, not yet done.

**Known limitation, disclosed rather than hidden:** the first two
expansion passes (up to entry 207) included some dialect content
written as the same underlying sentence translated across all four
dialect regions -- e.g. "my car broke down" phrased once each in Gulf,
Egyptian, Levantine, and Maghrebi. This is a real gap in topical
diversity, not caught until a dedicated quality-check tool was built
(see below) -- 16 such pairs exist among the first 207 entries. Every
entry added from #036/028 onward was written on a genuinely distinct
topic never reused across regions, and verified with the tool below
before being committed. The 16 older pairs are left in place rather
than silently rewritten, since they are still valid, correctly-labeled
sentences -- just less diverse than they should be -- and this note
exists so nobody mistakes them for something the tooling missed.

### Corpus quality checking

`check_corpus_quality.py` is a reusable QA tool (not part of the
installed package) that checks any corpus JSONL file for:
1. Near-duplicate sentences within the same category/region (word-set
   Jaccard similarity).
2. The exact cross-dialect content-reuse pattern described above.
3. Sentence length distribution (word count) per group.

```bash
python benchmark/check_corpus_quality.py
# or against a different file / threshold:
python benchmark/check_corpus_quality.py corpus/dialect_validation.jsonl --threshold 0.6
```

Run this before adding new sentences via PR -- it is the actual
mechanism that caught the limitation described above, not a retroactive
excuse for it.

Contributions of additional labeled sentences (real, naturally occurring
where possible, and license-clear) are welcome via PR to `corpus/`.

## Running the benchmark

```bash
pip install ar-tokenwise[tokenizers]
python benchmark/run_benchmark.py
```

Results are written to `benchmark/results/latest.md` with real fertility
numbers measured against the tiktoken `o200k_base` encoding. No numbers
in this repository are estimated or hardcoded.

**Measured on the current 242-sentence corpus** (LIGHT normalization):
savings remain near-zero for most registers (0.0%-1.5%), since most of
these sentences -- like most everyday Arabic text -- carry no
diacritics to strip in the first place. This is not a bug in the
measurement; it is the exact, honest confirmation of the "realistic
savings are often close to zero" caveat already stated in the root
README -- diacritic removal alone is not where most of this library's
value comes from for typical text. See `results/latest.md` for the
full per-category table after running the command above.

## Dialect detection validation

`corpus/dialect_validation.jsonl` (60 hand-labeled sentences, 12 per
category) is a **separate** corpus from `dialect.py`'s internal marker
lists, so the measured accuracy isn't circular. Run it with:

```bash
python benchmark/run_dialect_validation.py
```

**Important caveat on the current number:** both the marker lists in
`dialect.py` and this validation corpus were authored by the same person
in the same development pass, using the same dialectal knowledge. This
means the measured accuracy (currently ~95%) reflects **internal
consistency**, not independent real-world validation — a genuinely
unfamiliar author or naturally-occurring social media text would likely
score much closer to the academic ceiling for this task (NADI 2024's
best system: 50.57% F1 — see `dialect.py`'s module docstring). Treat the
current number as "the heuristic is internally coherent," not "the
heuristic is ~95% accurate in production." Independent validation by a
different contributor, or against real unconstrained text, is needed
before quoting this number anywhere outside this repository.