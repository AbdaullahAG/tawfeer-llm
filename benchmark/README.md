# tawfeer-llm Benchmark

## Corpus status: seed (v1)

`corpus/seed_corpus.jsonl` currently has 27 hand-authored sentences across
4 registers (MSA, dialect x4 regions, mixed Arabic-English, formal/legal
register). This is a starting point, not a statistically representative
sample — treat early results as directional, not final claims.

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