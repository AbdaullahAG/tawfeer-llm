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