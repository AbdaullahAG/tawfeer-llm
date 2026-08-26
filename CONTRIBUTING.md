# Contributing to tawfeer-llm

## Development setup

```bash
git clone https://github.com/AbdaullahAG/tawfeer-llm.git && cd tawfeer-llm
pip install -e ".[tokenizers,dev]"
```

`dev` pulls in `pytest`, `pytest-cov`, and `anyio` (needed for the async
integration tests) — not included in the plain `[tokenizers]` install
from the root [README.md](./README.md), since most users installing the
library don't need test tooling.

## Running tests

```bash
pytest
```

Two tests fail on their own the first time you run this, on any
network that can't reach `openaipublic.blob.core.windows.net`:
`test_get_default_counter_works_when_tiktoken_installed` and
`test_report_cli_prints_valid_json`. This is `tiktoken` downloading its
encoding file on first use (see "A note on tiktoken's first-use network
call" below) — not a broken test. Everything else should pass.

To test the optional integrations too:

```bash
pip install -e ".[tokenizers,dev,integrations,providers]"
pytest
```

## A note on tiktoken's first-use network call

`get_default_counter()` (used by `report_savings()`,
`benchmark/run_benchmark.py`, and `report_cli.py`) needs network access
**the first time it runs**, to download tiktoken's `o200k_base`
encoding file from `openaipublic.blob.core.windows.net`. After that
first successful call, the file is cached locally
(`~/.cache/tiktoken` by default) and no further network access is
needed.

If you're on a restrictive network (some corporate/institutional
firewalls block this specific Microsoft-owned endpoint even though
general internet access works), this first call will fail with an
`HTTPError`. There is no built-in offline fallback for this — either
get that endpoint allowed, pre-populate the cache from a machine that
can reach it, or pass your own `counter` callable to `report_savings()`
instead of relying on `get_default_counter()`.

## Code conventions

- Every public function: type hints on all parameters and the return
  value, a docstring with `Args`/`Returns`/`Raises`.
- Input validation before any processing: `str` type check + a
  `max_length` size guard, via the shared
  `ar_tokenwise._internal.validate_text_input()` — don't reimplement
  this per module.
- No `eval`, `exec`, `pickle`, or shell calls on user-supplied text,
  anywhere.
- New regex: state in a code comment what input pattern it expects and
  why it isn't ReDoS-prone (fixed-width lookarounds, no nested
  quantifiers) — every existing regex in this codebase does this.
- Prefer stdlib over a new dependency. If a new dependency is genuinely
  needed, it's optional (its own `pyproject.toml` extra) unless the
  core library cannot work without it at all.
- Every new module: a unit test file covering at minimum an empty
  input, an oversized input, and a wrong-type input, plus the actual
  logic.

## Adding to the benchmark corpus

Read [`benchmark/README.md`](./benchmark/README.md) first — it documents
a real repetition mistake made in an earlier pass (the same content
translated across all four dialects) and the tool that now catches it.
Before opening a PR that adds corpus sentences:

```bash
python benchmark/check_corpus_quality.py
```

New sentences should be original (not copied from a copyrighted
source), on a topic not already covered by an existing entry in that
category/region, and pass the quality check above with no new flagged
pairs.

## Pull requests

- Run `pytest` and `check_corpus_quality.py` (if you touched a corpus
  file) before opening a PR.
- Describe what changed and why in the PR description — this project's
  commit history favors explaining the reasoning behind a change, not
  just what changed.
- If you're fixing a bug, a regression test that reproduces it (and
  would fail without your fix) is expected, not optional.