This file is for anyone building or extending an integration in this
directory, not end users -- for "how do I use an integration", see the
root [README.md](../README.md)'s Integrations table instead.

## The pattern every file here follows

Each integration file is split into two parts, and any new integration
should follow the same split:

1. **A pure function** operating on plain data (strings/dicts/lists) --
   no dependency on the target framework, fully testable without it
   installed. E.g. `normalize_messages()`, `normalize_page_contents()`,
   `normalize_node_texts()`.
2. **A thin wrapper class** around part 1, implementing whatever
   interface the target framework requires. Only this part needs the
   framework installed; it raises a clear `ImportError` in `__init__`
   (pointing to `pip install ar-tokenwise[integrations]`) if the
   framework isn't available.

This keeps the core library's "few dependencies" principle intact for
anyone who doesn't use a given framework, and keeps the actual
normalization logic testable in CI without installing every framework.

## The three target frameworks are NOT structurally similar -- don't assume

Each framework's extension point turned out to have a different base
class shape. This was discovered by actually installing each library
and testing against it, not by reading docs alone -- do the same for
any new integration rather than assuming symmetry with an existing one.

| Framework | Base class | Shape |
|---|---|---|
| LiteLLM | `litellm.integrations.custom_logger.CustomLogger` | Plain class. `async_pre_call_hook(self, user_api_key_dict, cache, data, call_type)`. |
| LangChain | `langchain_core.documents.BaseDocumentTransformer` | Plain ABC. Only `transform_documents` is actually abstract (verified via `.__abstractmethods__`); `atransform_documents` is implemented too here for completeness. |
| LlamaIndex | `llama_index.core.schema.TransformComponent` | **Pydantic `BaseModel`** (verified via its `__mro__`), not a plain class. Config-style attributes (like `level`) must be declared as class-level annotated fields, not passed as ad-hoc `__init__` parameters. `__call__(self, nodes, **kwargs)` is the sync entry point; `acall` is the async one. |

## A real bug hit here, worth knowing about before it happens again

`integrations/llamaindex_wrapper.py` is the only file in this project
that does NOT start with `from __future__ import annotations`, unlike
every other module. This is deliberate, not an oversight: with deferred
annotations enabled, Pydantic v2 failed to resolve `TransformComponent`
subclass field types at class-definition time and raised
`PydanticUserError: ArabicNodeNormalizer is not fully defined`. This was
caught by actually running the test suite against a real installed
`llama-index-core`, not by code review. If you add a new
Pydantic-BaseModel-based integration, watch for the same failure mode.

## Testing without installing every framework

`tests/test_litellm_plugin.py`, `tests/test_langchain_wrapper.py`, and
`tests/test_llamaindex_wrapper.py` all:

- Load the integration file dynamically via `importlib` (these files
  live outside `src/`, so they're not part of the installed package and
  can't be imported the normal way).
- Test the pure function unconditionally (no framework needed).
- Test the wrapper class's `ImportError` path when the framework is
  NOT installed, and `pytest.mark.skipif(_XYZ_AVAILABLE, ...)` the
  real-integration tests so they only run when it IS installed.

Every wrapper class's tests in this project were verified against a
real installed copy of the target framework at least once during
development (not just the not-installed/ImportError path) -- do the
same for any new integration before considering it done.

## Adding a new integration

1. Install the target framework in a scratch environment and inspect
   its actual extension-point base class the same way this file's table
   was built (`__mro__`, `.__abstractmethods__`, actual construction) --
   don't assume it matches an existing entry here.
2. Write the pure function first, test it with no framework installed.
3. Write the thin wrapper class, following the ImportError-in-`__init__`
   pattern.
4. Add the framework to `pyproject.toml`'s `integrations` extra.
5. Add the framework to the table in the root README.md's Integrations
   section.
6. Update this file's table and "hit bugs" section if you find another
   framework-specific gotcha.
MDEOF
cd /home/claude/tawfeer-llm/tawfeer-llm && . .venv/bin/activate && python -m pytest -q 2>&1 | tail -6
git add integrations/README.md && git commit -q -m "docs: add integrations/ maintainer notes (base class differences, the pydantic bug hit, testing pattern)" && git log --oneline | head -3
Output


.venv/lib/python3.12/site-packages/requests/models.py:1167: HTTPError
=========================== short test summary info ============================
FAILED tests/test_report.py::test_get_default_counter_works_when_tiktoken_installed
FAILED tests/test_skill_scripts.py::test_report_cli_prints_valid_json - reque...
2 failed, 136 passed, 9 skipped in 0.50s
4a1227d docs: add integrations/ maintainer notes (base class differences, the pydantic bug hit, testing pattern)
bc24561 docs: add PyPI publishing checklist (not yet executed, business decision pending)
094143e docs: add explicit formulaic.py stub (blocked on real usage data, not forgotten)