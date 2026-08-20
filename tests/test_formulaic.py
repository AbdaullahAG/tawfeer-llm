"""Tests confirming ar_tokenwise.formulaic is an explicit, documented stub.

These are "tripwire" tests: they exist to make the stub status visible
in CI, not to test real functionality. If someone implements this module
for real, these tests SHOULD start failing -- that failure is the
correct signal to update/remove them as part of that implementation
work, not a bug to silently work around.
"""

import pytest

from ar_tokenwise.formulaic import (
    compress_formulaic_expressions,
    decompress_formulaic_expressions,
)


def test_compress_is_not_yet_implemented() -> None:
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        compress_formulaic_expressions("أي نص")


def test_decompress_is_not_yet_implemented() -> None:
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        decompress_formulaic_expressions("أي نص", {})


def test_formulaic_module_is_not_exported_from_public_api() -> None:
    """The stub must not leak into the public API surface (see module docstring)."""
    import ar_tokenwise

    assert "compress_formulaic_expressions" not in ar_tokenwise.__all__
    assert "decompress_formulaic_expressions" not in ar_tokenwise.__all__