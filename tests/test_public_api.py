# tests/test_public_api.py
"""Ensures the public API surface stays stable and matches __all__."""

import ar_tokenwise


def test_public_api_matches_all() -> None:
    for name in ar_tokenwise.__all__:
        assert hasattr(ar_tokenwise, name), f"{name} listed in __all__ but missing"


def test_normalize_importable_from_top_level() -> None:
    result = ar_tokenwise.normalize("مَرْحَبًا")
    assert result == "مرحبا"


def test_normalization_level_importable_from_top_level() -> None:
    assert ar_tokenwise.NormalizationLevel.MEDIUM.value == "medium"


def test_version_is_string() -> None:
    assert isinstance(ar_tokenwise.__version__, str)


def test_all_v2_modules_are_importable_from_top_level() -> None:
    """Explicit named-import check for every v2 module's main entry point.

    Unlike test_public_api_matches_all (which only checks internal
    consistency of __all__), this test would fail if a module's exports
    were accidentally left out of __init__.py entirely -- __all__ being
    self-consistent is not enough if it's simply missing an entry.
    """
    from ar_tokenwise import (
        check_content_warnings,
        chunk_text,
        detect_dialect,
        report_mixed_fertility,
    )

    assert callable(chunk_text)
    assert callable(report_mixed_fertility)
    assert callable(detect_dialect)
    assert callable(check_content_warnings)