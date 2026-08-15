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