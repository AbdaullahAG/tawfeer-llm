# tests/test_cache_keys.py
"""Unit tests for ar_tokenwise.cache_keys."""

import pytest

from ar_tokenwise.cache_keys import canonicalize_for_cache_key, generate_cache_key


# --- required minimum cases: empty / oversized / wrong type -----------


def test_generate_cache_key_empty_text_is_deterministic() -> None:
    key1 = generate_cache_key("")
    key2 = generate_cache_key("")
    assert key1 == key2
    assert len(key1) == 64  # SHA-256 hex digest length


def test_generate_cache_key_oversized_input_raises_value_error() -> None:
    with pytest.raises(ValueError, match="exceeds max_length"):
        generate_cache_key("كلمة " * 10, max_length=5)


def test_generate_cache_key_non_str_input_raises_type_error() -> None:
    with pytest.raises(TypeError):
        generate_cache_key(123)  # type: ignore[arg-type]


def test_canonicalize_oversized_input_raises_value_error() -> None:
    with pytest.raises(ValueError, match="exceeds max_length"):
        canonicalize_for_cache_key("كلمة " * 10, max_length=5)


def test_canonicalize_non_str_input_raises_type_error() -> None:
    with pytest.raises(TypeError):
        canonicalize_for_cache_key(123)  # type: ignore[arg-type]


# --- core purpose: semantically-equivalent text maps to the same key ----


def test_diacritized_and_undiacritized_text_produce_same_key() -> None:
    diacritized = "مَرْحَـبًا بِكُم"
    plain = "مرحبا بكم"
    assert generate_cache_key(diacritized) == generate_cache_key(plain)


def test_alef_variants_produce_same_key() -> None:
    assert generate_cache_key("أحمد") == generate_cache_key("إحمد")
    assert generate_cache_key("أحمد") == generate_cache_key("احمد")


def test_extra_whitespace_produces_same_key() -> None:
    assert generate_cache_key("مرحبا   بكم") == generate_cache_key("مرحبا بكم")
    assert generate_cache_key("  مرحبا بكم  ") == generate_cache_key("مرحبا بكم")


def test_latin_case_does_not_change_key() -> None:
    assert generate_cache_key("Hello World") == generate_cache_key("hello world")


def test_different_content_produces_different_keys() -> None:
    assert generate_cache_key("مرحبا") != generate_cache_key("وداعا")


# --- canonicalize_for_cache_key: direct behavior --------------------------


def test_canonicalize_never_returns_diacritics() -> None:
    result = canonicalize_for_cache_key("مَرْحَـبًا")
    assert result == "مرحبا"


def test_canonicalize_is_idempotent() -> None:
    text = "مَرْحَـبًا   بِكُم Hello"
    once = canonicalize_for_cache_key(text)
    twice = canonicalize_for_cache_key(once)
    assert once == twice


def test_canonicalize_empty_text_returns_empty_string() -> None:
    assert canonicalize_for_cache_key("") == ""


# --- key format sanity ----------------------------------------------------


def test_generate_cache_key_is_lowercase_hex() -> None:
    key = generate_cache_key("نص عربي عادي")
    assert all(c in "0123456789abcdef" for c in key)