"""Unit tests for integrations/litellm_plugin.py.

This module lives outside the installed package (integrations/, not
src/), so it's loaded dynamically via importlib -- same pattern as
tests/test_skill_scripts.py for skill/scripts/.
"""

import importlib.util
from pathlib import Path

import pytest

INTEGRATIONS_DIR = Path(__file__).parent.parent / "integrations"


def _load_module(filename: str):
    """Dynamically load a standalone module as an importable module."""
    module_path = INTEGRATIONS_DIR / filename
    spec = importlib.util.spec_from_file_location(filename, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


litellm_plugin = _load_module("litellm_plugin.py")


# --- normalize_messages(): pure function, no litellm required -----------


def test_normalize_messages_normalizes_string_content() -> None:
    messages = [{"role": "user", "content": "مَرْحَـبًا بكم"}]
    result = litellm_plugin.normalize_messages(messages)
    assert result[0]["content"] == "مرحبا بكم"


def test_normalize_messages_does_not_mutate_input() -> None:
    original = [{"role": "user", "content": "مَرْحَـبًا"}]
    litellm_plugin.normalize_messages(original)
    assert original[0]["content"] == "مَرْحَـبًا"  # unchanged


def test_normalize_messages_passes_through_non_string_content() -> None:
    # Vision-model style multi-part content -- must not crash or be altered.
    messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    result = litellm_plugin.normalize_messages(messages)
    assert result[0]["content"] == [{"type": "text", "text": "hi"}]


def test_normalize_messages_preserves_other_fields() -> None:
    messages = [{"role": "system", "content": "مَرْحَـبًا", "name": "bot"}]
    result = litellm_plugin.normalize_messages(messages)
    assert result[0]["role"] == "system"
    assert result[0]["name"] == "bot"


def test_normalize_messages_empty_list_returns_empty_list() -> None:
    assert litellm_plugin.normalize_messages([]) == []


def test_normalize_messages_non_list_input_raises_type_error() -> None:
    with pytest.raises(TypeError):
        litellm_plugin.normalize_messages("not a list")  # type: ignore[arg-type]


def test_normalize_messages_non_dict_element_raises_type_error() -> None:
    with pytest.raises(TypeError):
        litellm_plugin.normalize_messages(["not a dict"])  # type: ignore[list-item]


def test_normalize_messages_multiple_messages_all_normalized() -> None:
    messages = [
        {"role": "system", "content": "أَنْتَ مُساعِد"},
        {"role": "user", "content": "مَرْحَـبًا"},
    ]
    result = litellm_plugin.normalize_messages(messages)
    assert result[0]["content"] == "أنت مساعد"
    assert result[1]["content"] == "مرحبا"


# --- ArabicNormalizerLogger: behavior depends on whether litellm is installed --


def test_logger_raises_clear_error_without_litellm(monkeypatch: pytest.MonkeyPatch) -> None:
    if litellm_plugin._LITELLM_AVAILABLE:
        pytest.skip("litellm is installed in this environment; covered by the hook test instead")
    with pytest.raises(ImportError, match="ar-tokenwise\\[integrations\\]"):
        litellm_plugin.ArabicNormalizerLogger()


@pytest.mark.skipif(
    not litellm_plugin._LITELLM_AVAILABLE, reason="litellm not installed in this environment"
)
@pytest.mark.anyio
async def test_logger_hook_normalizes_messages_in_place() -> None:
    logger = litellm_plugin.ArabicNormalizerLogger()
    data = {"messages": [{"role": "user", "content": "مَرْحَـبًا بكم"}]}
    result = await logger.async_pre_call_hook(None, None, data, "completion")
    assert result["messages"][0]["content"] == "مرحبا بكم"


@pytest.mark.skipif(
    not litellm_plugin._LITELLM_AVAILABLE, reason="litellm not installed in this environment"
)
@pytest.mark.anyio
async def test_logger_hook_ignores_data_without_messages() -> None:
    logger = litellm_plugin.ArabicNormalizerLogger()
    data = {"some_other_key": "value"}
    result = await logger.async_pre_call_hook(None, None, data, "completion")
    assert result == {"some_other_key": "value"}