"""LiteLLM pre-call hook: normalize Arabic text before it reaches the model.

Optional integration -- requires `litellm` installed separately (not a
hard dependency of ar-tokenwise). Install with:
    pip install ar-tokenwise[integrations]

PERFORMANCE CONSTRAINT: async_pre_call_hook runs on every request's hot
path. normalize_messages() below only calls normalize() (pure regex/
lookup, no I/O), so it's safe there. Deliberately NOT included: any
token-counting or reporting (get_default_counter() does network I/O on
first use to fetch tiktoken's encoding file) -- that kind of work must
be cached/precomputed or done outside the hot path, never per-request
here.

This module is split into two parts:
1. normalize_messages() -- pure function, no litellm dependency needed,
   fully testable in isolation.
2. ArabicNormalizerLogger -- a thin LiteLLM CustomLogger wrapper around
   part 1, only usable if litellm is installed.
"""

from __future__ import annotations

from typing import Any

from ar_tokenwise.normalize import NormalizationLevel, normalize

try:
    from litellm.integrations.custom_logger import CustomLogger

    _LITELLM_AVAILABLE = True
except ImportError:
    CustomLogger = object  # fallback base so class definition below doesn't crash
    _LITELLM_AVAILABLE = False


def normalize_messages(
    messages: list[dict[str, Any]],
    level: NormalizationLevel = NormalizationLevel.LIGHT,
) -> list[dict[str, Any]]:
    """Return a new messages list with Arabic text normalized per message.

    Pure and lightweight (regex/lookup only, no I/O, no litellm
    dependency) -- safe to call on every request in a hot-path hook.
    Does not mutate the input list or its dicts.

    Only string "content" fields are normalized. Messages with
    non-string content (e.g. multi-part content lists used by vision
    models) are passed through unchanged, since normalize() only
    accepts plain strings.

    Args:
        messages: A list of message dicts, e.g. LiteLLM/OpenAI-style
            ``[{"role": "user", "content": "..."}, ...]``.
        level: Normalization aggressiveness applied to each message.

    Returns:
        A new list of message dicts (shallow-copied), safe to assign
        back to the caller without aliasing the original list/dicts.

    Raises:
        TypeError: if ``messages`` is not a list, or any element is not
            a dict.
    """
    if not isinstance(messages, list):
        raise TypeError(f"normalize_messages() expects list, got {type(messages).__name__}")

    result: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise TypeError(
                f"each message must be a dict, got {type(message).__name__}"
            )
        content = message.get("content")
        if isinstance(content, str):
            new_message = dict(message)
            new_message["content"] = normalize(content, level=level)
            result.append(new_message)
        else:
            result.append(message)
    return result


class ArabicNormalizerLogger(CustomLogger):  # type: ignore[misc]
    """LiteLLM pre-call hook that normalizes Arabic text in request messages.

    Usage (LiteLLM Python SDK):
        import litellm
        from integrations.litellm_plugin import ArabicNormalizerLogger
        litellm.callbacks = [ArabicNormalizerLogger()]

    Usage (LiteLLM Proxy, in your callback config file):
        proxy_handler_instance = ArabicNormalizerLogger()
        # then reference it in litellm_settings.callbacks per LiteLLM's
        # proxy callback docs.

    Requires litellm installed: pip install ar-tokenwise[integrations]
    """

    def __init__(self, level: NormalizationLevel = NormalizationLevel.LIGHT) -> None:
        if not _LITELLM_AVAILABLE:
            raise ImportError(
                "ArabicNormalizerLogger requires litellm. Install it with "
                "`pip install ar-tokenwise[integrations]`, or use "
                "normalize_messages() directly without litellm."
            )
        super().__init__()
        self.level = level

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict[str, Any],
        call_type: str,
    ) -> dict[str, Any]:
        """Normalize Arabic text in `data["messages"]` before the model call."""
        if "messages" in data:
            data["messages"] = normalize_messages(data["messages"], level=self.level)
        return data