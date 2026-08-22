"""A convenience wrapper combining normalize() and check_content_warnings().

DESIGN CONSTRAINT, stated up front: safety_modes.py's module docstring
says "Never call check_content_warnings() automatically as part of a
normalization pipeline" -- this module does not violate that. This is a
function the CALLER explicitly opts into calling (exactly like calling
normalize() and check_content_warnings() separately would be), not
something wired into normalize() itself, which remains untouched and
still never calls check_content_warnings() internally on its own.

What this function does NOT do, on purpose:
- It does NOT silently skip normalization for you. It runs the check,
  and if any warning fires, it returns the ORIGINAL (unnormalized) text
  alongside the warnings -- but the decision to actually use the
  original vs. the normalized text, or to proceed at all, is yours. See
  SmartPrepareResult.text's docstring for the exact rule.
- It does NOT suppress repeated warnings across calls. Same as
  check_content_warnings() itself: stateless, by design (see
  safety_modes.py's docstring for why repeat-suppression belongs in the
  caller's session logic, not here).
- It does NOT use dialect detection to decide anything. An earlier
  version of this idea considered feeding detect_dialect() into the
  decision, but dialect signal (a ~50% F1 ceiling heuristic) has no
  actual bearing on whether content is religious/legal/medical --
  conflating them would add complexity without adding safety.
"""

from __future__ import annotations

from dataclasses import dataclass

from ar_tokenwise.normalize import DEFAULT_MAX_LENGTH, NormalizationLevel, normalize
from ar_tokenwise.safety_modes import ContentWarning, check_content_warnings


@dataclass(frozen=True)
class SmartPrepareResult:
    """Result of a smart_prepare() call.

    Attributes:
        text: The normalized text, UNLESS at least one warning fired, in
            which case this is the original, unmodified text -- so a
            caller who ignores ``warnings`` entirely and just uses
            ``.text`` gets the conservative behavior (no normalization
            applied to text this function flagged as possibly
            sensitive) rather than the risky one. If you have reviewed
            the warnings and decided normalization is still appropriate
            for your specific text, call ``normalize()`` yourself.
        warnings: Output of :func:`check_content_warnings` on the
            original text -- empty list means no category was flagged
            (see that function's docstring: this is not a safety
            guarantee, just an absence of signal).
        was_normalized: True if ``text`` is the normalized version,
            False if it was left as the original because a warning fired.
    """

    text: str
    warnings: list[ContentWarning]
    was_normalized: bool


def smart_prepare(
    text: str,
    level: NormalizationLevel = NormalizationLevel.LIGHT,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> SmartPrepareResult:
    """Check text for content warnings, then normalize only if none fired.

    This is a convenience wrapper, not a different safety model: calling
    ``check_content_warnings(text)`` and ``normalize(text)`` yourself
    and making this same decision manually is equivalent. Use this when
    you want that specific policy (skip normalization when any warning
    fires) without writing the branch yourself; write the branch
    yourself if you want a different policy (e.g. still normalize at
    LIGHT level but log the warning, or always normalize regardless).

    Args:
        text: Input text.
        level: Normalization level to apply when no warning fires.
        max_length: Maximum accepted input length in characters, passed
            through to both underlying functions.

    Returns:
        A :class:`SmartPrepareResult`.

    Raises:
        TypeError: if ``text`` is not a string.
        ValueError: if ``text`` exceeds ``max_length``.
    """
    warnings = check_content_warnings(text, max_length=max_length)

    if warnings:
        return SmartPrepareResult(text=text, warnings=warnings, was_normalized=False)

    normalized = normalize(text, level=level, max_length=max_length)
    return SmartPrepareResult(text=normalized, warnings=[], was_normalized=True)