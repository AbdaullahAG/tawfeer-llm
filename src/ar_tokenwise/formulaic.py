"""Formulaic Arabic expression compression -- NOT YET IMPLEMENTED.

STATUS: deliberately unbuilt, blocked on real usage data. This is not a
forgotten feature -- see README.md's "Roadmap" section for the same
status, and the rationale below for why building it now would be
irresponsible.

INTENDED DESIGN (for whoever picks this up next):
- compress_formulaic_expressions(text) -> (compressed_text, mapping)
    Replace long, common, meaning-preserving formulaic phrases (formal
    letter openings/closings, contract boilerplate, standard greeting
    formulas) with a short placeholder token. Return the compressed
    text plus a mapping to restore the originals.
- decompress_formulaic_expressions(text, mapping) -> original_text
    The inverse operation, applied to the model's response if the
    placeholders need to be expanded back for the end user.

WHY NOT BUILT YET: the marker dictionary needs real frequency data --
which formulaic phrases actually recur often enough in production
Arabic LLM traffic to be worth substituting, and confirmation that each
substitution is safe to never alter meaning in the contexts it actually
appears in. Building this list from general knowledge of "phrases that
sound formulaic" would be exactly the kind of unverified claim this
project avoids everywhere else -- normalize.py's conservative levels,
dialect.py's explicit accuracy ceiling, safety_modes.py's stateless
design, and every benchmark number in this repo are all grounded in
something measured, not assumed. This module should be no different.

WHEN TO BUILD: once the project has real production usage (or a large
real corpus of Arabic LLM prompts, with permission to analyze it) to
mine for recurring formulaic phrases and measure actual token savings
and false-substitution risk before shipping a marker list.

Every function in this module currently raises NotImplementedError.
This module is intentionally NOT imported into ar_tokenwise/__init__.py
-- exporting a non-functional symbol in the public API would be worse
than not having it at all.
"""

from __future__ import annotations


def compress_formulaic_expressions(text: str) -> tuple[str, dict[str, str]]:
    """NOT YET IMPLEMENTED. See module docstring for why and what's needed.

    Args:
        text: Input text (unused -- always raises).

    Raises:
        NotImplementedError: always, until real usage data is available.
    """
    raise NotImplementedError(
        "compress_formulaic_expressions() is not yet implemented -- it is "
        "blocked on real Arabic LLM usage data, not forgotten. See this "
        "module's docstring and the project README's Roadmap section for "
        "why, and what's needed before it can be built responsibly."
    )


def decompress_formulaic_expressions(text: str, mapping: dict[str, str]) -> str:
    """NOT YET IMPLEMENTED. See module docstring for why and what's needed.

    Args:
        text: Compressed text (unused -- always raises).
        mapping: Placeholder-to-original mapping (unused -- always raises).

    Raises:
        NotImplementedError: always, until real usage data is available.
    """
    raise NotImplementedError(
        "decompress_formulaic_expressions() is not yet implemented -- it is "
        "blocked on real Arabic LLM usage data, not forgotten. See this "
        "module's docstring and the project README's Roadmap section for "
        "why, and what's needed before it can be built responsibly."
    )