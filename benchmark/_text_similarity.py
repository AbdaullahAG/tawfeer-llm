"""Token-overlap F1 scoring between a generated answer and a gold answer.

Standard SQuAD-style metric: normalize both strings (strip punctuation
and collapse whitespace), split into word tokens, then compute
precision/recall/F1 on the token multiset overlap. This is the
established metric for extractive QA evaluation -- NOT an LLM-as-judge
score, so none of the position/self-preference/verbosity biases
documented for LLM judges apply here (see
benchmark/run_comprehension_eval.py's module docstring for why that
matters for this project's specific comparison).

Not part of the installed package -- shared by benchmark eval scripts.
"""

from __future__ import annotations

import re
from collections import Counter

from ar_tokenwise.normalize import NormalizationLevel
from ar_tokenwise.normalize import normalize as ar_normalize

# Matches punctuation only (diacritics/tatweel are already handled by
# ar_tokenwise.normalize() at LIGHT level, reused below rather than
# reimplemented here). Character class with a trailing +, no nesting --
# linear in input length, no ReDoS risk.
_PUNCTUATION_PATTERN = re.compile(r"[.,!?;:\"'\u060C\u061B\u061F()\[\]{}\-]+")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_answer_for_comparison(text: str) -> str:
    """Lightly normalize an answer string for fair comparison.

    This is deliberately separate from calling ar_tokenwise.normalize()
    directly on its own -- that function optimizes text for sending to
    a model; this one exists purely to make two answer strings
    comparable. It reuses normalize() at LIGHT level for
    diacritic/tatweel stripping (already tested there, not
    reimplemented here), then additionally strips punctuation, which
    normalize() does not touch since punctuation isn't a token-cost
    concern for that function's purpose.
    """
    diacritics_and_tatweel_stripped = ar_normalize(text, level=NormalizationLevel.LIGHT)
    punctuation_stripped = _PUNCTUATION_PATTERN.sub(" ", diacritics_and_tatweel_stripped)
    collapsed = _WHITESPACE_PATTERN.sub(" ", punctuation_stripped).strip()
    return collapsed


def f1_score(predicted: str, gold: str) -> float:
    """Token-overlap F1 between a predicted answer and the gold answer.

    Args:
        predicted: The model-generated answer text.
        gold: The known-correct answer text.

    Returns:
        F1 in [0.0, 1.0]. Both empty (after normalization) -> 1.0
        (exact match on "no answer"). Exactly one empty -> 0.0.
    """
    predicted_tokens = normalize_answer_for_comparison(predicted).split()
    gold_tokens = normalize_answer_for_comparison(gold).split()

    if not predicted_tokens and not gold_tokens:
        return 1.0
    if not predicted_tokens or not gold_tokens:
        return 0.0

    predicted_counts = Counter(predicted_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum((predicted_counts & gold_counts).values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(predicted_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(predicted: str, gold: str) -> bool:
    """Whether predicted and gold are identical after light normalization."""
    return normalize_answer_for_comparison(predicted) == normalize_answer_for_comparison(gold)