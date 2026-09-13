"""Reference-backed preservation of Quranic passages during normalization.

This is deliberately a verifier, not a religious-keyword heuristic.  The
bundled corpus is loaded once per process and matching is done against its
normalized word sequences.  A passage is preserved exactly as the caller
provided it; the corpus is never used to rewrite user text.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable

from ar_tokenwise.normalize import _TASHKEEL_PATTERN, _TATWEEL_PATTERN

# ``\w`` deliberately excludes combining marks in Python.  Arabic harakat
# are combining marks, so using a generic word regex would split بِسْمِ into
# three tokens before ``quran_match_key`` had a chance to remove them.
_ARABIC_WORD_PATTERN = re.compile(
    r"[\u0621-\u063A\u0641-\u064A\u066E-\u066F\u0671]"
    r"[\u0610-\u061A\u0621-\u063A\u0640-\u065F\u066E-\u066F\u0670\u0671\u06D6-\u06ED]*"
)
_QURANIC_MARKS = re.compile(r"[\u0610-\u061A\u0653-\u065F\u06D6-\u06ED]")


def quran_match_key(text: str) -> str:
    """Return the comparison key used for Quran reference matching.

    It deliberately mirrors LIGHT normalization for tashkeel and tatweel,
    then removes Quranic annotation marks and treats punctuation as a word
    separator.  It does *not* fold meaning-bearing letters such as ``ة`` or
    ``ى``.
    """
    text = unicodedata.normalize("NFC", text).lstrip("\ufeff")
    text = _TATWEEL_PATTERN.sub("", text)
    text = _TASHKEEL_PATTERN.sub("", text)
    text = _QURANIC_MARKS.sub("", text)
    text = text.replace("ٱ", "ا")
    return " ".join(_ARABIC_WORD_PATTERN.findall(text))


@lru_cache(maxsize=1)
def load_quran_verses() -> tuple[str, ...]:
    """Load the bundled Quran reference corpus once per Python process.

    The file contains verbatim Arabic verse strings sourced from Quran JSON;
    see ``data/QURAN_DATA_NOTICE.md`` for attribution and licence details.
    """
    data_path = Path(__file__).with_name("data") / "quran_verses.json"
    with data_path.open(encoding="utf-8") as source:
        verses = json.load(source)
    if not isinstance(verses, list) or not all(isinstance(v, str) for v in verses):
        raise RuntimeError("Bundled Quran reference corpus is invalid.")
    return tuple(verses)


@dataclass(frozen=True)
class QuranSpan:
    """A character range in caller input confirmed against the corpus."""

    start: int
    end: int


class QuranGuard:
    """Find Quran passages and preserve them while another function normalizes.

    Four words are the minimum standalone proof.  A two- or three-word
    sequence is only retained when it extends a neighboring confirmed
    four-word sequence in the *same* reference verse.  This avoids treating
    short, common Arabic phrases as Quranic out of context.
    """

    _STANDALONE_MIN_WORDS = 4

    def __init__(self, verses: Iterable[str] | None = None) -> None:
        raw_verses = tuple(load_quran_verses() if verses is None else verses)
        # Consecutive ayat are one reference stream.  This deliberately lets
        # a confirmed quote continue across an ayah boundary, while still
        # requiring every word to be present in its canonical order.
        self._reference_words = tuple(
            word
            for verse in raw_verses
            for word in quran_match_key(verse).split()
        )
        self._four_word_index: dict[tuple[str, ...], list[int]] = {}
        for offset in range(len(self._reference_words) - self._STANDALONE_MIN_WORDS + 1):
            key = self._reference_words[offset : offset + self._STANDALONE_MIN_WORDS]
            self._four_word_index.setdefault(key, []).append(offset)

    @staticmethod
    def _tokens(text: str) -> tuple[list[str], list[tuple[int, int]]]:
        words: list[str] = []
        spans: list[tuple[int, int]] = []
        for match in _ARABIC_WORD_PATTERN.finditer(text):
            key = quran_match_key(match.group())
            if key:
                words.append(key)
                spans.append(match.span())
        return words, spans

    def find_quran_spans(self, text: str) -> list[QuranSpan]:
        """Return non-overlapping, reference-confirmed spans in ``text``."""
        if not isinstance(text, str):
            raise TypeError(f"find_quran_spans() expects str, got {type(text).__name__}")

        words, source_spans = self._tokens(text)
        candidates: list[tuple[int, int]] = []
        n = len(words)
        for start in range(n - self._STANDALONE_MIN_WORDS + 1):
            seed = tuple(words[start : start + self._STANDALONE_MIN_WORDS])
            locations = self._four_word_index.get(seed, ())
            best_end = start + self._STANDALONE_MIN_WORDS
            for reference_offset in locations:
                end = best_end
                while (
                    end < n
                    and reference_offset + (end - start) < len(self._reference_words)
                    and words[end]
                    == self._reference_words[reference_offset + (end - start)]
                ):
                    end += 1
                best_end = max(best_end, end)
            candidates.append((start, best_end))

        # Greedily take the longest confirmed quote at each position.  Its
        # rightward extension is what permits adjacent 2-3-word fragments;
        # isolated short matches never enter ``candidates``.
        selected: list[tuple[int, int]] = []
        cursor = 0
        while cursor < n:
            possible = [end for start, end in candidates if start == cursor]
            if possible:
                end = max(possible)
                selected.append((cursor, end))
                cursor = end
            else:
                cursor += 1

        return [QuranSpan(source_spans[start][0], source_spans[end - 1][1]) for start, end in selected]

    def protect_and_normalize(self, text: str, normalize_fn: Callable[[str], str]) -> str:
        """Normalize non-Quran text while copying confirmed spans verbatim."""
        spans = self.find_quran_spans(text)
        if not spans:
            return normalize_fn(text)

        parts: list[str] = []
        cursor = 0
        for span in spans:
            parts.append(normalize_fn(text[cursor : span.start]))
            parts.append(text[span.start : span.end])
            cursor = span.end
        parts.append(normalize_fn(text[cursor:]))
        return "".join(parts)


_DEFAULT_GUARD: QuranGuard | None = None


def protect_quranic_text(text: str, normalize_fn: Callable[[str], str]) -> str:
    """Convenience wrapper using the process-wide Quran reference index."""
    global _DEFAULT_GUARD
    if _DEFAULT_GUARD is None:
        _DEFAULT_GUARD = QuranGuard()
    return _DEFAULT_GUARD.protect_and_normalize(text, normalize_fn)
