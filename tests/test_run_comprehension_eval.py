"""Unit tests for benchmark/run_comprehension_eval.py's non-network logic.

The real API-calling answerers (_build_anthropic_answerer,
_build_gemini_answerer) are NOT tested here -- they need real
credentials and make real network calls. What IS tested: evaluate_examples()
against a fake, deterministic answerer, and summarize()'s reporting logic.
"""

import importlib.util
import sys
from pathlib import Path

BENCHMARK_DIR = Path(__file__).parent.parent / "benchmark"


def _load_module(filename: str):
    module_path = BENCHMARK_DIR / filename
    spec = importlib.util.spec_from_file_location(filename, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# run_comprehension_eval.py imports `from _text_similarity import f1_score`
# and `from _tydiqa_loader import ...` as top-level sibling imports (works
# when run as `python benchmark/run_comprehension_eval.py`, since Python
# puts the script's own directory on sys.path). For dynamic loading here,
# make sure benchmark/ is importable the same way.
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

rce = _load_module("run_comprehension_eval.py")
tydiqa_loader = _load_module("_tydiqa_loader.py")


def _fake_answerer_always_correct(passage: str, question: str) -> str:
    """Ignores input, but evaluate_examples() calls it with (passage, question)."""
    return "___UNUSED___"


def test_evaluate_examples_perfect_answerer_gives_f1_one() -> None:
    examples = [
        tydiqa_loader.GoldPExample(
            example_id="1", passage="نص تجريبي", question="ما هذا؟", answer_text="القاهرة"
        )
    ]
    answerer = lambda passage, question: "القاهرة"  # noqa: E731
    f1_original, f1_normalized = rce.evaluate_examples(
        examples, answerer, rce.NormalizationLevel.LIGHT
    )
    assert f1_original == [1.0]
    assert f1_normalized == [1.0]


def test_evaluate_examples_wrong_answerer_gives_f1_zero() -> None:
    examples = [
        tydiqa_loader.GoldPExample(
            example_id="1", passage="نص تجريبي", question="ما هذا؟", answer_text="القاهرة"
        )
    ]
    answerer = lambda passage, question: "دمشق"  # noqa: E731
    f1_original, f1_normalized = rce.evaluate_examples(
        examples, answerer, rce.NormalizationLevel.LIGHT
    )
    assert f1_original == [0.0]
    assert f1_normalized == [0.0]


def test_evaluate_examples_passes_normalized_passage_to_answerer() -> None:
    received_passages = []

    def _recording_answerer(passage: str, question: str) -> str:
        received_passages.append(passage)
        return "أي شي"

    examples = [
        tydiqa_loader.GoldPExample(
            example_id="1", passage="مَرْحَـبًا بكم", question="؟", answer_text="أي شي"
        )
    ]
    rce.evaluate_examples(examples, _recording_answerer, rce.NormalizationLevel.LIGHT)

    # First call: original passage (with diacritics). Second call: normalized.
    assert received_passages[0] == "مَرْحَـبًا بكم"
    assert received_passages[1] == "مرحبا بكم"


def test_evaluate_examples_empty_list_returns_empty_lists() -> None:
    f1_original, f1_normalized = rce.evaluate_examples(
        [], _fake_answerer_always_correct, rce.NormalizationLevel.LIGHT
    )
    assert f1_original == []
    assert f1_normalized == []


def test_summarize_empty_input() -> None:
    assert rce.summarize([], []) == "No examples evaluated."


def test_summarize_reports_negative_delta_as_real_harm() -> None:
    summary = rce.summarize([1.0, 1.0], [0.5, 0.5])
    assert "HURT comprehension" in summary
    assert "-0.500" in summary


def test_summarize_reports_no_material_difference() -> None:
    summary = rce.summarize([0.8, 0.8], [0.8, 0.8])
    assert "No material difference" in summary


def test_summarize_includes_mean_and_median() -> None:
    summary = rce.summarize([0.5, 1.0], [0.5, 1.0])
    assert "0.750" in summary  # mean of 0.5 and 1.0