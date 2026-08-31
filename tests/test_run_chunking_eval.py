"""Unit tests for benchmark/run_chunking_eval.py's non-network logic.

The real Gemini embedding calls (_build_gemini_embedder) are NOT tested
here -- they need real credentials and make real network calls. What IS
tested: cosine_similarity, naive_fixed_chunk, retrieve_top_k_indices,
is_answer_retrieved, proportion_ci95, summarize, and evaluate_retrieval
against a fake, deterministic embedder.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

BENCHMARK_DIR = Path(__file__).parent.parent / "benchmark"


def _load_module(filename: str):
    module_path = BENCHMARK_DIR / filename
    spec = importlib.util.spec_from_file_location(filename, module_path)
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec_module -- see
    # tests/test_tydiqa_loader.py's comment for why this matters
    # (dataclasses + deferred annotations need it to resolve types).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

rche = _load_module("run_chunking_eval.py")
tydiqa_loader = _load_module("_tydiqa_loader.py")


def _word_count_counter(text: str) -> int:
    return len(text.split())


# --- cosine_similarity ------------------------------------------------


def test_cosine_similarity_identical_vectors_is_one() -> None:
    assert rche.cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero() -> None:
    assert rche.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors_is_negative_one() -> None:
    assert rche.cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_returns_zero_not_error() -> None:
    assert rche.cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


# --- naive_fixed_chunk --------------------------------------------------


def test_naive_fixed_chunk_empty_text_returns_empty_list() -> None:
    assert rche.naive_fixed_chunk("", _word_count_counter, max_tokens=5) == []


def test_naive_fixed_chunk_respects_token_budget() -> None:
    text = "كلمة كلمة كلمة كلمة كلمة كلمة"
    chunks = rche.naive_fixed_chunk(text, _word_count_counter, max_tokens=3)
    assert len(chunks) == 2
    for chunk in chunks:
        assert _word_count_counter(chunk) <= 3


def test_naive_fixed_chunk_preserves_all_words() -> None:
    text = "واحد اثنين ثلاثة اربعة خمسة"
    chunks = rche.naive_fixed_chunk(text, _word_count_counter, max_tokens=2)
    rejoined = sum(_word_count_counter(c) for c in chunks)
    assert rejoined == _word_count_counter(text)


# --- retrieve_top_k_indices --------------------------------------------


def test_retrieve_top_k_indices_returns_best_matches_first() -> None:
    question_embedding = [1.0, 0.0]
    chunk_embeddings = [
        [0.0, 1.0],  # orthogonal, worst match
        [1.0, 0.0],  # identical, best match
        [0.7, 0.7],  # partial match
    ]
    top_indices = rche.retrieve_top_k_indices(question_embedding, chunk_embeddings, k=2)
    assert top_indices[0] == 1  # best match first
    assert len(top_indices) == 2


def test_retrieve_top_k_indices_k_larger_than_list_returns_all() -> None:
    top_indices = rche.retrieve_top_k_indices([1.0], [[1.0], [0.5]], k=10)
    assert len(top_indices) == 2


# --- is_answer_retrieved -------------------------------------------------


def test_is_answer_retrieved_true_when_answer_present() -> None:
    assert rche.is_answer_retrieved(["نص فيه القاهرة موجودة هنا"], "القاهرة") is True


def test_is_answer_retrieved_false_when_answer_absent() -> None:
    assert rche.is_answer_retrieved(["نص لا علاقة له بالسؤال"], "القاهرة") is False


def test_is_answer_retrieved_ignores_diacritics() -> None:
    assert rche.is_answer_retrieved(["فيه القَاهِرَة هنا"], "القاهرة") is True


def test_is_answer_retrieved_empty_answer_is_false() -> None:
    assert rche.is_answer_retrieved(["أي نص هنا"], "") is False


# --- proportion_ci95 ---------------------------------------------------


def test_proportion_ci95_zero_n_returns_zeros() -> None:
    assert rche.proportion_ci95(0, 0) == (0.0, 0.0, 0.0)


def test_proportion_ci95_all_success_gives_narrow_high_interval() -> None:
    p, lo, hi = rche.proportion_ci95(10, 10)
    assert p == 1.0
    assert hi == 1.0  # clamped, never exceeds 1.0


def test_proportion_ci95_half_success_centers_at_point_five() -> None:
    p, lo, hi = rche.proportion_ci95(5, 10)
    assert p == pytest.approx(0.5)
    assert lo < 0.5 < hi


# --- evaluate_retrieval: fake embedder, real pipeline logic --------------


def test_evaluate_retrieval_naive_method_finds_answer() -> None:
    # Fake embedder: embeds any text as [1.0] if it contains "القاهرة",
    # else [0.0] -- deterministic, no network.
    def fake_embedder(text: str) -> list:
        return [1.0] if "القاهرة" in text else [0.0]

    example = tydiqa_loader.PrimaryExample(
        example_id="1",
        document_plaintext="نص عن باريس هنا. نص عن القاهرة هنا وهي عاصمة.",
        question="ما عاصمة مصر القاهرة؟",
        answer_text="القاهرة",
        answer_start_byte=0,
        answer_end_byte=0,
    )
    results = rche.evaluate_retrieval(
        [example], fake_embedder, _word_count_counter, "naive", max_tokens=5, top_k=1
    )
    assert results == [True]


def test_evaluate_retrieval_returns_false_when_wrong_chunk_retrieved() -> None:
    def fake_embedder(text: str) -> list:
        return [1.0] if "باريس" in text else [0.0]  # biased toward the WRONG chunk

    example = tydiqa_loader.PrimaryExample(
        example_id="1",
        document_plaintext="نص عن باريس هنا فقط. نص عن القاهرة هنا وهي عاصمة.",
        question="ما عاصمة مصر القاهرة؟",
        answer_text="القاهرة",
        answer_start_byte=0,
        answer_end_byte=0,
    )
    results = rche.evaluate_retrieval(
        [example], fake_embedder, _word_count_counter, "naive", max_tokens=5, top_k=1
    )
    assert results == [False]


def test_evaluate_retrieval_unknown_method_raises_value_error() -> None:
    example = tydiqa_loader.PrimaryExample(
        example_id="1", document_plaintext="نص", question="؟", answer_text="نص",
        answer_start_byte=0, answer_end_byte=0,
    )
    with pytest.raises(ValueError, match="Unknown method"):
        rche.evaluate_retrieval(
            [example], lambda t: [0.0], _word_count_counter, "bogus", max_tokens=5, top_k=1
        )


# --- summarize -----------------------------------------------------------


def test_summarize_empty_input() -> None:
    assert rche.summarize([], []) == "No examples evaluated."


def test_summarize_reports_worse_retrieval_honestly() -> None:
    summary = rche.summarize([True, True], [False, False])
    assert "WORSE" in summary


def test_summarize_reports_no_material_difference() -> None:
    summary = rche.summarize([True, False], [True, False])
    assert "No material difference" in summary