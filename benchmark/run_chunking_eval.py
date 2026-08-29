"""Does chunk_text() help or hurt RAG retrieval accuracy vs naive chunking?

METHODOLOGY:
For each TyDiQA primary_task Arabic example (a full Wikipedia document,
a question, and a known answer string):
1. Chunk the document two ways: NAIVE (fixed token budget, cuts
   mid-sentence freely -- the common, unaware baseline) and OURS
   (ar_tokenwise.chunk_text(), sentence-boundary-aware).
2. Embed every chunk and the question (real embeddings, via Gemini's
   embed_content API).
3. Retrieve the top-k chunks by cosine similarity to the question.
4. Check whether the known answer TEXT actually appears in what was
   retrieved (content-containment, not byte-offset matching -- chosen
   because chunk_text() rejoins sentences with a single space, so
   reconstructed chunk text is not guaranteed byte-identical to the
   original document's whitespace/newlines; content containment sidesteps
   that entirely and is arguably the more meaningful success criterion
   anyway: "is the answer actually present in what we retrieved").

Retrieval accuracy (successes / n) is reported per method with a 95%
Wald confidence interval for a proportion.

*** IMPORTANT CAVEAT, same as run_comprehension_eval.py ***
Not run end-to-end by whoever wrote this (no network access to
huggingface.co or the Gemini API from this sandbox). The individual
pieces (cosine similarity, chunking, containment check, CI math) are
each unit-tested with fake data; the full real pipeline has not been
observed working. Run it, and report back if something breaks.

Cost/quota note: this calls the embedding API once per chunk plus once
per question, for every example. Full Wikipedia documents can produce
many chunks -- start with a small --limit (default 20) before scaling up.

Usage:
    pip install -e ".[tokenizers,providers]"
    pip install datasets
    export GEMINI_API_KEY="..."
    python benchmark/run_chunking_eval.py \
        --embedding-model gemini-embedding-001 --limit 20
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Callable

from ar_tokenwise.chunking import chunk_text
from ar_tokenwise.report import TokenCounter, get_default_counter

from _text_similarity import normalize_answer_for_comparison  # type: ignore[import-not-found]
from _tydiqa_loader import PrimaryExample, load_tydiqa_primary_arabic  # type: ignore[import-not-found]

RESULTS_PATH = Path(__file__).parent / "results" / "chunking_eval.md"

Embedder = Callable[[str], list[float]]


def _build_gemini_embedder(model: str) -> Embedder:
    """Return a callable text -> embedding vector, via Gemini's embed_content API."""
    try:
        from google import genai
    except ImportError as exc:
        raise ImportError(
            "Gemini embeddings need the google-genai package: "
            "pip install ar-tokenwise[providers]"
        ) from exc

    client = genai.Client()

    def _embed(text: str) -> list[float]:
        response = client.models.embed_content(model=model, contents=text)
        return list(response.embeddings[0].values)

    return _embed


def naive_fixed_chunk(text: str, counter: TokenCounter, max_tokens: int) -> list[str]:
    """Fixed-token-budget chunking with NO sentence-boundary awareness.

    The common, unaware baseline this evaluates chunk_text() against:
    cuts wherever the token budget runs out, mid-sentence or not.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for word in words:
        word_tokens = counter(word)
        if current and current_tokens + word_tokens > max_tokens:
            chunks.append(" ".join(current))
            current = [word]
            current_tokens = word_tokens
        else:
            current.append(word)
            current_tokens += word_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors. 0.0 if either is zero."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve_top_k_indices(
    question_embedding: list[float], chunk_embeddings: list[list[float]], k: int
) -> list[int]:
    """Return the indices of the k highest-cosine-similarity chunks, best first."""
    scored = [
        (cosine_similarity(question_embedding, emb), i) for i, emb in enumerate(chunk_embeddings)
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [i for _, i in scored[:k]]


def is_answer_retrieved(retrieved_chunks: list[str], answer_text: str) -> bool:
    """Whether the (lightly normalized) answer text appears in the retrieved chunks."""
    normalized_answer = normalize_answer_for_comparison(answer_text)
    if not normalized_answer:
        return False
    combined = normalize_answer_for_comparison(" ".join(retrieved_chunks))
    return normalized_answer in combined


def evaluate_retrieval(
    examples: list[PrimaryExample],
    embedder: Embedder,
    counter: TokenCounter,
    method: str,
    max_tokens: int,
    top_k: int,
) -> list[bool]:
    """Run the full chunk -> embed -> retrieve -> check pipeline for one method.

    Args:
        method: "naive" or "sentence_aware".

    Returns:
        List of per-example booleans (True = answer was retrieved).
    """
    results: list[bool] = []
    for example in examples:
        if method == "naive":
            chunks = naive_fixed_chunk(example.document_plaintext, counter, max_tokens)
        elif method == "sentence_aware":
            chunks = chunk_text(example.document_plaintext, counter=counter, max_tokens=max_tokens)
        else:
            raise ValueError(f"Unknown method: {method!r}")

        if not chunks:
            continue

        chunk_embeddings = [embedder(chunk) for chunk in chunks]
        question_embedding = embedder(example.question)
        top_indices = retrieve_top_k_indices(question_embedding, chunk_embeddings, top_k)
        retrieved_chunks = [chunks[i] for i in top_indices]

        results.append(is_answer_retrieved(retrieved_chunks, example.answer_text))
    return results


def proportion_ci95(successes: int, n: int) -> tuple[float, float, float]:
    """Wald 95% CI for a proportion. Valid at moderate-to-large n."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    margin = 1.96 * (p * (1 - p) / n) ** 0.5
    return p, max(0.0, p - margin), min(1.0, p + margin)


def summarize(naive_results: list[bool], sentence_aware_results: list[bool]) -> str:
    """Render a Markdown summary comparing retrieval accuracy between methods."""
    if not naive_results or not sentence_aware_results:
        return "No examples evaluated."

    naive_p, naive_lo, naive_hi = proportion_ci95(sum(naive_results), len(naive_results))
    sa_p, sa_lo, sa_hi = proportion_ci95(
        sum(sentence_aware_results), len(sentence_aware_results)
    )
    delta = sa_p - naive_p

    lines = [
        f"n = {len(naive_results)}",
        "",
        "| Method | Retrieval accuracy | 95% CI |",
        "|---|---|---|",
        f"| Naive fixed-token | {naive_p:.1%} | [{naive_lo:.1%}, {naive_hi:.1%}] |",
        f"| chunk_text() (sentence-aware) | {sa_p:.1%} | [{sa_lo:.1%}, {sa_hi:.1%}] |",
        "",
        f"**Delta (sentence-aware - naive): {delta:+.1%}**",
    ]
    if delta < -0.02:
        lines.append(
            "\nSentence-aware chunking retrieved WORSE on this sample. A real "
            "result to report as-is, not to explain away."
        )
    elif delta > 0.02:
        lines.append("\nSentence-aware chunking retrieved BETTER on this sample.")
    else:
        lines.append("\nNo material difference detected on this sample.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG retrieval accuracy: naive vs chunk_text().")
    parser.add_argument("--embedding-model", required=True, help="e.g. gemini-embedding-001")
    parser.add_argument("--chunk-max-tokens", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of documents to evaluate (each document costs multiple embedding calls).",
    )
    args = parser.parse_args()

    try:
        examples = load_tydiqa_primary_arabic(split=args.split, limit=args.limit)
    except ImportError as exc:
        print(f"Cannot load dataset: {exc}")
        return 1
    except KeyError as exc:
        print(f"Dataset schema mismatch: {exc}")
        return 1

    if not examples:
        print("No Arabic examples loaded -- check the dataset split and _tydiqa_loader.py.")
        return 1

    print(f"Loaded {len(examples)} Arabic TyDiQA primary_task examples.")

    try:
        embedder = _build_gemini_embedder(args.embedding_model)
    except ImportError as exc:
        print(f"Cannot build embedder: {exc}")
        return 1

    counter = get_default_counter()

    naive_results = evaluate_retrieval(
        examples, embedder, counter, "naive", args.chunk_max_tokens, args.top_k
    )
    sentence_aware_results = evaluate_retrieval(
        examples, embedder, counter, "sentence_aware", args.chunk_max_tokens, args.top_k
    )

    summary = summarize(naive_results, sentence_aware_results)
    output = (
        f"# Chunking/retrieval eval: {args.embedding_model}, "
        f"max_tokens={args.chunk_max_tokens}, top_k={args.top_k}\n\n{summary}\n"
    )

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(output, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())