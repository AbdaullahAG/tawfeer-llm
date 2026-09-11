"""Does normalize() change model comprehension? Ground-truth QA, no LLM judge.

METHODOLOGY, and why it deliberately avoids LLM-as-judge entirely:
Recent research on LLM-as-judge (searched and reviewed before writing
this script, mid-2026) documents real, material biases: self-preference
(a judge favors outputs from its own model family), position bias (one
2025 study found 8 of 9 tested judge models favored whichever answer
was shown first), and verbosity bias (longer answers score higher
independent of quality). All of that is IRRELEVANT here, by
construction: this script never asks a model to judge another model's
output. Instead, it uses TyDiQA-GoldP -- a real, human-annotated Arabic
QA dataset with a KNOWN CORRECT answer span for every question -- and
scores the model's generated answer against that fixed ground truth
using token-overlap F1 (the standard, established metric for extractive
QA, not an LLM opinion).

The actual experiment: for each (passage, question, gold_answer), ask a
real model to answer using the ORIGINAL passage, and separately using
the passage after ar_tokenwise.normalize(). Compare F1 against the gold
answer in both conditions. If normalize() doesn't hurt comprehension,
F1_normalized should be close to F1_original -- and this script reports
whatever the real numbers say, including if they show a real drop.

Requires a real model to actually generate answers (not just count
tokens) -- a capability that does NOT belong in the core `ar_tokenwise`
library (see README's "How this compares": this project is
middleware/measurement, not an LLM-calling wrapper). The minimal
generation helpers below exist ONLY in this evaluation script.

*** IMPORTANT CAVEAT, stated plainly ***
This script could not be run end-to-end in the environment it was
written in (no network access to huggingface.co or the provider APIs
from that sandbox). The TyDiQA loading (benchmark/_tydiqa_loader.py)
and the F1 scoring (benchmark/_text_similarity.py) were each verified
independently, but the full pipeline -- real dataset rows flowing
through real API calls -- has NOT been observed to work end-to-end by
whoever wrote this. Run it, and if something breaks, the error message
is written to tell you where to look, not to hide the seam.

Usage:
    pip install -e ".[tokenizers,providers]"
    pip install datasets

    # Anthropic:
    export ANTHROPIC_API_KEY="..."
    python benchmark/run_comprehension_eval.py \
        --provider anthropic --model claude-opus-4-8 --limit 50

    # Azure AI Foundry (any model deployed there, e.g. an OpenAI GPT
    # family model): --model is your DEPLOYMENT NAME (an alias you
    # chose when deploying), not necessarily the underlying model's own
    # name -- Foundry lets you name deployments however you like.
    pip install openai
    export AZURE_INFERENCE_CREDENTIAL="..."
    python benchmark/run_comprehension_eval.py \
        --provider azure_foundry --model GPT-5.6-sol \
        --azure-endpoint https://<your-resource>.services.ai.azure.com \
        --azure-api-version 2024-10-21 --limit 50
"""

from __future__ import annotations

import argparse
import os
import statistics
from pathlib import Path

from ar_tokenwise.normalize import NormalizationLevel, normalize

from _text_similarity import f1_score  # type: ignore[import-not-found]
from _tydiqa_loader import GoldPExample, load_tydiqa_goldp_arabic  # type: ignore[import-not-found]

RESULTS_PATH = Path(__file__).parent / "results" / "comprehension_eval.md"

_ANSWER_PROMPT_TEMPLATE = (
    "اقرأ المقطع التالي وأجب عن السؤال بعبارة قصيرة ومباشرة مقتبسة من المقطع، "
    "بدون أي شرح إضافي.\n\nالمقطع:\n{passage}\n\nالسؤال: {question}\n\nالإجابة:"
)


def _build_anthropic_answerer(model: str):
    """Return a callable (passage, question) -> answer_text, via Anthropic.

    This is a minimal, single-purpose generation helper -- it exists
    only here, not in ar_tokenwise, per this module's docstring.
    """
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "Anthropic answer generation needs the anthropic package: "
            "pip install ar-tokenwise[providers]"
        ) from exc

    client = anthropic.Anthropic()

    def _answer(passage: str, question: str) -> str:
        message = client.messages.create(
            model=model,
            max_tokens=64,
            messages=[
                {
                    "role": "user",
                    "content": _ANSWER_PROMPT_TEMPLATE.format(passage=passage, question=question),
                }
            ],
        )
        return "".join(block.text for block in message.content if hasattr(block, "text"))

    return _answer


def _build_gemini_answerer(model: str):
    """Return a callable (passage, question) -> answer_text, via Gemini."""
    try:
        from google import genai
    except ImportError as exc:
        raise ImportError(
            "Gemini answer generation needs the google-genai package: "
            "pip install ar-tokenwise[providers]"
        ) from exc

    client = genai.Client()

    def _answer(passage: str, question: str) -> str:
        response = client.models.generate_content(
            model=model,
            contents=_ANSWER_PROMPT_TEMPLATE.format(passage=passage, question=question),
        )
        return response.text or ""

    return _answer


def _build_azure_foundry_answerer(deployment_name: str, endpoint: str, api_version: str):
    """Return a callable (passage, question) -> answer_text, via Azure AI Foundry.

    Azure AI Foundry (formerly Azure OpenAI Service) uses the `openai`
    package's AzureOpenAI client -- NOT a separate Azure-specific SDK.
    `deployment_name` is the alias YOU chose when deploying the model in
    your Foundry resource, not necessarily the underlying model's own
    name (e.g. you could deploy a GPT-family model under the deployment
    name "GPT-5.6-sol" and that's what goes in --model here).

    Args:
        deployment_name: Your Foundry deployment name.
        endpoint: Your Foundry resource endpoint, e.g.
            "https://<resource>.services.ai.azure.com".
        api_version: Azure OpenAI API version string, e.g. "2024-10-21"
            -- required exactly, unlike a model name there's no safe
            guess; check your Foundry resource's documentation for the
            current value if this one is rejected.

    Raises:
        ImportError: if the `openai` package is not installed.
        RuntimeError: if no Azure credential is found in the environment.
    """
    try:
        from openai import AzureOpenAI
    except ImportError as exc:
        raise ImportError(
            "Azure AI Foundry answer generation needs the openai package: "
            "pip install openai"
        ) from exc

    api_key = os.environ.get("AZURE_INFERENCE_CREDENTIAL") or os.environ.get(
        "AZURE_OPENAI_API_KEY"
    )
    if not api_key:
        raise RuntimeError(
            "Set AZURE_INFERENCE_CREDENTIAL or AZURE_OPENAI_API_KEY with your "
            "Foundry resource's key."
        )

    client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)

    def _answer(passage: str, question: str) -> str:
        completion = client.chat.completions.create(
            model=deployment_name,
            max_completion_tokens=64,
            messages=[
                {
                    "role": "user",
                    "content": _ANSWER_PROMPT_TEMPLATE.format(passage=passage, question=question),
                }
            ],
        )
        return completion.choices[0].message.content or ""

    return _answer


def evaluate_examples(
    examples: list[GoldPExample],
    answerer,
    level: NormalizationLevel,
) -> tuple[list[float], list[float]]:
    """Return (f1_scores_original, f1_scores_normalized) for a set of examples.

    Args:
        examples: GoldP examples to evaluate.
        answerer: A (passage, question) -> answer_text callable.
        level: Normalization level applied to the passage in the
            "normalized" condition.
    """
    f1_original: list[float] = []
    f1_normalized: list[float] = []

    for example in examples:
        original_answer = answerer(example.passage, example.question)
        f1_original.append(f1_score(original_answer, example.answer_text))

        normalized_passage = normalize(example.passage, level=level)
        normalized_answer = answerer(normalized_passage, example.question)
        f1_normalized.append(f1_score(normalized_answer, example.answer_text))

    return f1_original, f1_normalized


def summarize(f1_original: list[float], f1_normalized: list[float]) -> str:
    """Render a Markdown summary comparing original vs normalized F1."""
    if not f1_original:
        return "No examples evaluated."

    orig_mean = statistics.mean(f1_original)
    norm_mean = statistics.mean(f1_normalized)
    orig_median = statistics.median(f1_original)
    norm_median = statistics.median(f1_normalized)
    delta = norm_mean - orig_mean

    lines = [
        f"n = {len(f1_original)}",
        "",
        "| Condition | Mean F1 | Median F1 |",
        "|---|---|---|",
        f"| Original passage | {orig_mean:.3f} | {orig_median:.3f} |",
        f"| Normalized passage | {norm_mean:.3f} | {norm_median:.3f} |",
        "",
        f"**Delta (normalized - original): {delta:+.3f}**",
    ]
    if delta < -0.02:
        lines.append(
            "\nNormalization measurably HURT comprehension on this sample. "
            "This is a real result to report, not to explain away."
        )
    elif delta > 0.02:
        lines.append(
            "\nNormalization measurably IMPROVED apparent comprehension on this "
            "sample -- plausibly just noise or a token-budget effect (shorter "
            "prompt leaves more room), not a claim that normalization aids "
            "understanding (see README's 'What it doesn't claim')."
        )
    else:
        lines.append("\nNo material difference detected on this sample.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="QA comprehension eval: original vs normalized.")
    parser.add_argument("--provider", choices=["anthropic", "gemini", "azure_foundry"], default="anthropic")
    parser.add_argument(
        "--model",
        required=True,
        help="e.g. claude-opus-4-8, gemini-3-flash, or your Azure Foundry deployment name",
    )
    parser.add_argument(
        "--azure-endpoint",
        default=None,
        help="Required if --provider azure_foundry, e.g. https://<resource>.services.ai.azure.com",
    )
    parser.add_argument(
        "--azure-api-version",
        default=None,
        help="Required if --provider azure_foundry, e.g. 2024-10-21",
    )
    parser.add_argument("--level", choices=["light", "medium"], default="light")
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument(
        "--limit", type=int, default=50, help="Number of examples to evaluate (API calls cost time/money)."
    )
    args = parser.parse_args()

    try:
        examples = load_tydiqa_goldp_arabic(split=args.split, limit=args.limit)
    except ImportError as exc:
        print(f"Cannot load dataset: {exc}")
        return 1
    except KeyError as exc:
        print(f"Dataset schema mismatch: {exc}")
        return 1

    if not examples:
        print("No Arabic examples loaded -- check the dataset split and _tydiqa_loader.py.")
        return 1

    print(f"Loaded {len(examples)} Arabic TyDiQA-GoldP examples.")

    try:
        if args.provider == "anthropic":
            answerer = _build_anthropic_answerer(args.model)
        elif args.provider == "gemini":
            answerer = _build_gemini_answerer(args.model)
        else:
            if not args.azure_endpoint or not args.azure_api_version:
                print(
                    "--provider azure_foundry requires both --azure-endpoint "
                    "and --azure-api-version."
                )
                return 1
            answerer = _build_azure_foundry_answerer(
                args.model, args.azure_endpoint, args.azure_api_version
            )
    except ImportError as exc:
        print(f"Cannot build answerer: {exc}")
        return 1
    except RuntimeError as exc:
        print(f"Cannot build answerer: {exc}")
        return 1

    level = NormalizationLevel.LIGHT if args.level == "light" else NormalizationLevel.MEDIUM
    f1_original, f1_normalized = evaluate_examples(examples, answerer, level)

    summary = summarize(f1_original, f1_normalized)
    output = (
        f"# Comprehension eval: {args.provider}/{args.model}, level={args.level}\n\n{summary}\n"
    )

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(output, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
