"""Shared TyDiQA-Arabic loading helpers for benchmark eval scripts.

Not part of the installed package. Downloads via the HuggingFace
`datasets` library at call time -- nothing is cached in this repo (same
posture as run_real_world_benchmark.py's FLORES-200 loading).

*** IMPORTANT, READ BEFORE TRUSTING THIS FILE ***
The field names used below (`context`/`question`/`answers` for GoldP,
`document_plaintext`/`question_text`/`annotations` for primary_task)
are based on the documented HuggingFace dataset card conventions for
`google-research-datasets/tydiqa`, NOT verified by actually running
this code -- this sandbox cannot reach huggingface.co to download and
inspect the dataset live (unlike everything else in this project,
which was verified by real execution before being called done). If a
field name below is wrong, you'll get a clear KeyError naming the
missing field and suggesting you inspect `dataset.features` yourself --
report back what the actual schema is and this file should be
corrected to match, not silently patched around.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoldPExample:
    """One TyDiQA-GoldP (secondary_task) Arabic example: short passage + answer span."""

    example_id: str
    passage: str
    question: str
    answer_text: str


@dataclass(frozen=True)
class PrimaryExample:
    """One TyDiQA primary_task Arabic example: full document + answer location.

    Attributes:
        answer_start_byte / answer_end_byte: byte offsets into
            ``document_plaintext`` for the passage that contains the
            answer -- used by the chunking eval to check whether the
            chunk retrieved for a question actually overlaps the
            correct passage, not just whether the answer STRING happens
            to appear elsewhere in the document by coincidence.
    """

    example_id: str
    document_plaintext: str
    question: str
    answer_text: str
    answer_start_byte: int
    answer_end_byte: int


def _require_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "This script needs the `datasets` library: pip install datasets"
        ) from exc
    return load_dataset


def _is_arabic_example(example_id: str) -> bool:
    """TyDiQA example IDs are prefixed by language, e.g. 'arabic-2385726...'."""
    return example_id.lower().startswith("arabic")


def load_tydiqa_goldp_arabic(
    split: str = "validation", limit: int | None = None
) -> list[GoldPExample]:
    """Load Arabic TyDiQA-GoldP (secondary_task) examples.

    Args:
        split: "train" or "validation" (there is no public test split).
        limit: optional cap on example count, for a quick smoke run
            before committing to the full set.

    Returns:
        List of GoldPExample, Arabic only, examples with no answer
        skipped (GoldP includes some unanswerable questions).

    Raises:
        ImportError: if `datasets` isn't installed.
        KeyError: if the dataset's actual field names don't match what
            this function expects -- see the module docstring.
    """
    load_dataset = _require_datasets()
    dataset = load_dataset("google-research-datasets/tydiqa", "secondary_task", split=split)

    examples: list[GoldPExample] = []
    for row in dataset:
        try:
            example_id = row["id"]
            if not _is_arabic_example(example_id):
                continue
            answer_texts = row["answers"]["text"]
            if not answer_texts:
                continue
            examples.append(
                GoldPExample(
                    example_id=example_id,
                    passage=row["context"],
                    question=row["question"],
                    answer_text=answer_texts[0],
                )
            )
        except KeyError as exc:
            raise KeyError(
                f"TyDiQA-GoldP row is missing expected field {exc}. The dataset's "
                "actual schema may differ from what this loader assumes -- inspect "
                "`dataset.features` yourself and report back so this loader can be "
                "corrected (see this module's docstring)."
            ) from exc

        if limit and len(examples) >= limit:
            break

    return examples


def load_tydiqa_primary_arabic(
    split: str = "validation", limit: int | None = None
) -> list[PrimaryExample]:
    """Load Arabic TyDiQA primary_task examples (full documents).

    Args:
        split: "train" or "validation".
        limit: optional cap on example count.

    Returns:
        List of PrimaryExample, Arabic only, examples with no minimal
        answer annotation skipped.

    Raises:
        ImportError: if `datasets` isn't installed.
        KeyError: if the dataset's actual field names don't match what
            this function expects -- see the module docstring.
    """
    load_dataset = _require_datasets()
    dataset = load_dataset("google-research-datasets/tydiqa", "primary_task", split=split)

    examples: list[PrimaryExample] = []
    for row in dataset:
        try:
            example_id = row["document_url"] if "document_url" in row else row.get("example_id", "")
            if "language" in row:
                if row["language"].lower() != "arabic":
                    continue
            elif not _is_arabic_example(str(example_id)):
                continue

            document_plaintext = row["document_plaintext"]
            annotations = row["annotations"]
            if not annotations:
                continue

            minimal_starts = annotations[0].get("minimal_answers_start_byte", [])
            minimal_ends = annotations[0].get("minimal_answers_end_byte", [])
            if not minimal_starts or minimal_starts[0] < 0:
                continue  # unanswerable example

            start_byte = minimal_starts[0]
            end_byte = minimal_ends[0]
            answer_text = document_plaintext.encode("utf-8")[start_byte:end_byte].decode(
                "utf-8", errors="ignore"
            )
            if not answer_text.strip():
                continue

            examples.append(
                PrimaryExample(
                    example_id=str(example_id),
                    document_plaintext=document_plaintext,
                    question=row["question_text"],
                    answer_text=answer_text,
                    answer_start_byte=start_byte,
                    answer_end_byte=end_byte,
                )
            )
        except KeyError as exc:
            raise KeyError(
                f"TyDiQA primary_task row is missing expected field {exc}. The "
                "dataset's actual schema may differ from what this loader assumes "
                "-- inspect `dataset.features` yourself and report back so this "
                "loader can be corrected (see this module's docstring)."
            ) from exc

        if limit and len(examples) >= limit:
            break

    return examples