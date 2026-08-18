"""Validate ar_tokenwise.dialect against an independent hand-labeled corpus.

Not part of the installable package -- this is a repo-level validation
script. The corpus here (dialect_validation.jsonl) was authored separately
from dialect.py's marker lists specifically to avoid a circular/self-
validating accuracy number: this measures whether the heuristic actually
works on unseen sentences, not whether it can find the words it was told
to look for.

"Correct" is defined as: the highest-probability category in the returned
distribution matches the expected_category label. INSUFFICIENT_TEXT and
NO_SIGNAL results are counted as incorrect (the tool failed to identify
the dialect), which is the honest, conservative choice.
"""

import json
from collections import defaultdict
from pathlib import Path

from ar_tokenwise.dialect import DetectionStatus, detect_dialect

CORPUS_PATH = Path(__file__).parent / "corpus" / "dialect_validation.jsonl"
RESULTS_PATH = Path(__file__).parent / "results" / "dialect_validation.md"


def main() -> None:
    lines = [
        line for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()
    ]

    total_by_category: dict[str, int] = defaultdict(int)
    correct_by_category: dict[str, int] = defaultdict(int)
    total = 0
    correct = 0

    for line in lines:
        entry = json.loads(line)
        expected = entry["expected_category"]
        text = entry["text"]

        result = detect_dialect(text)
        total += 1
        total_by_category[expected] += 1

        is_correct = False
        if result.status is DetectionStatus.DISTRIBUTION:
            top_category = max(result.distribution, key=result.distribution.get)
            is_correct = top_category.value == expected

        if is_correct:
            correct += 1
            correct_by_category[expected] += 1

    overall_accuracy = (correct / total) * 100.0 if total > 0 else 0.0

    lines_out = [
        "# dialect.py Validation Results",
        "",
        f"Measured on {total} hand-labeled sentences, held independently "
        "from the marker lists used by the detector.",
        "",
        f"**Overall accuracy (top-category match): {overall_accuracy:.1f}%**",
        "",
        "| Category | Entries | Correct | Accuracy |",
        "|---|---|---|---|",
    ]
    for category in sorted(total_by_category):
        cat_total = total_by_category[category]
        cat_correct = correct_by_category[category]
        cat_accuracy = (cat_correct / cat_total) * 100.0 if cat_total > 0 else 0.0
        lines_out.append(f"| {category} | {cat_total} | {cat_correct} | {cat_accuracy:.1f}% |")

    output = "\n".join(lines_out) + "\n"
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()