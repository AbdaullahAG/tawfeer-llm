"""Run the tawfeer-llm fertility benchmark and write results/latest.md.

Not part of the installable package -- this is a repo-level publishing
script, run manually or via CI to refresh the published benchmark numbers.
"""

from pathlib import Path

from ar_tokenwise.benchmark import load_corpus, render_markdown_table, run_benchmark
from ar_tokenwise.report import get_default_counter

CORPUS_PATH = Path(__file__).parent / "corpus" / "seed_corpus.jsonl"
RESULTS_PATH = Path(__file__).parent / "results" / "latest.md"


def main() -> None:
    entries = load_corpus(CORPUS_PATH)
    counter = get_default_counter()
    results = run_benchmark(entries, counter=counter)

    table = render_markdown_table(results)
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(
        f"# tawfeer-llm Benchmark Results\n\n"
        f"Measured against tiktoken `o200k_base` on {len(entries)} seed "
        f"corpus entries.\n\n{table}\n",
        encoding="utf-8",
    )
    print(table)


if __name__ == "__main__":
    main()