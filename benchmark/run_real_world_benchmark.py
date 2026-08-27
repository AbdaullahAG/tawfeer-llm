"""Real-world, multi-provider fertility benchmark.

Data sources:
1. FLORES-200 devtest split (arb_Arab=MSA, ars_Arab=Najdi/Gulf-adjacent,
   ary_Arab=Moroccan/Maghrebi, arz_Arab=Egyptian) -- professionally
   translated, quality-controlled (TQS > 90%), CC BY-SA 4.0, the
   field-standard corpus for exactly this kind of tokenizer-fertility
   comparison (cited across dozens of tokenizer papers). Downloaded at
   RUN TIME via the `datasets` library -- never stored in this repo, to
   avoid any redistribution question for a dataset we don't own.
   NOTE: FLORES has no Levantine Arabic variant and no
   mixed-Arabic-English or formal/legal register content -- that's what
   this project's own corpus (source 2) is for.
2. This project's own benchmark/corpus/seed_corpus.jsonl -- hand-
   authored, documented as such (see benchmark/README.md), included here
   specifically for the registers FLORES-200 doesn't cover.

Tokenizers/counters (each skipped, with a clear reason printed, if its
requirement isn't met -- never silently omitted):
- tiktoken o200k_base, via get_default_counter() -- needs the
  `tokenizers` extra and network on first use only.
- Anthropic's real count_tokens API, via get_anthropic_counter() --
  needs the `providers` extra, an ANTHROPIC_API_KEY, and a
  --anthropic-model you supply (no default: model names go stale fast,
  see report.py's docstring for the same principle applied to pricing).
- Gemini's real count_tokens API, via get_gemini_counter() -- same
  shape, GEMINI_API_KEY (or GOOGLE_API_KEY) and --gemini-model.

Normalization levels tested: LIGHT and MEDIUM, both, always -- decided
before running, not selected afterward based on which one looks better.

Statistics reported per group: n, mean with a 95% CI (normal
approximation -- valid at FLORES's sample sizes, ~1,000/language), and
median (robust to outlier sentences skewing the mean). This measures
fertility (tokens per word) only -- not downstream generation quality,
latency, or accuracy.

Usage:
    pip install datasets  # not a project dependency, only needed here
    python benchmark/run_real_world_benchmark.py
    python benchmark/run_real_world_benchmark.py \
        --anthropic-model claude-opus-4-8 --gemini-model gemini-3-flash
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Callable, NamedTuple

from ar_tokenwise.normalize import NormalizationLevel, normalize
from ar_tokenwise.report import TokenCounter, get_default_counter

FLORES_LANG_CODES: dict[str, str] = {
    "msa": "arb_Arab",
    "gulf_najdi": "ars_Arab",
    "maghrebi_moroccan": "ary_Arab",
    "egyptian": "arz_Arab",
}

CORPUS_PATH = Path(__file__).parent / "corpus" / "seed_corpus.jsonl"
RESULTS_PATH = Path(__file__).parent / "results" / "real_world.md"


class GroupStats(NamedTuple):
    """Summary statistics for one (source, register, tokenizer, level) group."""

    label: str
    n: int
    orig_mean: float
    orig_ci: tuple[float, float]
    orig_median: float
    opt_mean: float
    opt_ci: tuple[float, float]
    opt_median: float
    percent_saved: float


def load_flores_sentences(lang_code: str, split: str = "devtest") -> list[str]:
    """Load real FLORES-200 sentences for one language code, at call time.

    Args:
        lang_code: A FLORES-200 language code, e.g. "arb_Arab".
        split: "dev" or "devtest" (the hidden "test" split isn't public).

    Returns:
        List of sentence strings.

    Raises:
        ImportError: if the `datasets` library isn't installed.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "load_flores_sentences() needs the `datasets` library: "
            "pip install datasets"
        ) from exc

    dataset = load_dataset("facebook/flores", lang_code, split=split, trust_remote_code=True)
    return [row["sentence"] for row in dataset]


def load_own_corpus_by_group() -> dict[str, list[str]]:
    """Load this project's own corpus, grouped by (category[:region])."""
    from ar_tokenwise.benchmark import load_corpus

    entries = load_corpus(CORPUS_PATH)
    groups: dict[str, list[str]] = {}
    for entry in entries:
        key = f"{entry.category.value}:{entry.region}" if entry.region else entry.category.value
        groups.setdefault(key, []).append(entry.text)
    return groups


def mean_and_ci95(values: list[float]) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) via normal approximation.

    Valid for large samples (roughly n >= 30) -- FLORES devtest groups
    have ~1,000 sentences, comfortably large enough; this project's own
    corpus groups are much smaller (12-40), where the CI will be wide --
    reported honestly rather than hidden.
    """
    n = len(values)
    mean = statistics.mean(values)
    if n < 2:
        return mean, mean, mean
    stdev = statistics.stdev(values)
    margin = 1.96 * stdev / (n**0.5)
    return mean, mean - margin, mean + margin


def compute_fertility_series(
    sentences: list[str], counter: TokenCounter, level: NormalizationLevel
) -> tuple[list[float], list[float]]:
    """Compute per-sentence fertility before/after normalization.

    Sentences that become zero words after normalization (shouldn't
    happen in practice, but guarded) are skipped from that side's list
    rather than causing a division error.

    Returns:
        (original_fertilities, optimized_fertilities) -- same length
        unless a normalized sentence had zero words.
    """
    original: list[float] = []
    optimized: list[float] = []
    for sentence in sentences:
        words = len(sentence.split())
        if words == 0:
            continue
        original.append(counter(sentence) / words)

        normalized_text = normalize(sentence, level=level)
        normalized_words = len(normalized_text.split())
        if normalized_words == 0:
            continue
        optimized.append(counter(normalized_text) / normalized_words)
    return original, optimized


def summarize_group(label: str, original: list[float], optimized: list[float]) -> GroupStats:
    """Build a GroupStats summary from raw fertility lists."""
    orig_mean, orig_lo, orig_hi = mean_and_ci95(original)
    opt_mean, opt_lo, opt_hi = mean_and_ci95(optimized)
    percent_saved = ((orig_mean - opt_mean) / orig_mean) * 100.0 if orig_mean > 0 else 0.0

    return GroupStats(
        label=label,
        n=len(original),
        orig_mean=orig_mean,
        orig_ci=(orig_lo, orig_hi),
        orig_median=statistics.median(original) if original else 0.0,
        opt_mean=opt_mean,
        opt_ci=(opt_lo, opt_hi),
        opt_median=statistics.median(optimized) if optimized else 0.0,
        percent_saved=percent_saved,
    )


def render_markdown_table(groups: list[GroupStats]) -> str:
    """Render a list of GroupStats as a Markdown table."""
    header = (
        "| Group | n | Fertility before (mean [95% CI], median) | "
        "Fertility after (mean [95% CI], median) | Saved % |\n"
        "|---|---|---|---|---|\n"
    )
    rows = []
    for g in groups:
        before = f"{g.orig_mean:.2f} [{g.orig_ci[0]:.2f}, {g.orig_ci[1]:.2f}], {g.orig_median:.2f}"
        after = f"{g.opt_mean:.2f} [{g.opt_ci[0]:.2f}, {g.opt_ci[1]:.2f}], {g.opt_median:.2f}"
        rows.append(f"| {g.label} | {g.n} | {before} | {after} | {g.percent_saved:.1f}% |")
    return header + "\n".join(rows)


def build_available_counters(
    anthropic_model: str | None, gemini_model: str | None
) -> dict[str, tuple[TokenCounter | None, str]]:
    """Build every requested counter, or a clear skip-reason if unavailable.

    Returns:
        Mapping of counter name -> (counter_or_None, note). ``note`` is
        always populated (either "ok" or why it was skipped), so a
        skipped counter is visible in the report, not silently absent.
    """
    counters: dict[str, tuple[TokenCounter | None, str]] = {}

    try:
        counters["tiktoken_o200k_base"] = (get_default_counter(), "ok")
    except ImportError as exc:
        counters["tiktoken_o200k_base"] = (None, f"skipped: {exc}")
    except Exception as exc:  # noqa: BLE001 -- network failures vary (HTTPError, ConnectionError, ...)
        counters["tiktoken_o200k_base"] = (
            None,
            f"skipped: tiktoken encoding fetch failed ({type(exc).__name__}: {exc}) -- "
            "likely a network/firewall issue on first use, see CONTRIBUTING.md's "
            "'A note on tiktoken's first-use network call'",
        )

    if not os.environ.get("ANTHROPIC_API_KEY"):
        counters["anthropic"] = (None, "skipped: ANTHROPIC_API_KEY not set")
    elif not anthropic_model:
        counters["anthropic"] = (None, "skipped: --anthropic-model not given")
    else:
        try:
            from ar_tokenwise.provider_counters import get_anthropic_counter

            counters[f"anthropic:{anthropic_model}"] = (
                get_anthropic_counter(model=anthropic_model),
                "ok",
            )
        except ImportError as exc:
            counters["anthropic"] = (None, f"skipped: {exc}")

    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        counters["gemini"] = (None, "skipped: GEMINI_API_KEY/GOOGLE_API_KEY not set")
    elif not gemini_model:
        counters["gemini"] = (None, "skipped: --gemini-model not given")
    else:
        try:
            from ar_tokenwise.provider_counters import get_gemini_counter

            counters[f"gemini:{gemini_model}"] = (get_gemini_counter(model=gemini_model), "ok")
        except ImportError as exc:
            counters["gemini"] = (None, f"skipped: {exc}")

    return counters


def main() -> int:
    parser = argparse.ArgumentParser(description="Real-world multi-provider fertility benchmark.")
    parser.add_argument("--anthropic-model", default=None, help="e.g. claude-opus-4-8")
    parser.add_argument("--gemini-model", default=None, help="e.g. gemini-3-flash")
    parser.add_argument(
        "--flores-split", default="devtest", choices=["dev", "devtest"], help="FLORES-200 split."
    )
    parser.add_argument(
        "--skip-flores", action="store_true", help="Skip FLORES-200 (own corpus only)."
    )
    parser.add_argument(
        "--skip-own-corpus", action="store_true", help="Skip this project's own corpus."
    )
    args = parser.parse_args()

    counters = build_available_counters(args.anthropic_model, args.gemini_model)

    print("## Counters\n")
    for name, (counter, note) in counters.items():
        print(f"- {name}: {note}")
    print()

    active_counters = {name: c for name, (c, _) in counters.items() if c is not None}
    if not active_counters:
        print("No counters available -- nothing to measure. Install `tokenizers` at minimum.")
        return 1

    text_groups: dict[str, list[str]] = {}

    if not args.skip_flores:
        for name, lang_code in FLORES_LANG_CODES.items():
            try:
                text_groups[f"flores:{name}"] = load_flores_sentences(
                    lang_code, split=args.flores_split
                )
            except ImportError as exc:
                print(f"Skipping FLORES-200 entirely: {exc}")
                break

    if not args.skip_own_corpus:
        text_groups.update({f"own_corpus:{k}": v for k, v in load_own_corpus_by_group().items()})

    if not text_groups:
        print("No text groups to measure (both sources skipped or FLORES unavailable).")
        return 1

    all_stats: list[GroupStats] = []
    for level in (NormalizationLevel.LIGHT, NormalizationLevel.MEDIUM):
        for group_name, sentences in text_groups.items():
            for counter_name, counter in active_counters.items():
                original, optimized = compute_fertility_series(sentences, counter, level)
                if not original or not optimized:
                    continue
                label = f"{group_name} / {counter_name} / {level.value}"
                all_stats.append(summarize_group(label, original, optimized))

    table = render_markdown_table(all_stats)
    counters_summary = "\n".join(f"- {name}: {note}" for name, (_, note) in counters.items())

    output = (
        "# Real-world multi-provider fertility benchmark\n\n"
        "Data: FLORES-200 devtest (professionally translated, CC BY-SA 4.0) "
        "+ this project's own corpus. Both normalization levels tested, "
        "pre-registered (not selected after seeing results).\n\n"
        f"## Counters used\n\n{counters_summary}\n\n"
        f"## Results\n\n{table}\n"
    )

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(output, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())