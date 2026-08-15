"""CLI wrapper: report token savings from normalizing Arabic text.

Reads original text from --input <path> or stdin, normalizes it internally,
and prints a JSON token-savings report to stdout. Standalone script for the
arabic-token-optimizer skill, invoked via the bash tool.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from ar_tokenwise import NormalizationLevel, normalize, report_savings
from ar_tokenwise.report import get_default_counter

MAX_FILE_BYTES = 20_000_000  # 20 MB, same guard as normalize_cli.py


def _read_input(input_path: str | None) -> str:
    """Read text from a file path or stdin, with a size guard on files."""
    if input_path is None:
        return sys.stdin.read()

    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ValueError(
            f"Input file is {size} bytes, exceeding the {MAX_FILE_BYTES} "
            "byte limit for this CLI. Split the file first."
        )

    return path.read_text(encoding="utf-8")


def main(argv: list[str]) -> int:
    """Entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        description="Report token savings from normalizing Arabic text."
    )
    parser.add_argument(
        "--input", default=None, help="Path to input text file. Reads stdin if omitted."
    )
    parser.add_argument(
        "--level",
        choices=[level.value for level in NormalizationLevel],
        default=NormalizationLevel.LIGHT.value,
        help="Normalization aggressiveness. Default: light.",
    )
    parser.add_argument(
        "--cost-per-million",
        type=float,
        default=None,
        help="Optional model price (USD per 1M tokens) to estimate dollar savings.",
    )
    args = parser.parse_args(argv)

    try:
        original = _read_input(args.input)
    except UnicodeDecodeError as exc:
        print(f"Error: input is not valid UTF-8 text: {exc}", file=sys.stderr)
        return 1
    except (FileNotFoundError, ValueError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        optimized = normalize(original, level=NormalizationLevel(args.level))
        counter = get_default_counter()
        report = report_savings(
            original,
            optimized,
            counter=counter,
            cost_per_million_tokens=args.cost_per_million,
        )
    except (ValueError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        # Raised by get_default_counter() when tiktoken isn't installed.
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))