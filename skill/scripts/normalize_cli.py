"""CLI wrapper: normalize Arabic text conservatively for LLM token savings.

Reads text from --input <path> or stdin, writes normalized text to stdout.
Standalone script (not part of the installed package) intended to be
invoked by an agent via the bash tool as part of the arabic-token-optimizer
skill.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ar_tokenwise import NormalizationLevel, normalize

# Guard against reading an accidentally huge file into memory (size-based
# DoS guard), checked via file size before reading, not after.
MAX_FILE_BYTES = 20_000_000  # 20 MB


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
    parser = argparse.ArgumentParser(description="Normalize Arabic text.")
    parser.add_argument(
        "--input", default=None, help="Path to input text file. Reads stdin if omitted."
    )
    parser.add_argument(
        "--level",
        choices=[level.value for level in NormalizationLevel],
        default=NormalizationLevel.LIGHT.value,
        help="Normalization aggressiveness. Default: light.",
    )
    args = parser.parse_args(argv)

    try:
        text = _read_input(args.input)
    except UnicodeDecodeError as exc:
        # Caught separately from ValueError (its parent class) for a
        # clearer, non-technical error message.
        print(f"Error: input is not valid UTF-8 text: {exc}", file=sys.stderr)
        return 1
    except (FileNotFoundError, ValueError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        result = normalize(text, level=NormalizationLevel(args.level))
    except (ValueError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))