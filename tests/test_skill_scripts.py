"""Unit tests for the standalone SKILL.md CLI wrappers.

These scripts live under skill/scripts/ (not the installed package), so
they're loaded dynamically via importlib rather than a normal import.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SKILL_SCRIPTS_DIR = Path(__file__).parent.parent / "skill" / "scripts"


def _load_module(filename: str):
    """Dynamically load a standalone script as an importable module."""
    module_path = SKILL_SCRIPTS_DIR / filename
    spec = importlib.util.spec_from_file_location(filename, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


normalize_cli = _load_module("normalize_cli.py")
report_cli = _load_module("report_cli.py")


# --- normalize_cli.py ------------------------------------------------------


def test_normalize_cli_reads_file_and_prints_result(tmp_path: Path, capsys) -> None:
    input_file = tmp_path / "in.txt"
    input_file.write_text("مَرْحَـبًا", encoding="utf-8")

    exit_code = normalize_cli.main(["--input", str(input_file), "--level", "light"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.strip() == "مرحبا"


def test_normalize_cli_reads_stdin(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(sys, "stdin", __import__("io").StringIO("مَرْحَـبًا"))

    exit_code = normalize_cli.main(["--level", "light"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.strip() == "مرحبا"


def test_normalize_cli_missing_file_returns_error_code(capsys) -> None:
    exit_code = normalize_cli.main(["--input", "does/not/exist.txt"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error" in captured.err


def test_normalize_cli_oversized_file_rejected(tmp_path: Path, monkeypatch, capsys) -> None:
    input_file = tmp_path / "in.txt"
    input_file.write_text("نص", encoding="utf-8")

    # Simulate an oversized file without actually allocating 20MB on disk.
    monkeypatch.setattr(normalize_cli, "MAX_FILE_BYTES", 1)

    exit_code = normalize_cli.main(["--input", str(input_file)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "exceeding" in captured.err


def test_normalize_cli_invalid_utf8_gives_clear_error(tmp_path: Path, capsys) -> None:
    input_file = tmp_path / "bad_encoding.txt"
    input_file.write_bytes(b"\xff\xfe\x00invalid")

    exit_code = normalize_cli.main(["--input", str(input_file)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "not valid UTF-8" in captured.err


# --- report_cli.py -----------------------------------------------------


def test_report_cli_prints_valid_json(tmp_path: Path, capsys) -> None:
    pytest.importorskip("tiktoken", reason="optional dependency not installed")

    input_file = tmp_path / "in.txt"
    input_file.write_text("مَرْحَـبًا بكم اليوم", encoding="utf-8")

    exit_code = report_cli.main(["--input", str(input_file), "--level", "light"])
    captured = capsys.readouterr()

    assert exit_code == 0
    report = json.loads(captured.out)
    assert "tokens_saved" in report
    assert "percent_saved" in report


def test_report_cli_without_tiktoken_gives_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    import builtins

    input_file = tmp_path / "in.txt"
    input_file.write_text("نص عربي", encoding="utf-8")

    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name == "tiktoken":
            raise ImportError("simulated missing tiktoken")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    exit_code = report_cli.main(["--input", str(input_file)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "tokenizers" in captured.err


def test_report_cli_missing_file_returns_error_code(capsys) -> None:
    exit_code = report_cli.main(["--input", "does/not/exist.txt"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error" in captured.err


def test_report_cli_invalid_utf8_gives_clear_error(tmp_path: Path, capsys) -> None:
    input_file = tmp_path / "bad_encoding.txt"
    input_file.write_bytes(b"\xff\xfe\x00invalid")

    exit_code = report_cli.main(["--input", str(input_file)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "not valid UTF-8" in captured.err