"""Unit tests for benchmark/_tydiqa_loader.py.

Only the network-independent logic is tested here (the Arabic-ID
filter, the ImportError path). The actual field-parsing logic against
real TyDiQA rows is NOT verified by these tests -- it needs live
network access to huggingface.co that this sandbox doesn't have. See
_tydiqa_loader.py's module docstring for why, and what to do if the
schema doesn't match.
"""

import builtins
import importlib.util
from pathlib import Path

import pytest

BENCHMARK_DIR = Path(__file__).parent.parent / "benchmark"


def _load_module(filename: str):
    module_path = BENCHMARK_DIR / filename
    spec = importlib.util.spec_from_file_location(filename, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


loader = _load_module("_tydiqa_loader.py")


def test_is_arabic_example_matches_arabic_prefix() -> None:
    assert loader._is_arabic_example("arabic-2385726501324569680-0") is True


def test_is_arabic_example_case_insensitive() -> None:
    assert loader._is_arabic_example("Arabic-123") is True


def test_is_arabic_example_rejects_other_languages() -> None:
    assert loader._is_arabic_example("english-123") is False
    assert loader._is_arabic_example("japanese-456") is False


def test_load_goldp_raises_clear_error_without_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "datasets":
            raise ImportError("simulated missing datasets")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="pip install datasets"):
        loader.load_tydiqa_goldp_arabic()


def test_load_primary_raises_clear_error_without_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "datasets":
            raise ImportError("simulated missing datasets")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="pip install datasets"):
        loader.load_tydiqa_primary_arabic()