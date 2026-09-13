"""Tests for paired bootstrap inference used by benchmark tracks."""

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parent.parent / "benchmark" / "_paired_inference.py"
SPEC = importlib.util.spec_from_file_location("_paired_inference", MODULE_PATH)
paired = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(paired)


def test_paired_difference_is_treatment_minus_baseline() -> None:
    estimate, low, high = paired.paired_mean_difference_ci95(
        [0.2, 0.4], [0.3, 0.5], resamples=100
    )
    assert estimate == pytest.approx(0.1)
    assert low == pytest.approx(0.1)
    assert high == pytest.approx(0.1)


def test_paired_bootstrap_is_reproducible_and_contains_estimate() -> None:
    first = paired.paired_mean_difference_ci95([0.0, 1.0, 0.5], [0.5, 0.5, 1.0])
    second = paired.paired_mean_difference_ci95([0.0, 1.0, 0.5], [0.5, 0.5, 1.0])
    assert first == second
    assert first[1] <= first[0] <= first[2]


def test_paired_bootstrap_rejects_unpaired_input() -> None:
    with pytest.raises(ValueError, match="same length"):
        paired.paired_mean_difference_ci95([0.0], [0.0, 1.0])
