"""Dependency-free paired inference helpers for benchmark scripts.

The before/after values in the evaluation tracks come from the same example.
Inference therefore operates on their per-example differences, not on two
independent confidence intervals.
"""

from __future__ import annotations

import random
import statistics


def mean_and_ci95(values: list[float]) -> tuple[float, float, float]:
    """Return a descriptive normal-approximation CI for one series."""
    if not values:
        raise ValueError("mean_and_ci95() needs at least one value")
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, mean, mean
    margin = 1.96 * statistics.stdev(values) / (len(values) ** 0.5)
    return mean, mean - margin, mean + margin


def _percentile(sorted_values: list[float], probability: float) -> float:
    """Linear-interpolated percentile without a NumPy dependency."""
    if not sorted_values:
        raise ValueError("_percentile() needs at least one value")
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])


def paired_mean_difference_ci95(
    baseline: list[float],
    treatment: list[float],
    *,
    resamples: int = 10_000,
    seed: int = 20260913,
) -> tuple[float, float, float]:
    """Return mean(treatment - baseline) and a percentile bootstrap 95% CI.

    Pairs are resampled together, preserving the covariance that separate
    condition-level intervals discard. A fixed seed makes reports
    reproducible.
    """
    if len(baseline) != len(treatment):
        raise ValueError("Paired inputs must have the same length")
    if not baseline:
        raise ValueError("paired_mean_difference_ci95() needs at least one pair")
    if resamples < 1:
        raise ValueError("resamples must be at least 1")

    differences = [after - before for before, after in zip(baseline, treatment)]
    estimate = statistics.mean(differences)
    if len(differences) == 1 or len(set(differences)) == 1:
        return estimate, estimate, estimate

    generator = random.Random(seed)
    n = len(differences)
    bootstrap_means = sorted(
        sum(differences[generator.randrange(n)] for _ in range(n)) / n
        for _ in range(resamples)
    )
    return estimate, _percentile(bootstrap_means, 0.025), _percentile(bootstrap_means, 0.975)
