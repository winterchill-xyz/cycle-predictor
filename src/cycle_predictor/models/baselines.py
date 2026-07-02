"""Baseline predictors: `history -> predicted next cycle length` (days).

These are the numbers the Bayesian generative model (PLAN.md M3) must beat — on
*calibration*, since the literature shows point MAE saturates across model families.
`DEFAULT_CYCLE` is the fallback when a user has no history (cold-start).
"""
from __future__ import annotations

import statistics
from typing import Sequence

DEFAULT_CYCLE = 28


def constant(k: int = DEFAULT_CYCLE):
    def predict(history: Sequence[int]) -> float:
        return float(k)
    predict.__name__ = f"constant_{k}"
    return predict


def last_cycle(history: Sequence[int]) -> float:
    """Predict the most recent observed length (a random-walk / persistence model)."""
    return float(history[-1]) if history else float(DEFAULT_CYCLE)


def personal_mean(history: Sequence[int]) -> float:
    return statistics.fmean(history) if history else float(DEFAULT_CYCLE)


def personal_median(history: Sequence[int]) -> float:
    return float(statistics.median(history)) if history else float(DEFAULT_CYCLE)


def rolling_mean(k: int = 3):
    def predict(history: Sequence[int]) -> float:
        window = history[-k:]
        return statistics.fmean(window) if window else float(DEFAULT_CYCLE)
    predict.__name__ = f"rolling_mean_{k}"
    return predict


def registry() -> dict:
    """Named baseline predictors for the eval harness."""
    return {
        "constant_28": constant(28),
        "last_cycle": last_cycle,
        "personal_mean": personal_mean,
        "personal_median": personal_median,
        "rolling_mean_3": rolling_mean(3),
    }
