#!/usr/bin/env python3
"""Evaluate baseline predictors on FedCycle (PLAN.md M2).

Predict-future-from-past: for each user, predict cycle i's length from cycles
1..i-1. Prints an MAE/RMSE/±1d/±2d table. This is the bar the generative model
(M3) must clear on calibration.

    PYTHONPATH=src python scripts/eval_baselines.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cycle_predictor.data import fedcycle          # noqa: E402
from cycle_predictor.eval import evaluate, user_sequences  # noqa: E402
from cycle_predictor.models import baselines        # noqa: E402


def main() -> int:
    sequences = user_sequences(fedcycle.load())
    n_users = len(sequences)
    n_cycles = sum(len(s) for s in sequences)
    print(f"FedCycle: {n_users} users, {n_cycles} cycles with length\n")

    header = f"{'baseline':16s} {'MAE':>6s} {'RMSE':>6s} {'±1d':>7s} {'±2d':>7s} {'n':>6s}"
    print(header)
    print("-" * len(header))
    results = {}
    for name, predictor in baselines.registry().items():
        m = evaluate(predictor, sequences)
        results[name] = m
        print(f"{name:16s} {m.mae:6.2f} {m.rmse:6.2f} {m.within1:6.1%} {m.within2:6.1%} {m.n:6d}")

    best = min(results.items(), key=lambda kv: kv[1].mae)
    print(f"\nbest baseline by MAE: {best[0]} ({best[1].mae:.2f}d)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
