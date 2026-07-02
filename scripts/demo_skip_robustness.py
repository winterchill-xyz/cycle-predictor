#!/usr/bin/env python3
"""Demonstrate WHERE the skip-aware generative model (v2) wins (PLAN.md M3 v2).

FedCycle is too clean to separate v2 from v1, so we inject the very artifact v2 is
built for: a "forgotten log" that merges adjacent cycles into one long observed
cycle. We corrupt each held-out user's history at increasing skip rates and measure
how well each predictor still recovers the user's next *true* cycle length.

Models are fit on the clean training split; only the test users' histories are
corrupted (a prediction-time robustness test). pi encodes the prior belief that
skips are possible, so v2 keeps a nonzero skip rate even though train data is clean.

    .venv/bin/python scripts/demo_skip_robustness.py
"""
from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cycle_predictor.data import fedcycle                       # noqa: E402
from cycle_predictor.eval import split, user_sequences           # noqa: E402
from cycle_predictor.models.hierarchical import HierarchicalBayes    # noqa: E402
from cycle_predictor.models.generative import SkipAwareGenerative    # noqa: E402


def corrupt(seq, p, rng):
    """Merge each cycle into a run of following cycles with prob p (a skipped log)."""
    out, i = [], 0
    while i < len(seq):
        val = seq[i]; i += 1
        while i < len(seq) and rng.random() < p:
            val += seq[i]; i += 1
        out.append(val)
    return out


def mae_under_corruption(predict, test, p, seed):
    """MAE of predicting each user's true last cycle from a corrupted prior history."""
    rng = random.Random(seed)
    errs = []
    for seq in test:
        if len(seq) < 4:
            continue
        target = seq[-1]                       # true next cycle (uncorrupted)
        hist = corrupt(seq[:-1], p, rng)       # corrupted history the model sees
        errs.append(abs(predict(hist) - target))
    return statistics.fmean(errs), len(errs)


def main() -> int:
    seqs = user_sequences(fedcycle.load())
    train, test = split(seqs, 0.7, seed=0)

    # Fit population hyperparameters on the CLEAN training split (fast moment fits).
    v1 = HierarchicalBayes.fit_moments(train)
    v2 = SkipAwareGenerative.fit_moments(train, pi=0.05)   # pi = "skips are possible"
    predictors = {
        "personal_mean": lambda h: statistics.fmean(h) if h else 29.0,
        "hierarchical_v1": v1.point,
        "skip_generative_v2": v2.point,
    }

    print("Recover the next TRUE cycle length from a history corrupted with skips.")
    print("MAE (days) vs injected skip rate p — lower is better:\n")
    header = f"{'skip rate p':>11s} | " + " ".join(f"{n:>18s}" for n in predictors)
    print(header); print("-" * len(header))
    for p in (0.0, 0.05, 0.10, 0.20, 0.30):
        cells = []
        for fn in predictors.values():
            mae, n = mae_under_corruption(fn, test, p, seed=100)
            cells.append(f"{mae:18.2f}")
        print(f"{p:11.2f} | " + " ".join(cells))

    print("\nAs p rises, naive/v1 estimates get dragged up by merged cycles; v2 explains")
    print("them as skipped logs and stays closer to the true rate. That gap is the point.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
