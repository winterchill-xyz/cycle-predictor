#!/usr/bin/env python3
"""Compare baselines vs the hierarchical Bayesian model on FedCycle (PLAN.md M2/M3).

User-holdout split (test users unseen at fit time). Reports overall point accuracy,
a cold-start slice (predictions made from <=2 prior cycles), and — for the Bayesian
model — calibration of its predictive intervals, which is the property it's meant
to win on.

    .venv/bin/python scripts/eval_models.py            # PyMC fit
    .venv/bin/python scripts/eval_models.py --fast     # method-of-moments fit
"""
from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cycle_predictor.data import fedcycle                    # noqa: E402
from cycle_predictor.eval import (                            # noqa: E402
    evaluate, evaluate_prob, split, user_sequences,
)
from cycle_predictor.models import baselines                  # noqa: E402
from cycle_predictor.models.hierarchical import HierarchicalBayes  # noqa: E402
from cycle_predictor.models.generative import SkipAwareGenerative  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true", help="method-of-moments fit (skip PyMC)")
    ap.add_argument("--frac-train", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seqs = user_sequences(fedcycle.load())
    train, test = split(seqs, args.frac_train, seed=args.seed)
    print(f"FedCycle: {len(seqs)} users → {len(train)} train / {len(test)} test "
          f"(held-out). Test cycles: {sum(len(s) for s in test)}\n")

    if args.fast:
        model = HierarchicalBayes.fit_moments(train)
        gen = SkipAwareGenerative.fit_moments(train, pi=0.05)
    else:
        print("fitting hierarchical (v1) and skip-aware generative (v2) with PyMC…")
        model = HierarchicalBayes.fit(train, draws=1000, tune=1000, chains=2, seed=args.seed)
        gen = SkipAwareGenerative.fit(train, draws=1000, tune=1000, chains=2, seed=args.seed)
    print(f"v1 [{model.diagnostics.get('method')}]: mu_pop={model.mu_pop:.2f}  "
          f"tau={model.tau2**0.5:.2f}  sigma={model.sigma2**0.5:.2f}")
    print(f"v2 [{gen.diagnostics.get('method')}]: rate={math.exp(gen.mu_log):.2f}  "
          f"pi={gen.pi:.3f} (skips/cycle={gen.expected_skip_rate():.3f})\n")

    # ---- overall point accuracy on held-out users -------------------------------
    predictors = dict(baselines.registry())
    predictors["hierarchical_v1"] = model.point
    predictors["skip_generative_v2"] = gen.point

    hdr = f"{'predictor':20s} {'MAE':>6s} {'RMSE':>6s} {'±1d':>6s} {'±2d':>6s} {'coldMAE':>8s}"
    print(hdr); print("-" * len(hdr))
    for name, fn in predictors.items():
        m = evaluate(fn, test)
        cold = evaluate(fn, test, max_history=2)   # predictions from <=2 prior cycles
        print(f"{name:20s} {m.mae:6.2f} {m.rmse:6.2f} {m.within1:5.0%} {m.within2:5.0%} "
              f"{cold.mae:8.2f}")

    # ---- calibration of the two Bayesian models ---------------------------------
    print("\ncalibration:")
    print(f"  v1 hierarchical:    {evaluate_prob(model.predict, test)}")
    print(f"  v2 skip-generative: {evaluate_prob(gen.predict, test)}")
    print("  (well-calibrated ⇒ e.g. the 80% interval covers ~80% of held-out cycles)")
    print("\nNote: FedCycle has ~no skip artifacts, so v2≈v1 here. "
          "See scripts/demo_skip_robustness.py for where v2 wins.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
