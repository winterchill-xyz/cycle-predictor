#!/usr/bin/env python3
"""Export the shipped model parameters to artifacts/model.json.

model.json is the single source of truth every language port loads. By default it
writes the canonical baked-in params (cycle_predictor.portable.DEFAULT_PARAMS). With
--refit it re-fits from local data (FedCycle backbone + mcPHASES luteal) and prints
the params so you can update the constants deliberately.

    .venv/bin/python scripts/export_model.py            # write canonical params
    .venv/bin/python scripts/export_model.py --refit    # print a fresh fit
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cycle_predictor import portable  # noqa: E402

OUT = ROOT / "artifacts" / "model.json"


def refit() -> dict:
    from cycle_predictor.data import fedcycle, mcphases
    from cycle_predictor.eval import user_sequences
    from cycle_predictor.models.generative import SkipAwareGenPoisson
    from cycle_predictor.models.twophase import TwoPhaseModel

    b = SkipAwareGenPoisson.fit_moments(user_sequences(fedcycle.load()))
    tp = TwoPhaseModel.fit(mcphases.load(), confirmed_only=True)
    params = json.loads(json.dumps(portable.DEFAULT_PARAMS))  # deep copy
    params["backbone"].update(mu_log=round(b.mu_log, 4), tau=round(b.tau, 4),
                              xi=round(b.xi, 4))
    params["twophase"] = {"luteal_mean": round(tp.luteal_mean, 2),
                          "luteal_sd": round(tp.luteal_sd, 2)}
    return params


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refit", action="store_true")
    args = ap.parse_args()

    if args.refit:
        print(json.dumps(refit(), indent=2))
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(portable.DEFAULT_PARAMS, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} (version {portable.DEFAULT_PARAMS['version']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
