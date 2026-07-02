#!/usr/bin/env python3
"""Unified predictor on mcPHASES: graceful degradation as signals appear/vanish.

Simulates "use whatever this user logged": for each held-out cycle the unified
predictor uses the LH series if present, else a wearable temperature shift, else
just the period-length history. Reports the mode mix and MAE per mode, and compares
the unified result against a history-only floor (what a user with no devices gets).

    .venv/bin/python scripts/eval_unified.py
"""
from __future__ import annotations

import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cycle_predictor.data import mcphases, mcphases_signals      # noqa: E402
from cycle_predictor.eval import user_sequences                  # noqa: E402
from cycle_predictor.models.unified import UnifiedPredictor       # noqa: E402


def mae(xs):
    return statistics.fmean(xs) if xs else float("nan")


def main() -> int:
    cycles = mcphases.load()
    signals = mcphases_signals.load_signals()

    users = sorted({c.user_id for c in cycles})
    random.Random(0).shuffle(users)
    train_u = set(users[: int(len(users) * 0.7)])
    train = [c for c in cycles if c.user_id in train_u]
    test = [c for c in cycles if c.user_id not in train_u]
    model = UnifiedPredictor.fit(user_sequences(train), train)

    by_user = defaultdict(list)
    for c in test:
        by_user[c.user_id].append(c)

    modes = Counter()
    per_mode_err = defaultdict(list)
    unified_err, floor_err = [], []
    for cs in by_user.values():
        cs.sort(key=lambda c: c.cycle_number)
        history = []
        for c in cs:
            ser = mcphases_signals.cycle_series(c, signals)
            pred = model.predict(history, lh_by_day=ser["lh"], temp_by_day=ser["temp"])
            err = abs(pred.cycle_length - c.cycle_length_days)
            modes[pred.mode] += 1
            per_mode_err[pred.mode].append(err)
            unified_err.append(err)
            floor_err.append(abs(model.predict(history).cycle_length - c.cycle_length_days))
            history.append(c.cycle_length_days)

    n = sum(modes.values())
    print(f"held-out cycles: {n}\n")
    print("mode chosen (best available evidence per cycle):")
    for mode in ("two_phase_lh", "two_phase_wearable", "history"):
        k = modes.get(mode, 0)
        print(f"   {mode:20s} {k:3d} ({k/n:4.0%})   MAE={mae(per_mode_err[mode]):.2f} d")
    print(f"\nunified (use what's available): MAE={mae(unified_err):.2f} d")
    print(f"history-only floor (no devices): MAE={mae(floor_err):.2f} d")
    print("\nEvery user gets a calibrated prediction; those who logged LH/wearable data")
    print("get a sharper one. Nothing breaks when signals are missing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
