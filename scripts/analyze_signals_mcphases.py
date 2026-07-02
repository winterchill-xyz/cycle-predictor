#!/usr/bin/env python3
"""M5 signal-rich model on mcPHASES: ovulation detection + two-phase next-period.

Three results:
  1. Luteal stability — the luteal phase is tighter than the whole cycle, so knowing
     ovulation helps.
  2. Wearable thermal-shift ovulation detector vs the LH-surge ground truth (MAE, in
     days) — a non-invasive ovulation estimate from nightly skin temperature alone.
  3. Two-phase next-period model — once ovulation is observed, predict next period as
     ovulation + luteal length. Compared with (a) no ovulation info and (b) ovulation
     from the wearable detector (fully non-invasive).

    .venv/bin/python scripts/analyze_signals_mcphases.py
"""
from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cycle_predictor.data import mcphases, mcphases_signals          # noqa: E402
from cycle_predictor.models.ovulation import detect_thermal_shift     # noqa: E402
from cycle_predictor.models.twophase import TwoPhaseModel, luteal_lengths  # noqa: E402


def mae(errs):
    return statistics.fmean(errs) if errs else float("nan")


def main() -> int:
    cycles = mcphases.load()
    signals = mcphases_signals.load_signals()
    ov_cycles = [c for c in cycles if c.estimated_ovulation_day is not None]

    # 1. luteal stability (surge-confirmed cycles) ----------------------------
    confirmed = [c for c in ov_cycles if c.extra.get("ovulation_confirmed")]
    lut = luteal_lengths(confirmed, confirmed_only=True)
    clen = [c.cycle_length_days for c in confirmed]
    print("1. Phase stability (n=%d LH-surge-confirmed cycles):" % len(confirmed))
    print(f"   whole cycle length: mean={statistics.fmean(clen):.1f}  sd={statistics.pstdev(clen):.2f}")
    print(f"   luteal length:      mean={statistics.fmean(lut):.1f}  sd={statistics.pstdev(lut):.2f}"
          "   ← tighter ⇒ ovulation is informative\n")

    # 2. wearable thermal-shift detector vs LH-surge ovulation ----------------
    errs, fired = [], 0
    for c in ov_cycles:
        temp = mcphases_signals.cycle_series(c, signals)["temp"]
        est = detect_thermal_shift(temp)
        if est is not None:
            fired += 1
            errs.append(abs(est - c.estimated_ovulation_day))
    print("2. Wearable thermal-shift ovulation detector (nightly skin temp, no LH):")
    print(f"   fired on {fired}/{len(ov_cycles)} cycles; MAE vs LH-surge ovulation = {mae(errs):.1f} d\n")

    # 3. two-phase next-period model (user-holdout) ---------------------------
    users = sorted({c.user_id for c in cycles})
    random.Random(0).shuffle(users)
    train_u = set(users[: int(len(users) * 0.7)])
    train = [c for c in cycles if c.user_id in train_u]
    test = [c for c in cycles if c.user_id not in train_u]
    tp = TwoPhaseModel.fit(train, confirmed_only=True)
    pop_mean = statistics.fmean([c.cycle_length_days for c in train])
    print(f"3. Two-phase next-period (luteal={tp.luteal_mean:.1f}±{tp.luteal_sd:.1f} d from "
          f"{tp.n} confirmed train cycles; test on held-out users):")

    base, lh_tp, wear_tp = [], [], []
    for c in test:
        actual = c.cycle_length_days
        if c.estimated_ovulation_day is not None and c.extra.get("ovulation_confirmed"):
            base.append(abs(pop_mean - actual))                               # no ovulation info
            lh_tp.append(abs(tp.predict_from_ovulation(c.estimated_ovulation_day)[0] - actual))
        est = detect_thermal_shift(mcphases_signals.cycle_series(c, signals)["temp"])
        if est is not None:
            wear_tp.append(abs(tp.predict_from_ovulation(est)[0] - actual))
    print(f"   baseline (population mean, no ovulation):   MAE={mae(base):.2f} d  (n={len(base)})")
    print(f"   two-phase (LH-surge ovulation observed):    MAE={mae(lh_tp):.2f} d  (n={len(lh_tp)})")
    print(f"   two-phase (wearable thermal-shift, no LH):  MAE={mae(wear_tp):.2f} d  (n={len(wear_tp)})")
    print("\n   Once ovulation is known mid-cycle, next-period error drops toward the luteal")
    print("   sd — the signal-rich model sharpens the forecast a whole cycle wouldn't.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
