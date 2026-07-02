#!/usr/bin/env python3
"""Exploratory data analysis of FedCycle (PLAN.md M1) — pure stdlib.

Characterizes what a model must handle: cycle-length distribution, per-user
regularity, cold-start (short histories), and skip artifacts (a forgotten period
log shows up as a ~2x-length cycle). Prints a text report.

    PYTHONPATH=src python scripts/eda_fedcycle.py
"""
from __future__ import annotations

import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cycle_predictor.data import fedcycle          # noqa: E402
from cycle_predictor.eval import user_sequences     # noqa: E402

# A user is "irregular" if the spread of their cycle lengths is large. Clinically,
# cycle-to-cycle variation >7-9 days is considered irregular; we use within-user SD.
IRREGULAR_SD_DAYS = 7.0


def histogram(values, lo, hi, step, width=40):
    counts = Counter()
    for v in values:
        bucket = min(hi, max(lo, int(v // step) * step))
        counts[bucket] += 1
    peak = max(counts.values()) if counts else 1
    lines = []
    for b in range(lo, hi + step, step):
        n = counts.get(b, 0)
        bar = "#" * round(width * n / peak)
        lines.append(f"  {b:3d}-{b+step-1:<3d} | {bar} {n}")
    return "\n".join(lines)


def main() -> int:
    cycles = fedcycle.load()
    seqs = user_sequences(cycles)
    lengths = [x for s in seqs for x in s]

    print("=" * 60)
    print("FedCycle EDA")
    print("=" * 60)
    print(f"users: {len(seqs)}   cycles (with length): {len(lengths)}")
    print(f"cycle length: mean={statistics.mean(lengths):.1f}  "
          f"median={statistics.median(lengths)}  sd={statistics.pstdev(lengths):.1f}  "
          f"min={min(lengths)}  max={max(lengths)}")

    print("\n-- cycle length distribution (days) --")
    print(histogram(lengths, 15, 60, 5))

    # cycles per user → cold-start picture
    per_user = sorted(len(s) for s in seqs)
    print("\n-- cycles per user --")
    print(f"min={per_user[0]}  median={statistics.median(per_user)}  "
          f"max={per_user[-1]}  mean={statistics.mean(per_user):.1f}")
    short = sum(n <= 3 for n in per_user)
    print(f"users with <=3 cycles (cold-start-ish): {short} ({short/len(per_user):.0%})")

    # regularity
    sds = [statistics.pstdev(s) for s in seqs if len(s) >= 3]
    irregular = sum(sd > IRREGULAR_SD_DAYS for sd in sds)
    print("\n-- per-user regularity (users with >=3 cycles) --")
    print(f"within-user SD: median={statistics.median(sds):.1f}  "
          f"mean={statistics.mean(sds):.1f}  max={max(sds):.1f}")
    print(f"irregular (SD > {IRREGULAR_SD_DAYS:g}d): {irregular}/{len(sds)} "
          f"({irregular/len(sds):.0%})")

    # skip artifacts: a forgotten period doubles a cycle. Flag cycles that are
    # >1.6x the user's median (and absolutely long) — candidate merged/skipped cycles.
    suspected = 0
    for s in seqs:
        if len(s) < 3:
            continue
        med = statistics.median(s)
        suspected += sum(1 for x in s if x > 1.6 * med and x >= 40)
    print("\n-- suspected skip artifacts (>1.6x user median & >=40d) --")
    print(f"{suspected} cycles ({suspected/len(lengths):.1%} of all) — the generative")
    print("model (M3) should absorb these as skipped-cycle events, not real lengths.")

    print("\nTakeaways for modeling:")
    print("  * point MAE is easy here (regular NFP cohort); calibration is the goal.")
    print("  * a nontrivial cold-start tail (short histories) needs a population prior.")
    print("  * skip artifacts exist → prefer skip-aware / robust likelihoods.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
