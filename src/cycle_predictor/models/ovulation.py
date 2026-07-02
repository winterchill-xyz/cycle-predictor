"""Wearable ovulation detection (PLAN.md M5) — the non-invasive predictor.

After ovulation, progesterone raises basal body temperature; in mcPHASES nightly
skin temperature runs ~0.3°C warmer in the luteal phase. The classic "coverline"
rule finds the sustained thermal shift: the first day whose temperature (and the
next few days) sits clearly above the preceding baseline. Ovulation is estimated as
the day just before that shift — using only a wearable, no hormone test.
"""
from __future__ import annotations

from statistics import fmean


def detect_thermal_shift(temp_by_day: dict[int, float], *, baseline: int = 6,
                         run: int = 3, delta: float = 0.15,
                         search: tuple[int, int] = (5, 22)) -> int | None:
    """Estimate ovulation day-of-cycle from a {day_of_cycle: nightly_temp} series.

    Rule: the first candidate day d in `search` such that the next `run` days are
    all above (mean of the prior `baseline` days + `delta` °C). Ovulation ≈ d - 1.
    Returns None if no clear shift (missing data, anovulatory, or flat).
    """
    if len(temp_by_day) < baseline + run:
        return None
    lo, hi = search
    for d in range(lo, hi + 1):
        base = [temp_by_day[k] for k in range(d - baseline, d) if k in temp_by_day]
        fut = [temp_by_day[d + j] for j in range(run) if (d + j) in temp_by_day]
        if len(base) < baseline - 1 or len(fut) < run:
            continue
        coverline = fmean(base) + delta
        if all(t > coverline for t in fut):
            return max(d - 1, 0)
    return None
