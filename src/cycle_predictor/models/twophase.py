"""Two-phase next-period model (PLAN.md M5).

A cycle splits into a variable **follicular** phase (menses onset → ovulation) and a
stable **luteal** phase (ovulation → next onset, ~14 days). So once ovulation is
observed mid-cycle (via the LH surge, or a wearable thermal shift), the next period
is well predicted by `ovulation_day + luteal_length`, and the predictive spread is
the luteal sd — tighter than the whole-cycle sd. This is the "sharpen as the cycle
progresses" behavior the literature reports.

`TwoPhaseModel.fit` learns the luteal length distribution from training cycles that
carry an ovulation estimate; `predict_from_ovulation` gives the next-cycle length
(and sd) once ovulation is known.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import Sequence


def _has_ovulation(c, confirmed_only: bool) -> bool:
    if c.estimated_ovulation_day is None or not c.cycle_length_days:
        return False
    return c.extra.get("ovulation_confirmed", True) if confirmed_only else True


def luteal_lengths(cycles, confirmed_only: bool = False) -> list[int]:
    """cycle_length - ovulation_day. With confirmed_only, keep only cycles whose
    ovulation came from a real LH surge (tight luteal), not the LH-peak fallback."""
    return [c.cycle_length_days - c.estimated_ovulation_day
            for c in cycles if _has_ovulation(c, confirmed_only)]


@dataclass
class TwoPhaseModel:
    luteal_mean: float
    luteal_sd: float
    n: int

    @classmethod
    def fit(cls, cycles, confirmed_only: bool = True) -> "TwoPhaseModel":
        lut = luteal_lengths(cycles, confirmed_only=confirmed_only)
        if not lut:
            raise ValueError("no cycles with an ovulation estimate")
        return cls(fmean(lut), pstdev(lut) or 1.0, len(lut))

    def predict_from_ovulation(self, ovulation_day: int) -> tuple[float, float]:
        """Predicted cycle length (and sd) given ovulation observed at `ovulation_day`
        (days from menses onset). Next period ≈ ovulation + a stable luteal phase."""
        return ovulation_day + self.luteal_mean, self.luteal_sd
