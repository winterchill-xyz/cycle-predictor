"""Unified next-period predictor with graceful degradation (production shape).

Real users differ in what they log. Most have **only past period dates**; some also
have **LH tests** (e.g. Mira) or a **wearable temperature** stream. This predictor
always works from cycle-length history and *opportunistically* sharpens when
mid-cycle signals resolve ovulation — never requiring a signal that may be absent.

Evidence ladder (best first), for predicting the current cycle's length:
  1. LH surge observed this cycle       → two-phase (ovulation + stable luteal)
  2. wearable thermal shift observed     → two-phase (noisier ovulation estimate)
  3. neither                             → generative backbone (history only)

The backbone is the calibrated, cold-start-robust Generalized-Poisson model (v2.1);
the sharpener is the two-phase model (M5). Every prediction reports its `mode`.
"""
from __future__ import annotations

from dataclasses import dataclass

from .generative import SkipAwareGenPoisson
from .ovulation import detect_thermal_shift
from .twophase import TwoPhaseModel


@dataclass
class Prediction:
    cycle_length: float     # predicted length of the current cycle (days from onset)
    sd: float               # predictive sd (days)
    mode: str               # "history" | "two_phase_lh" | "two_phase_wearable"


@dataclass
class UnifiedPredictor:
    backbone: object            # any .predict(history) -> (mean, sd); default SkipAwareGenPoisson
    twophase: TwoPhaseModel
    lh_surge: float = 25.0

    @classmethod
    def fit(cls, sequences, cycles, *, lh_surge: float = 25.0) -> "UnifiedPredictor":
        """Fit the history backbone on cycle-length sequences and the two-phase
        sharpener on cycles that carry a (confirmed) ovulation estimate."""
        return cls(
            backbone=SkipAwareGenPoisson.fit_moments(sequences),
            twophase=TwoPhaseModel.fit(cycles, confirmed_only=True),
            lh_surge=lh_surge,
        )

    def _ovulation_from_signals(self, lh_by_day, temp_by_day):
        if lh_by_day:
            surge = sorted(d for d, v in lh_by_day.items() if v is not None and v >= self.lh_surge)
            if surge:
                return surge[0], "two_phase_lh"
        if temp_by_day:
            est = detect_thermal_shift(temp_by_day)
            if est is not None:
                return est, "two_phase_wearable"
        return None, None

    def predict(self, history, *, lh_by_day=None, temp_by_day=None) -> Prediction:
        """Predict the current cycle length. `history` = past cycle lengths (may be
        empty for a brand-new user). `lh_by_day`/`temp_by_day` are optional
        {day_of_cycle: value} series observed so far this cycle."""
        ovulation, mode = self._ovulation_from_signals(lh_by_day, temp_by_day)
        if ovulation is not None:
            mean, sd = self.twophase.predict_from_ovulation(ovulation)
            return Prediction(mean, sd, mode)
        mean, sd = self.backbone.predict(history)
        return Prediction(mean, sd, "history")
