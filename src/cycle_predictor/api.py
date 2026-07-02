"""Product API: predict_next_period(user_log) -> date + calibrated interval (M6).

Turns a user's log (period dates, optionally LH tests and/or wearable temperature)
into a predicted next-period **date** and a **date interval** at a confidence level.
Works for a brand-new user (falls back to baked-in population priors) and sharpens
automatically when the current cycle has an LH surge or a wearable thermal shift —
see `models.unified.UnifiedPredictor`.

    from cycle_predictor.api import predict_next_period, UserLog
    f = predict_next_period(UserLog(period_starts=["2026-05-04", "2026-06-01"]))
    print(f)  # Next period ~2026-06-29 (80% window 06-26…07-03), via history
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from statistics import NormalDist

from .models.generative import SkipAwareGenPoisson
from .models.twophase import TwoPhaseModel
from .models.unified import UnifiedPredictor

# Population defaults (fit offline: backbone on FedCycle, luteal on mcPHASES surge
# cycles). Used when a caller doesn't pass a model, so the API works with no setup.
_BACKBONE_DEFAULT = dict(mu_log=4.0438, tau=0.1001, pi=0.05, xi=-0.9309)
_LUTEAL_DEFAULT = dict(luteal_mean=13.85, luteal_sd=3.08, n=74)

_DEFAULT_MODEL: UnifiedPredictor | None = None


def default_model() -> UnifiedPredictor:
    """A UnifiedPredictor with population-default hyperparameters (cached)."""
    global _DEFAULT_MODEL
    if _DEFAULT_MODEL is None:
        _DEFAULT_MODEL = UnifiedPredictor(
            backbone=SkipAwareGenPoisson(**_BACKBONE_DEFAULT),
            twophase=TwoPhaseModel(**_LUTEAL_DEFAULT),
        )
    return _DEFAULT_MODEL


def _as_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return datetime.strptime(str(d), "%Y-%m-%d").date()


def _by_day_of_cycle(mapping, onset: date) -> dict[int, float] | None:
    """Convert {date: value} into {day_of_cycle: value} relative to cycle onset."""
    if not mapping:
        return None
    out = {}
    for d, v in mapping.items():
        offset = (_as_date(d) - onset).days
        if offset >= 0 and v is not None:
            out[offset] = float(v)
    return out or None


@dataclass
class UserLog:
    """What a user has logged. Only `period_starts` is required."""
    period_starts: list          # dates (date or 'YYYY-MM-DD') a period began
    lh_tests: dict = None         # {date: LH value} this/most-recent cycle (optional)
    wearable_temp: dict = None    # {date: nightly temperature} (optional)
    today: object = None          # defaults to the last period start


@dataclass
class PeriodForecast:
    predicted_start: date
    earliest: date
    latest: date
    confidence: float
    days_until: int               # from `today` to predicted_start
    cycle_length_days: float
    sd_days: float
    mode: str                     # history | two_phase_lh | two_phase_wearable
    n_history: int                # number of prior cycles used

    def __str__(self) -> str:
        window = f"{self.earliest:%b %d}…{self.latest:%b %d}"
        return (f"Next period ~{self.predicted_start:%Y-%m-%d} "
                f"(in {self.days_until}d; {self.confidence:.0%} window {window}) "
                f"— via {self.mode}, {self.n_history} prior cycle(s)")


def predict_next_period(user_log: UserLog, model: UnifiedPredictor | None = None,
                        confidence: float = 0.8) -> PeriodForecast:
    """Predict the next period start date and a calibrated date interval."""
    model = model or default_model()
    starts = sorted(_as_date(d) for d in user_log.period_starts)
    if not starts:
        raise ValueError("need at least one period start date to anchor the prediction")
    onset = starts[-1]                                   # current cycle began here
    history = [(starts[i + 1] - starts[i]).days for i in range(len(starts) - 1)]
    today = _as_date(user_log.today) if user_log.today else onset

    pred = model.predict(
        history,
        lh_by_day=_by_day_of_cycle(user_log.lh_tests, onset),
        temp_by_day=_by_day_of_cycle(user_log.wearable_temp, onset),
    )
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    half = z * pred.sd
    center = onset + timedelta(days=round(pred.cycle_length))
    return PeriodForecast(
        predicted_start=center,
        earliest=onset + timedelta(days=round(pred.cycle_length - half)),
        latest=onset + timedelta(days=round(pred.cycle_length + half)),
        confidence=confidence,
        days_until=(center - today).days,
        cycle_length_days=round(pred.cycle_length, 1),
        sd_days=round(pred.sd, 1),
        mode=pred.mode,
        n_history=len(history),
    )
