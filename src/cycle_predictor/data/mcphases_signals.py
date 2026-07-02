"""mcPHASES daily wearable signals, aligned to cycles (for the M5 signal-rich model).

Loads the daily biosignals that carry the ovulation/luteal information — nightly
skin temperature (the BBT analog; luteal phase runs ~0.3°C warmer), resting heart
rate, and sleep respiratory rate — keyed by (subject, interval, day_in_study), and
provides per-cycle series indexed by day-of-cycle (0 = menses onset).

Requires the mcPHASES CSVs in data/raw/mcphases/ (credentialed; gitignored).
"""
from __future__ import annotations

import csv
from pathlib import Path

from .mcphases import DEFAULT_DIR


def _load_daily(path: Path, day_col: str, val_col: str, keep=lambda r: True
                ) -> dict[tuple[str, str, int], float]:
    out: dict[tuple[str, str, int], float] = {}
    if not path.exists():
        return out
    for r in csv.DictReader(open(path, newline="")):
        if not keep(r):
            continue
        try:
            out[(str(r["id"]).strip(), str(r["study_interval"]).strip(), int(r[day_col]))] = float(r[val_col])
        except (KeyError, ValueError, TypeError):
            continue
    return out


def load_signals(path: str | Path = DEFAULT_DIR) -> dict[str, dict]:
    """Return {signal_name: {(id, interval, day): value}} for the daily biosignals."""
    path = Path(path)
    return {
        "lh": _load_daily(path / "hormones_and_selfreport.csv", "day_in_study", "lh"),
        "temp": _load_daily(path / "computed_temperature.csv",
                            "sleep_start_day_in_study", "nightly_temperature",
                            keep=lambda r: r.get("type") == "SKIN"),
        "rhr": _load_daily(path / "resting_heart_rate.csv", "day_in_study", "value"),
        "resp": _load_daily(path / "respiratory_rate_summary.csv",
                            "day_in_study", "full_sleep_breathing_rate"),
    }


def cycle_series(cycle, signals: dict[str, dict]) -> dict[str, dict[int, float]]:
    """For one Cycle, return {signal: {day_of_cycle: value}} over [onset, next_onset).

    day_of_cycle is 0-based from menses onset. Requires the cycle to carry
    onset_day/next_onset_day in .extra (mcPHASES adapter provides these).
    """
    sid = cycle.extra["subject"]
    interval = cycle.extra["interval"]
    a = cycle.extra["onset_day"]
    b = cycle.extra["next_onset_day"]
    out: dict[str, dict[int, float]] = {}
    for name, series in signals.items():
        days = {}
        for day in range(a, b):
            v = series.get((sid, interval, day))
            if v is not None:
                days[day - a] = v
        out[name] = days
    return out
