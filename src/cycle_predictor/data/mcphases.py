"""Adapter for the mcPHASES dataset → canonical Cycle records.

Source: https://physionet.org/content/mcphases/ (credentialed; extract the CSVs to
data/raw/mcphases/ — see data/DATASETS.md). This adapter uses only
`hormones_and_selfreport.csv` (daily hormone-verified phase + self-report) and
`subject-info.csv` (demographics); the wearable time series are separate.

Cycles are derived from **menstruation onsets**: within each (subject, study
interval) we find runs of `phase == "Menstrual"` days (a gap > `gap` days starts a
new period) and take the first day of each run as a cycle start. Cycle length is
the gap between consecutive onsets in `day_in_study` (a per-interval calendar-day
index). Ovulation day is the offset (from onset) of the peak LH within the cycle.

Because `day_in_study` resets each study interval, we treat each (subject, interval)
as one canonical "user" so sequences are contiguous. ~62 such units, ~128 cycles.

⚠ Credentialed PhysioNet data under a DUA — do NOT commit or redistribute the raw
files (they are gitignored).
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from . import Cycle

DEFAULT_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "mcphases"


def _float(x: str | None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _menstrual_runs(days: list[int], gap: int) -> list[list[int]]:
    runs: list[list[int]] = []
    for d in sorted(days):
        if not runs or d - runs[-1][-1] > gap:
            runs.append([d])
        else:
            runs[-1].append(d)
    return runs


def _birth_years(path: Path) -> dict[str, int]:
    f = path / "subject-info.csv"
    out: dict[str, int] = {}
    if f.exists():
        for r in csv.DictReader(open(f, newline="")):
            by = r.get("birth_year")
            if by and by.strip().isdigit():
                out[str(r["id"]).strip()] = int(by)
    return out


def load(path: str | Path = DEFAULT_DIR, **kw) -> list[Cycle]:
    return list(iter_cycles(path, **kw))


def iter_cycles(path: str | Path = DEFAULT_DIR, *, min_len: int = 15,
                max_len: int = 60, gap: int = 3) -> Iterator[Cycle]:
    path = Path(path)
    horm = path / "hormones_and_selfreport.csv"
    if not horm.exists():
        raise FileNotFoundError(
            f"{horm} not found — extract mcPHASES to {path} (see data/DATASETS.md)"
        )
    birth = _birth_years(path)

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in csv.DictReader(open(horm, newline="")):
        groups[(str(r["id"]).strip(), str(r["study_interval"]).strip())].append(r)

    for (sid, interval), recs in groups.items():
        for r in recs:
            r["_day"] = int(r["day_in_study"])
        lh_by_day = {r["_day"]: _float(r.get("lh")) for r in recs if _float(r.get("lh")) is not None}
        runs = _menstrual_runs([r["_day"] for r in recs if r.get("phase") == "Menstrual"], gap)
        onsets = [run[0] for run in runs]
        run_span = {run[0]: run[-1] - run[0] + 1 for run in runs}

        age = None
        if sid in birth and interval.isdigit():
            age = int(interval) - birth[sid]

        for k, (a, b) in enumerate(zip(onsets, onsets[1:]), start=1):
            length = b - a
            if not (min_len <= length <= max_len):
                continue
            window = {d: v for d, v in lh_by_day.items() if a <= d < b}
            ovulation = (max(window, key=window.get) - a) if window else None
            yield Cycle(
                user_id=f"{sid}_{interval}",
                cycle_number=k,
                cycle_length_days=length,
                period_length_days=run_span.get(a),
                estimated_ovulation_day=ovulation,
                extra={"subject": sid, "interval": interval, "age": age},
            )


if __name__ == "__main__":  # smoke check
    import statistics
    cs = load()
    lens = [c.cycle_length_days for c in cs]
    ov = [c.estimated_ovulation_day for c in cs if c.estimated_ovulation_day is not None]
    print(f"cycles={len(cs)}  users={len({c.user_id for c in cs})}  "
          f"mean_len={statistics.mean(lens):.1f}d  "
          f"ovulation_day(median)={statistics.median(ov):.0f} (n={len(ov)})")
