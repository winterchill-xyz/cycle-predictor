"""Adapter for the FedCycle / Marquette dataset → canonical Cycle records.

Source: https://epublications.marquette.edu/data_nfp/7/ (download via
`scripts/fetch_datasets.py --only fedcycle` → data/raw/fedcycle/).
80 columns, ~1665 cycles from ~159 women. Cells are sometimes blank or a single
space; numeric fields are coerced leniently.

⚠ Licensing for reuse is UNVERIFIED (see data/DATASETS.md). Research use only until
confirmed.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from . import Cycle

DEFAULT_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "raw" / "fedcycle" / "FedCycleData071012.csv"
)


def _int(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def load(path: str | Path = DEFAULT_PATH) -> list[Cycle]:
    """Read FedCycle CSV into canonical Cycle records (chronological per user)."""
    return list(iter_cycles(path))


def iter_cycles(path: str | Path = DEFAULT_PATH) -> Iterator[Cycle]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run: python scripts/fetch_datasets.py --only fedcycle"
        )
    with open(path, newline="", encoding="utf-8-sig") as f:  # utf-8-sig strips the BOM
        reader = csv.DictReader(f)
        for row in reader:
            client = (row.get("ClientID") or "").strip()
            if not client:
                continue
            yield Cycle(
                user_id=client,
                cycle_number=_int(row.get("CycleNumber")) or 0,
                cycle_length_days=_int(row.get("LengthofCycle")),
                period_length_days=_int(row.get("LengthofMenses")),
                estimated_ovulation_day=_int(row.get("EstimatedDayofOvulation")),
                luteal_length_days=_int(row.get("LengthofLutealPhase")),
                extra={
                    "group": (row.get("Group") or "").strip(),
                    "reproductive_category": (row.get("ReproductiveCategory") or "").strip(),
                    "mean_cycle_length": _int(row.get("MeanCycleLength")),
                    "has_peak": (row.get("CycleWithPeakorNot") or "").strip(),
                },
            )


if __name__ == "__main__":  # quick smoke check
    cycles = load()
    lengths = [c.cycle_length_days for c in cycles if c.cycle_length_days]
    users = {c.user_id for c in cycles}
    print(f"cycles={len(cycles)} users={len(users)} "
          f"with_length={len(lengths)} mean_len={sum(lengths)/len(lengths):.1f}d")
