"""Dataset adapters → the canonical cycle schema.

Canonical record (one row per observed cycle):
    user_id: str
    cycle_number: int          # 1..N in chronological order per user
    cycle_length_days: int | None
    period_length_days: int | None
    estimated_ovulation_day: int | None
    luteal_length_days: int | None
    extra: dict                # dataset-specific covariates kept verbatim
"""
from dataclasses import dataclass, field


@dataclass
class Cycle:
    user_id: str
    cycle_number: int
    cycle_length_days: int | None = None
    period_length_days: int | None = None
    estimated_ovulation_day: int | None = None
    luteal_length_days: int | None = None
    extra: dict = field(default_factory=dict)
