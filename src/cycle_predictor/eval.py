"""Evaluation harness: the contract every model plugs into (see PLAN.md M2).

A *predictor* is any callable `history -> predicted_next_length`, where `history`
is the list of a user's prior cycle lengths (chronological). We evaluate by walking
each user's sequence and, at each position i>=1, predicting length[i] from
length[:i] — a strict predict-future-from-past protocol.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

Predictor = Callable[[Sequence[int]], float]


@dataclass
class Metrics:
    mae: float          # mean absolute error, days
    rmse: float         # root mean squared error, days
    within1: float      # fraction of predictions within +/-1 day
    within2: float      # fraction within +/-2 days
    n: int              # number of (history, target) evaluations

    def __str__(self) -> str:
        return (f"MAE={self.mae:5.2f}d  RMSE={self.rmse:5.2f}d  "
                f"±1d={self.within1:5.1%}  ±2d={self.within2:5.1%}  n={self.n}")


def evaluate(predictor: Predictor, sequences: Iterable[Sequence[int]],
             min_history: int = 1) -> Metrics:
    """Score a predictor over per-user chronological cycle-length sequences."""
    abs_errs, sq_errs, w1, w2, n = 0.0, 0.0, 0, 0, 0
    for seq in sequences:
        for i in range(min_history, len(seq)):
            pred = predictor(seq[:i])
            err = abs(pred - seq[i])
            abs_errs += err
            sq_errs += err * err
            w1 += err <= 1
            w2 += err <= 2
            n += 1
    if n == 0:
        return Metrics(math.nan, math.nan, 0.0, 0.0, 0)
    return Metrics(abs_errs / n, math.sqrt(sq_errs / n), w1 / n, w2 / n, n)


def user_sequences(cycles, length_attr: str = "cycle_length_days",
                   order_attr: str = "cycle_number") -> list[list[int]]:
    """Group canonical Cycle records into per-user, chronologically-ordered
    lists of cycle lengths (dropping cycles with no length)."""
    by_user: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for c in cycles:
        length = getattr(c, length_attr)
        if length:
            by_user[c.user_id].append((getattr(c, order_attr), length))
    sequences = []
    for items in by_user.values():
        items.sort(key=lambda t: t[0])
        sequences.append([length for _, length in items])
    return sequences
