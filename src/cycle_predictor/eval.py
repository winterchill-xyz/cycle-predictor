"""Evaluation harness: the contract every model plugs into (see PLAN.md M2).

A *predictor* is any callable `history -> predicted_next_length`, where `history`
is the list of a user's prior cycle lengths (chronological). We evaluate by walking
each user's sequence and, at each position i>=1, predicting length[i] from
length[:i] — a strict predict-future-from-past protocol.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from statistics import NormalDist
from typing import Callable, Iterable, Sequence

# A point predictor maps history -> predicted next length.
Predictor = Callable[[Sequence[int]], float]
# A probabilistic predictor maps history -> (mean, sd) of a predictive Normal.
ProbPredictor = Callable[[Sequence[int]], "tuple[float, float]"]


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
             min_history: int = 1, max_history: int | None = None) -> Metrics:
    """Score a point predictor over per-user chronological sequences.

    Predict seq[i] from seq[:i] for min_history <= i, optionally capping the
    history length at max_history (use e.g. max_history=3 for a cold-start slice).
    """
    abs_errs, sq_errs, w1, w2, n = 0.0, 0.0, 0, 0, 0
    for seq in sequences:
        for i in range(min_history, len(seq)):
            if max_history is not None and i > max_history:
                continue
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


@dataclass
class ProbMetrics:
    mae: float
    rmse: float
    coverage: dict          # level -> empirical coverage of the central interval
    sharpness: dict         # level -> mean width of the central interval (days)
    n: int

    def __str__(self) -> str:
        cov = "  ".join(f"{int(l*100)}%:{c:.0%}(w{self.sharpness[l]:.1f})"
                        for l, c in self.coverage.items())
        return f"MAE={self.mae:5.2f}d  RMSE={self.rmse:5.2f}d  cover[{cov}]  n={self.n}"


def evaluate_prob(predictor: ProbPredictor, sequences: Iterable[Sequence[int]],
                  min_history: int = 1, levels: Sequence[float] = (0.5, 0.8, 0.95)
                  ) -> ProbMetrics:
    """Score a probabilistic predictor: point error on the mean plus calibration.

    Calibration = does the central L-interval (mean ± z*sd) actually cover ~L of
    the held-out targets? A well-calibrated 80% interval covers ~80%.
    """
    z = {l: NormalDist().inv_cdf(0.5 + l / 2) for l in levels}
    abs_errs = sq_errs = 0.0
    covered = {l: 0 for l in levels}
    width = {l: 0.0 for l in levels}
    n = 0
    for seq in sequences:
        for i in range(min_history, len(seq)):
            mean, sd = predictor(seq[:i])
            sd = max(sd, 1e-6)
            actual = seq[i]
            err = abs(mean - actual)
            abs_errs += err
            sq_errs += err * err
            for l in levels:
                half = z[l] * sd
                covered[l] += (mean - half) <= actual <= (mean + half)
                width[l] += 2 * half
            n += 1
    if n == 0:
        return ProbMetrics(math.nan, math.nan, {}, {}, 0)
    return ProbMetrics(
        abs_errs / n, math.sqrt(sq_errs / n),
        {l: covered[l] / n for l in levels},
        {l: width[l] / n for l in levels},
        n,
    )


def split(sequences: list[Sequence[int]], frac_train: float = 0.7, seed: int = 0
          ) -> tuple[list, list]:
    """Deterministic user-holdout split (each sequence is one user). The test
    users are unseen at fit time → this measures cold-start honestly."""
    order = list(range(len(sequences)))
    random.Random(seed).shuffle(order)
    cut = int(len(order) * frac_train)
    train = [sequences[i] for i in order[:cut]]
    test = [sequences[i] for i in order[cut:]]
    return train, test


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
