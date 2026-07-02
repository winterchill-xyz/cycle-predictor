"""Tests for the evaluation harness (point + probabilistic + split)."""
import random

from cycle_predictor.eval import evaluate, evaluate_prob, split


def test_evaluate_point_known_errors():
    # positions: i=1 hist[28]->pred28 vs 30 (err2); i=2 hist[28,30]->pred28 vs 28 (err0)
    m = evaluate(lambda h: 28.0, [[28, 30, 28]])
    assert m.n == 2
    assert abs(m.mae - 1.0) < 1e-9


def test_evaluate_max_history_coldstart_slice():
    seq = [[28, 30, 31, 33, 35]]
    full = evaluate(lambda h: 28.0, seq)
    cold = evaluate(lambda h: 28.0, seq, max_history=2)
    assert cold.n == 2 and full.n == 4          # only positions i=1,2 counted


def test_split_deterministic_and_disjoint():
    seqs = [[i] for i in range(100)]
    a1, b1 = split(seqs, 0.7, seed=5)
    a2, b2 = split(seqs, 0.7, seed=5)
    assert a1 == a2 and b1 == b2
    assert len(a1) == 70 and len(b1) == 30
    assert not ({s[0] for s in a1} & {s[0] for s in b1})   # no user in both


def test_evaluate_prob_perfect_calibration():
    rng = random.Random(0)
    seqs = [[rng.gauss(29, 3) for _ in range(6)] for _ in range(400)]
    m = evaluate_prob(lambda h: (29.0, 3.0), seqs, levels=(0.8, 0.95))
    assert 0.76 <= m.coverage[0.8] <= 0.84
    assert 0.92 <= m.coverage[0.95] <= 0.98
