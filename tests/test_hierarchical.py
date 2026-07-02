"""Tests for the hierarchical Bayesian model. Fast paths use the analytic
method-of-moments fit; the PyMC fit gets one marked smoke test."""
import math
import random

import pytest

from cycle_predictor.eval import evaluate_prob
from cycle_predictor.models.hierarchical import HierarchicalBayes


def synth(seed=0, n_users=200, mu_pop=29.0, tau=3.0, sigma=2.5, min_c=4, max_c=10):
    rng = random.Random(seed)
    seqs = []
    for _ in range(n_users):
        mu_i = rng.gauss(mu_pop, tau)
        k = rng.randint(min_c, max_c)
        seqs.append([rng.gauss(mu_i, sigma) for _ in range(k)])
    return seqs


def test_moments_recovers_hyperparams():
    m = HierarchicalBayes.fit_moments(synth())
    assert abs(m.mu_pop - 29.0) < 1.0
    assert abs(m.tau2 ** 0.5 - 3.0) < 1.0
    assert abs(m.sigma2 ** 0.5 - 2.5) < 0.7


def test_coldstart_returns_population_prior():
    m = HierarchicalBayes(mu_pop=29.0, tau2=9.0, sigma2=6.25)
    mean, sd = m.predict([])
    assert mean == 29.0
    assert math.isclose(sd, math.sqrt(9.0 + 6.25))


def test_shrinkage_toward_population_with_short_history():
    m = HierarchicalBayes(mu_pop=29.0, tau2=4.0, sigma2=9.0)
    mean1, _ = m.predict([40])          # one far obs → pulled toward pop
    mean2, _ = m.predict([40] * 30)     # much data → close to the data mean
    assert 29.0 < mean1 < 40.0
    assert abs(mean2 - 40) < abs(mean1 - 40)


def test_more_history_shrinks_predictive_sd():
    m = HierarchicalBayes(mu_pop=29.0, tau2=4.0, sigma2=9.0)
    _, sd0 = m.predict([])
    _, sd_short = m.predict([29])
    _, sd_long = m.predict([29] * 20)
    assert sd0 > sd_short > sd_long   # more evidence → tighter interval


def test_point_equals_predict_mean():
    m = HierarchicalBayes(29.0, 4.0, 9.0)
    h = [28, 30, 29]
    assert m.point(h) == m.predict(h)[0]


def test_calibration_near_nominal_on_synthetic():
    train = synth(seed=1, n_users=300)
    test = synth(seed=2, n_users=300)
    m = HierarchicalBayes.fit_moments(train)
    pm = evaluate_prob(m.predict, test, levels=(0.8,))
    assert 0.72 <= pm.coverage[0.8] <= 0.88


@pytest.mark.slow
def test_pymc_fit_smoke():
    pytest.importorskip("pymc")
    m = HierarchicalBayes.fit(synth(seed=3, n_users=40),
                              draws=100, tune=100, chains=1, seed=0)
    assert m.diagnostics["method"] == "pymc"
    assert 26.0 < m.mu_pop < 32.0
