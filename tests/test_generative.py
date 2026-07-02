"""Tests for the skip-aware generative model. Requires numpy (project .venv)."""
import math
import random
import statistics

import pytest

pytest.importorskip("numpy")
from cycle_predictor.eval import evaluate_prob                         # noqa: E402
from cycle_predictor.models.generative import (                        # noqa: E402
    SkipAwareGenerative, SkipAwareGenPoisson,
)


def test_coldstart_is_prior_mean():
    mu_log, tau = math.log(29.0), 0.10
    m = SkipAwareGenerative(mu_log=mu_log, tau=tau, pi=0.05)
    mean, sd = m.predict([])
    assert abs(mean - math.exp(mu_log + tau ** 2 / 2)) < 0.3   # E[lambda] of lognormal
    assert sd > 0


def test_skip_artifact_is_explained_away():
    m = SkipAwareGenerative(mu_log=math.log(29.0), tau=0.10, pi=0.06)
    regular = [28, 29, 28, 27, 29]
    with_skip = regular + [56]                      # one forgotten log → ~2x cycle
    gen = m.predict(with_skip)[0]
    naive = statistics.fmean(with_skip)
    assert gen < naive - 2.0                        # not dragged up by the artifact
    assert abs(gen - statistics.fmean(regular)) < 2.0   # ~recovers the true rate


def test_expected_skip_rate_increases_with_pi():
    lo = SkipAwareGenerative(mu_log=math.log(29), tau=0.1, pi=0.02).expected_skip_rate()
    hi = SkipAwareGenerative(mu_log=math.log(29), tau=0.1, pi=0.20).expected_skip_rate()
    assert 0 <= lo < hi


def test_moments_fit_matches_log_mean():
    seqs = [[28, 30, 29, 31], [27, 29, 28]]
    m = SkipAwareGenerative.fit_moments(seqs)
    flat = [x for s in seqs for x in s]
    assert abs(m.mu_log - statistics.fmean([math.log(x) for x in flat])) < 0.05
    assert m.diagnostics["method"] == "moments"


def _synth_with_skips(seed, n_users=60, mu=29.0, tau_log=0.08, pi=0.10):
    rng = random.Random(seed)
    seqs = []
    for _ in range(n_users):
        lam = mu * math.exp(rng.gauss(0, tau_log))
        seq = []
        for _ in range(rng.randint(6, 12)):
            skips = 0
            while rng.random() < pi and skips < 3:
                skips += 1
            val = sum(max(1, round(rng.gauss(lam, lam ** 0.5))) for _ in range(1 + skips))
            seq.append(val)
        seqs.append(seq)
    return seqs


@pytest.mark.slow
def test_pymc_fit_recovers_rate_and_detects_skips():
    pytest.importorskip("pymc")
    m = SkipAwareGenerative.fit(_synth_with_skips(seed=1), draws=150, tune=150,
                                chains=1, seed=0)
    assert m.diagnostics["method"] == "pymc"
    assert 26.0 < math.exp(m.mu_log) < 32.0       # recovers population rate
    assert m.pi > 0.01                            # detects that skips happen


# --------------------------------------------------------------- Generalized Poisson
def _underdispersed(seed, n_users=200, mu=29.0, sd=2.9, k=8):
    rng = random.Random(seed)
    return [[max(1, round(rng.gauss(mu, sd))) for _ in range(k)] for _ in range(n_users)]


def test_gp_detects_underdispersion():
    m = SkipAwareGenPoisson.fit_moments(_underdispersed(0))
    assert m.xi < 0                      # variance < mean ⇒ xi negative
    assert m.diagnostics["phi"] < 1.0


def test_gp_calibration_near_nominal_on_underdispersed():
    train, test = _underdispersed(1), _underdispersed(2)
    m = SkipAwareGenPoisson.fit_moments(train)
    pm = evaluate_prob(m.predict, test, levels=(0.8,))
    assert 0.70 <= pm.coverage[0.8] <= 0.90


def test_gp_intervals_tighter_than_poisson():
    # On under-dispersed data the Poisson (v2) intervals are too wide; GP fixes it.
    train, test = _underdispersed(3), _underdispersed(4)
    pois = evaluate_prob(SkipAwareGenerative.fit_moments(train, pi=0.05).predict, test, levels=(0.8,))
    gp = evaluate_prob(SkipAwareGenPoisson.fit_moments(train).predict, test, levels=(0.8,))
    assert gp.sharpness[0.8] < pois.sharpness[0.8]                 # tighter
    assert abs(gp.coverage[0.8] - 0.8) < abs(pois.coverage[0.8] - 0.8)   # better-calibrated


def test_gp_still_skip_robust():
    m = SkipAwareGenPoisson.fit_moments(_underdispersed(5), pi=0.06)
    regular = [28, 29, 28, 27, 29]
    gen = m.predict(regular + [56])[0]
    assert gen < statistics.fmean(regular + [56]) - 2.0     # explains the skip
