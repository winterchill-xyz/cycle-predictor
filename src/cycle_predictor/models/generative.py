"""Skip-aware generative cycle-length model (PLAN.md M3 v2).

The generative story from Li et al. (JAMIA 2022) / Urteaga et al. (2021): a user's
*true* cycle length is Poisson(lambda_i); when they forget to log a period, s
consecutive true cycles are recorded as ONE observed cycle, so

    observed_length | s, lambda_i  ~  Poisson((1 + s) * lambda_i),
    s ~ (truncated) Geometric(pi)   # number of skipped logs, usually 0

We marginalize the discrete skip count s out of the likelihood (a mixture over
s = 0..s_max). The payoff: an occasional ~2x-length cycle is explained as a skip
(s=1) instead of inflating the estimate of lambda_i — so both the point estimate
and its uncertainty stay clean.

Fitting (`fit`) uses PyMC with this marginalized likelihood to estimate the three
population hyperparameters (mu_log, tau, pi). Prediction is a per-user 1-D grid
posterior over lambda_i given that user's observed history and the fitted
hyperparameters — fast, deterministic, and cold-start-aware (empty history → prior).
We predict the next *true* cycle (s=0), i.e. assuming the user does log it.

Requires numpy (project .venv). `fit` falls back to a lognormal moment fit with a
small fixed skip rate if PyMC is unavailable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import fmean
from typing import Sequence

import numpy as np


def _logsumexp(a, axis=None):
    # Robust to all -inf slices (returns -inf, not nan): GP support can make a whole
    # (grid, skip) row impossible.
    a = np.asarray(a, dtype=float)
    m = np.max(a, axis=axis, keepdims=True)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        s = np.log(np.sum(np.exp(a - m_safe), axis=axis, keepdims=True))
    out = np.where(np.isfinite(m), m_safe + s, m)
    return float(np.squeeze(out)) if axis is None else np.squeeze(out, axis=axis)


@dataclass
class SkipAwareGenerative:
    mu_log: float           # population mean of log(lambda)
    tau: float              # population sd of log(lambda) (between-user)
    pi: float               # skip probability (Geometric)
    s_max: int = 3          # truncate skip count at this many
    lam_min: float = 15.0
    lam_max: float = 55.0
    grid: int = 180
    diagnostics: dict | None = None
    _built: bool = field(default=False, repr=False)

    def __post_init__(self):
        S = np.arange(0, self.s_max + 1)
        self._mult = (1 + S).astype(float)                      # (K,)
        logw = np.log(1 - self.pi) + S * np.log(self.pi)        # unnormalized Geometric
        self._logw = logw - _logsumexp(logw)                   # (K,)
        self._loglam = np.linspace(math.log(self.lam_min), math.log(self.lam_max), self.grid)
        self._lam = np.exp(self._loglam)                       # (G,)
        self._logprior = (-0.5 * ((self._loglam - self.mu_log) / self.tau) ** 2
                          - math.log(self.tau) - 0.5 * math.log(2 * math.pi))
        self._lam_eff = self._mult[None, :] * self._lam[:, None]   # (G,K)
        self._log_lam_eff = np.log(self._lam_eff)
        self._built = True

    # ---------------------------------------------------------------- prediction
    def _log_posterior(self, history: Sequence[float]) -> np.ndarray:
        logpost = self._logprior.copy()
        for d in history:
            di = max(1, int(round(d)))
            # Poisson logpmf up to the constant -lgamma(d+1) (cancels on normalize):
            pois = di * self._log_lam_eff - self._lam_eff          # (G,K)
            logpost = logpost + _logsumexp(self._logw[None, :] + pois, axis=1)
        return logpost - _logsumexp(logpost)

    def predict(self, history: Sequence[float]) -> tuple[float, float]:
        """Posterior-predictive (mean, sd) for the next TRUE cycle length."""
        post = np.exp(self._log_posterior(history))
        mean = float(np.sum(post * self._lam))
        ex2 = float(np.sum(post * self._lam ** 2))
        var_lambda = max(ex2 - mean ** 2, 0.0)
        var = mean + var_lambda            # Poisson predictive: E[lam] + Var[lam]
        return mean, math.sqrt(var)

    def point(self, history: Sequence[float]) -> float:
        return self.predict(history)[0]

    def expected_skip_rate(self) -> float:
        """Prior mean number of skipped logs per observed cycle."""
        S = np.arange(0, self.s_max + 1)
        return float(np.sum(np.exp(self._logw) * S))

    # ----------------------------------------------------------------- fitting
    @classmethod
    def fit_moments(cls, train_sequences, pi: float = 0.03, **kw) -> "SkipAwareGenerative":
        """Lognormal moment fit with a fixed small skip rate (no PyMC)."""
        vals = [x for s in train_sequences for x in s if x]
        logs = np.log(np.asarray(vals, dtype=float))
        return cls(float(logs.mean()), float(logs.std(ddof=1)) or 0.1, pi,
                   diagnostics={"method": "moments"}, **kw)

    @classmethod
    def fit(cls, train_sequences, *, draws: int = 1000, tune: int = 1000,
            chains: int = 2, seed: int = 0, s_max: int = 3,
            progressbar: bool = False, **kw) -> "SkipAwareGenerative":
        """Estimate (mu_log, tau, pi) with PyMC and the skip-marginalized likelihood."""
        try:
            import pymc as pm
        except ModuleNotFoundError:
            return cls.fit_moments(train_sequences, s_max=s_max, **kw)

        users = [list(s) for s in train_sequences if len(s) > 0]
        d, uidx = [], []
        for i, s in enumerate(users):
            for x in s:
                d.append(int(round(x)))
                uidx.append(i)
        d = np.asarray(d, dtype=float)
        uidx = np.asarray(uidx)
        U = len(users)
        S = np.arange(0, s_max + 1).astype(float)
        mult = 1 + S

        with pm.Model():
            mu_log = pm.Normal("mu_log", math.log(29.0), 0.2)
            tau = pm.HalfNormal("tau", 0.15)
            loglam = pm.Normal("loglam", mu_log, tau, shape=U)
            lam = pm.math.exp(loglam)
            pi = pm.Beta("pi", 1.0, 20.0)                       # skips are rare
            logw = pm.math.log(1 - pi) + S * pm.math.log(pi)     # (K,)
            logw = logw - pm.math.logsumexp(logw)
            lam_obs = lam[uidx]                                  # (N,)
            lam_eff = mult[None, :] * lam_obs[:, None]           # (N,K)
            pois = d[:, None] * pm.math.log(lam_eff) - lam_eff   # drop const -lgamma(d+1)
            logp = pm.math.logsumexp(logw[None, :] + pois, axis=1)
            pm.Potential("lik", pm.math.sum(logp))
            idata = pm.sample(draws, tune=tune, chains=chains, cores=1,
                              random_seed=seed, progressbar=progressbar)

        post = idata.posterior
        return cls(float(post["mu_log"].mean()), float(post["tau"].mean()),
                   float(post["pi"].mean()), s_max=s_max,
                   diagnostics={"method": "pymc", "users": U,
                                "pi": float(post["pi"].mean())}, **kw)


@dataclass
class SkipAwareGenPoisson:
    """Skip-aware GENERALIZED-Poisson model (PLAN.md M3 v2.1).

    Same skip story as SkipAwareGenerative, but the component distribution is the
    Consul Generalized Poisson GP(lambda, xi) instead of Poisson. Two facts make
    this the right upgrade:

      * GP is closed under convolution: sum of m iid GP(lambda, xi) = GP(m*lambda,
        xi). So a skip of s still gives  observed ~ GP((1+s)*lambda, xi)  — the
        marginalization is unchanged.
      * xi is a second free parameter: mean = lambda/(1-xi), var = lambda/(1-xi)^3.
        xi < 0 gives UNDER-dispersion (var < mean), which real cycle lengths have
        (within-user sd ~2.9 d vs Poisson's sqrt(29) ~5.4 d). This fixes v2's
        over-wide, over-covering intervals.

    We fit xi by moment-matching the dispersion ratio phi = var/mean of the
    training cycles (phi = 1/(1-xi)^2 ⇒ xi = 1 - phi**-0.5), i.e. literally fitting
    the mean and the variance separately — the Generalized-Poisson selling point.
    Prediction is the same per-user 1-D grid posterior over lambda; the next true
    cycle's predictive uses mean=E[lambda]/(1-xi), var=E[lambda]/(1-xi)^3 +
    Var(lambda)/(1-xi)^2.
    """
    mu_log: float
    tau: float
    pi: float
    xi: float                     # dispersion; xi < 0 => under-dispersed
    s_max: int = 3
    lam_min: float = 12.0
    lam_max: float = 140.0
    grid: int = 240
    diagnostics: dict | None = None
    _built: bool = field(default=False, repr=False)

    def __post_init__(self):
        S = np.arange(0, self.s_max + 1)
        self._mult = (1 + S).astype(float)
        logw = np.log(1 - self.pi) + S * np.log(self.pi)
        self._logw = logw - _logsumexp(logw)
        self._loglam = np.linspace(math.log(self.lam_min), math.log(self.lam_max), self.grid)
        self._lam = np.exp(self._loglam)
        self._logprior = (-0.5 * ((self._loglam - self.mu_log) / self.tau) ** 2
                          - math.log(self.tau) - 0.5 * math.log(2 * math.pi))
        self._a = self._mult[None, :] * self._lam[:, None]     # (G,K) = (1+s)*lambda
        self._log_a = np.log(self._a)
        self._c1 = 1.0 / (1.0 - self.xi)
        self._c3 = 1.0 / (1.0 - self.xi) ** 3
        self._built = True

    def _loglik_obs(self, d: int) -> np.ndarray:
        # GP logpmf for integer d over the grid (drop const -lgamma(d+1)); the GP
        # support requires a + d*xi > 0, else that (grid, skip) cell is impossible.
        arg = self._a + d * self.xi
        with np.errstate(invalid="ignore", divide="ignore"):
            lp = self._log_a + (d - 1) * np.log(arg) - arg
        lp = np.where(arg > 0, lp, -np.inf)
        return _logsumexp(self._logw[None, :] + lp, axis=1)

    def _log_posterior(self, history: Sequence[float]) -> np.ndarray:
        logpost = self._logprior.copy()
        for d in history:
            logpost = logpost + self._loglik_obs(max(1, int(round(d))))
        return logpost - _logsumexp(logpost)

    def predict(self, history: Sequence[float]) -> tuple[float, float]:
        post = np.exp(self._log_posterior(history))
        elam = float(np.sum(post * self._lam))
        var_lam = max(float(np.sum(post * self._lam ** 2)) - elam ** 2, 0.0)
        mean = elam * self._c1
        var = elam * self._c3 + var_lam * self._c1 ** 2
        return mean, math.sqrt(var)

    def point(self, history: Sequence[float]) -> float:
        return self.predict(history)[0]

    @classmethod
    def fit_moments(cls, train_sequences, pi: float = 0.05, **kw) -> "SkipAwareGenPoisson":
        """Method-of-moments fit: population log-rate from user means, dispersion xi
        from the pooled within-user variance/mean ratio."""
        users = [list(s) for s in train_sequences if len(s) > 0]
        if not users:
            raise ValueError("no training data")
        user_means = [fmean(s) for s in users]

        ss, df, tot, cnt = 0.0, 0, 0.0, 0
        for s in users:
            if len(s) >= 2:
                m = fmean(s)
                ss += sum((x - m) ** 2 for x in s)
                df += len(s) - 1
                tot += sum(s)
                cnt += len(s)
        within_var = ss / df if df > 0 else 9.0
        within_mean = tot / cnt if cnt > 0 else fmean(user_means)
        phi = min(max(within_var / within_mean if within_mean else 1.0, 0.05), 3.0)
        xi = 1.0 - phi ** -0.5                                  # phi<1 => xi<0

        log_mu = np.log(np.asarray(user_means, dtype=float))
        mu_log = float(log_mu.mean()) + math.log(1 - xi)       # log(lambda)=log(mu)+log(1-xi)
        tau = float(log_mu.std(ddof=1)) or 0.1
        return cls(mu_log, tau, pi, xi,
                   diagnostics={"method": "moments", "phi": phi, "xi": xi}, **kw)
