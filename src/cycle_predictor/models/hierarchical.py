"""Hierarchical Bayesian cycle-length model (PLAN.md M3, v1).

Partial-pooling Normal-Normal model:

    mu_pop                      population mean cycle length
    mu_i  ~ Normal(mu_pop, tau) each user's typical length (tau = between-user sd)
    y_ic  ~ Normal(mu_i, sigma) observed cycle lengths   (sigma = within-user sd)

We fit the three hyperparameters (mu_pop, tau, sigma) on TRAIN users — with PyMC
(`fit`) or analytically by method of moments (`fit_moments`, instant, no deps).
Prediction for any user is then the closed-form Normal-Normal posterior predictive
for their next cycle given their own history:

    mu_i | history ~ Normal(m_post, s2_post),   1/s2_post = 1/tau^2 + n/sigma^2
    next cycle     ~ Normal(m_post, s2_post + sigma^2)

This yields a point estimate AND a calibrated predictive interval, and a user with
no history falls back to the population prior (cold-start) — exactly the behavior
the literature says matters more than shaving point MAE.

Not yet modeled: skipped-cycle artifacts (a forgotten log ~doubles a cycle). The
generative SOTA (Li/Urteaga) adds a latent skip count; that's the planned v2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean, variance
from typing import Sequence


@dataclass
class HierarchicalBayes:
    mu_pop: float       # population mean cycle length (days)
    tau2: float         # between-user variance
    sigma2: float       # within-user variance
    diagnostics: dict | None = None

    # ---------------------------------------------------------------- prediction
    def predict(self, history: Sequence[int]) -> tuple[float, float]:
        """Posterior-predictive (mean, sd) for the next cycle length."""
        n = len(history)
        if n == 0:  # cold-start → population prior
            return self.mu_pop, math.sqrt(self.tau2 + self.sigma2)
        xbar = fmean(history)
        precision = 1.0 / self.tau2 + n / self.sigma2
        s2_post = 1.0 / precision
        m_post = s2_post * (self.mu_pop / self.tau2 + n * xbar / self.sigma2)
        return m_post, math.sqrt(s2_post + self.sigma2)

    def point(self, history: Sequence[int]) -> float:
        """Point predictor (posterior-predictive mean) for the point eval harness."""
        return self.predict(history)[0]

    # ----------------------------------------------------------------- fitting
    @classmethod
    def fit_moments(cls, train_sequences: Sequence[Sequence[int]]) -> "HierarchicalBayes":
        """Analytic method-of-moments fit (no PyMC). Deterministic and instant."""
        users = [list(s) for s in train_sequences if len(s) > 0]
        if not users:
            raise ValueError("no training data")
        user_means = [fmean(s) for s in users]
        mu_pop = fmean(user_means)

        # pooled within-user variance (sigma^2)
        ss, df = 0.0, 0
        for s in users:
            if len(s) >= 2:
                m = fmean(s)
                ss += sum((x - m) ** 2 for x in s)
                df += len(s) - 1
        sigma2 = ss / df if df > 0 else 9.0

        # between-user variance (tau^2) by method of moments:
        # Var(user means) ≈ tau^2 + sigma^2 / nbar
        if len(user_means) >= 2:
            nbar = fmean([len(s) for s in users])
            tau2 = max(variance(user_means) - sigma2 / nbar, 0.1)
        else:
            tau2 = 9.0
        return cls(mu_pop, tau2, sigma2, diagnostics={"method": "moments", "users": len(users)})

    @classmethod
    def fit(cls, train_sequences: Sequence[Sequence[int]], *, draws: int = 1000,
            tune: int = 1000, chains: int = 2, seed: int = 0,
            progressbar: bool = False) -> "HierarchicalBayes":
        """Fit hyperparameters with PyMC (full Bayesian). Falls back to fit_moments
        if PyMC is unavailable."""
        try:
            import numpy as np
            import pymc as pm
        except ModuleNotFoundError:
            return cls.fit_moments(train_sequences)

        users = [list(s) for s in train_sequences if len(s) > 0]
        y, uidx = [], []
        for i, s in enumerate(users):
            y.extend(s)
            uidx.extend([i] * len(s))
        y = np.asarray(y, dtype=float)
        uidx = np.asarray(uidx)
        U = len(users)

        with pm.Model():
            mu_pop = pm.Normal("mu_pop", mu=29.0, sigma=5.0)
            tau = pm.HalfNormal("tau", sigma=5.0)
            sigma = pm.HalfNormal("sigma", sigma=5.0)
            mu_i = pm.Normal("mu_i", mu=mu_pop, sigma=tau, shape=U)
            pm.Normal("y", mu=mu_i[uidx], sigma=sigma, observed=y)
            idata = pm.sample(draws, tune=tune, chains=chains, cores=1,
                              random_seed=seed, progressbar=progressbar)

        post = idata.posterior
        mu_pop_hat = float(post["mu_pop"].mean())
        tau_hat = float(post["tau"].mean())
        sigma_hat = float(post["sigma"].mean())
        return cls(mu_pop_hat, tau_hat ** 2, sigma_hat ** 2,
                   diagnostics={"method": "pymc", "users": U,
                                "draws": draws, "chains": chains})
