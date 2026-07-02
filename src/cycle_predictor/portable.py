"""Pure-Python (numpy-free) reference inference kernel — the port target.

This is the canonical spec every cross-language port (Swift, TypeScript, …) must
reproduce. It reads a plain params dict (see DEFAULT_PARAMS / model.json) and does
the same math as the research models, in ~150 lines of elementary arithmetic with
no dependencies. `tests/test_portable.py` asserts it matches the numpy models, and
`scripts/gen_test_vectors.py` freezes its outputs as a cross-language conformance
suite.

Algorithm (predict the current cycle's length from onset):
  * If the current cycle shows an LH surge (LH ≥ lh_surge) or a wearable thermal
    shift → ovulation_day + luteal_mean, sd = luteal_sd (the two-phase model).
  * Else → the skip-aware Generalized-Poisson backbone posterior over the per-user
    rate λ, given the history of past cycle lengths (grid over log λ).
"""
from __future__ import annotations

import math

NEG_INF = float("-inf")

# Population defaults (mirror cycle_predictor.api constants; a test keeps them in sync).
DEFAULT_PARAMS = {
    "version": "0.0.1",
    "backbone": {
        "type": "skip_aware_genpoisson",
        "mu_log": 4.0438, "tau": 0.1001, "pi": 0.05, "xi": -0.9309,
        "s_max": 3, "lam_min": 12.0, "lam_max": 140.0, "grid": 240,
    },
    "twophase": {"luteal_mean": 13.85, "luteal_sd": 3.08},
    "lh_surge": 25.0,
    "thermal_shift": {"baseline": 6, "run": 3, "delta": 0.15, "search": [5, 22]},
}


def _logsumexp(vals) -> float:
    m = max(vals)
    if m == NEG_INF:
        return NEG_INF
    return m + math.log(sum(math.exp(v - m) for v in vals))


def backbone_predict(bp: dict, history) -> tuple[float, float]:
    """Generalized-Poisson posterior-predictive (mean, sd) for the next cycle length."""
    G = bp["grid"]
    lo, hi = math.log(bp["lam_min"]), math.log(bp["lam_max"])
    loglam = [lo + (hi - lo) * i / (G - 1) for i in range(G)]
    lam = [math.exp(x) for x in loglam]
    xi, pi, s_max = bp["xi"], bp["pi"], bp["s_max"]
    mu_log, tau = bp["mu_log"], bp["tau"]

    S = list(range(s_max + 1))
    logw_un = [math.log(1 - pi) + s * math.log(pi) for s in S]
    zz = _logsumexp(logw_un)
    logw = [w - zz for w in logw_un]
    mult = [1 + s for s in S]

    logpost = [-0.5 * ((x - mu_log) / tau) ** 2 - math.log(tau) - 0.5 * math.log(2 * math.pi)
               for x in loglam]
    for d in history:
        di = max(1, round(d))
        for gi in range(G):
            terms = []
            for si in range(len(S)):
                a = mult[si] * lam[gi]
                arg = a + di * xi
                terms.append(logw[si] + math.log(a) + (di - 1) * math.log(arg) - arg
                             if arg > 0 else NEG_INF)
            logpost[gi] += _logsumexp(terms)

    z2 = _logsumexp(logpost)
    post = [math.exp(v - z2) for v in logpost]
    elam = sum(post[i] * lam[i] for i in range(G))
    elam2 = sum(post[i] * lam[i] * lam[i] for i in range(G))
    var_lam = max(elam2 - elam * elam, 0.0)
    c1 = 1.0 / (1.0 - xi)
    c3 = 1.0 / (1.0 - xi) ** 3
    return elam * c1, math.sqrt(elam * c3 + var_lam * c1 * c1)


def detect_thermal_shift(temp_by_day: dict, ts: dict):
    baseline, run, delta = ts["baseline"], ts["run"], ts["delta"]
    lo, hi = ts["search"]
    if len(temp_by_day) < baseline + run:
        return None
    for d in range(lo, hi + 1):
        base = [temp_by_day[k] for k in range(d - baseline, d) if k in temp_by_day]
        fut = [temp_by_day[d + j] for j in range(run) if (d + j) in temp_by_day]
        if len(base) < baseline - 1 or len(fut) < run:
            continue
        cover = sum(base) / len(base) + delta
        if all(t > cover for t in fut):
            return max(d - 1, 0)
    return None


def predict(params: dict, history, lh_by_day=None, temp_by_day=None) -> dict:
    """Return {cycle_length, sd, mode} — the numpy-free equivalent of UnifiedPredictor."""
    if lh_by_day:
        surge = sorted(d for d, v in lh_by_day.items() if v is not None and v >= params["lh_surge"])
        if surge:
            tp = params["twophase"]
            return {"cycle_length": surge[0] + tp["luteal_mean"], "sd": tp["luteal_sd"],
                    "mode": "two_phase_lh"}
    if temp_by_day:
        est = detect_thermal_shift({int(k): v for k, v in temp_by_day.items()}, params["thermal_shift"])
        if est is not None:
            tp = params["twophase"]
            return {"cycle_length": est + tp["luteal_mean"], "sd": tp["luteal_sd"],
                    "mode": "two_phase_wearable"}
    mean, sd = backbone_predict(params["backbone"], list(history))
    return {"cycle_length": mean, "sd": sd, "mode": "history"}
