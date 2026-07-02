# Results

Reproduce: `.venv/bin/python scripts/eval_models.py` (PyMC fit) or `--fast`
(method-of-moments). Numbers below: FedCycle, 159 users, **70/30 user-holdout**
(test users unseen at fit time), seed 0, 558 held-out cycles / 510 scored
predictions.

## Next-cycle-length prediction (held-out users)

| predictor | MAE (d) | RMSE (d) | ±1d | ±2d | cold-start MAE (d) |
|-----------|--------:|---------:|----:|----:|-------------------:|
| constant_28 | 2.72 | 3.86 | 41% | 61% | 3.26 |
| last_cycle | 2.81 | 3.84 | 38% | 60% | 2.75 |
| personal_mean | 2.20 | 2.97 | 37% | 61% | 2.41 |
| personal_median | 2.15 | 2.96 | **47%** | **68%** | 2.41 |
| rolling_mean_3 | 2.29 | 3.12 | 38% | 61% | 2.41 |
| **hierarchical_bayes** | **2.13** | **2.86** | 35% | 60% | **2.24** |

*cold-start MAE = predictions made from ≤2 prior cycles.*

## Calibration (hierarchical_bayes)

Central predictive interval coverage on held-out cycles:

| nominal | empirical coverage | mean width (d) |
|--------:|-------------------:|---------------:|
| 50% | 62% | 4.2 |
| 80% | 85% | 8.0 |
| 95% | 96% | 12.2 |

## Reading of the results

- **Point MAE saturates**, exactly as the literature predicts: the Bayesian model
  (2.13 d) barely edges the trivial `personal_mean/median` (2.15–2.20 d). On a
  regular cohort, point accuracy is not where models separate.
- **The Bayesian model wins where it should:** best **RMSE** (fewer large misses)
  and best **cold-start MAE** (2.24 vs 2.41) — with little history it shrinks to the
  population prior instead of trusting 1–2 noisy cycles.
- **It ships calibrated uncertainty** (near-nominal interval coverage), which the
  point baselines simply cannot provide. Intervals slightly **over-cover** (80% →
  85%), i.e. a touch wide — a tuning target for v2.
- `personal_median` is the strongest *point* baseline on hit-rate (±1d 47%) because
  the length distribution is right-skewed.

## M3 v2 — skip-aware generative model

The v2 model (`models/generative.py`) treats an observed cycle as `(1+s)` true
cycles glued together when a period log is skipped, and marginalizes the latent
skip count `s`. On clean FedCycle it ties v1 (MAE 2.13 vs 2.14) — there's nothing
to fix. Its value shows up under skip artifacts. `scripts/demo_skip_robustness.py`
injects skips into held-out users' histories and measures recovery of the next
*true* cycle length:

| injected skip rate p | personal_mean | hierarchical_v1 | **skip_generative_v2** |
|---------------------:|--------------:|----------------:|-----------------------:|
| 0.00 | 2.31 | 2.29 | 2.29 |
| 0.05 | 3.24 | 3.14 | **2.26** |
| 0.10 | 4.26 | 3.99 | **2.26** |
| 0.20 | 6.99 | 6.20 | **2.32** |
| 0.30 | 12.21 | 10.35 | **2.54** |

Naive/v1 estimates get dragged up by merged cycles; **v2 stays flat** because it
attributes the long cycle to a forgotten log, not a real 56-day cycle. That gap is
the entire reason the generative approach is the literature's SOTA.

## M3 v2.1 — Generalized-Poisson (calibration fix)  ✅

v2's plain-**Poisson** intervals over-cover (80% → 98%) because Poisson is
over-dispersed for cycle lengths (Poisson sd ≈ √29 ≈ 5.4 d, but real within-user
sd ≈ 2.9 d — cycle lengths are *under*-dispersed). The **Generalized Poisson**
(`SkipAwareGenPoisson`) adds a dispersion parameter ξ (mean = λ/(1−ξ),
var = λ/(1−ξ)³); ξ < 0 gives under-dispersion. It's closed under convolution, so the
skip marginalization is unchanged. We fit ξ by moment-matching φ = var/mean
(on FedCycle φ ≈ 0.28 → ξ ≈ −0.88) — literally fitting mean and variance separately.

Calibration on held-out FedCycle (80% interval should cover ~80%):

| model | 50% | 80% | 95% | point MAE | cold MAE |
|-------|----:|----:|----:|----------:|---------:|
| v1 hierarchical (Normal) | 61% | 85% | 96% | 2.14 | 2.23 |
| v2 skip-Poisson | 86% | 98% | 100% | 2.13 | 2.30 |
| **v2.1 skip-GenPoisson** | **62%** | **87%** | **97%** | **2.14** | **2.23** |

**v2.1 is the best of both:** skip-robust like v2 (same point behavior under the
demo's injected skips) *and* calibrated like v1. It's the current recommended model.

## Cross-dataset check — mcPHASES (hormone-verified)

Same harness, `--dataset mcphases` (128 cycles / 60 subject-intervals; ~2
cycles/user, so almost every prediction is cold-start). 70/30 user-holdout,
18 test users / 22 scored predictions — small, so read as a trend, not a decimal.

| predictor | MAE (d) | RMSE (d) | cold-start MAE (d) |
|-----------|--------:|---------:|-------------------:|
| constant_28 | 2.91 | 3.48 | 2.86 |
| personal_mean | 2.90 | 3.58 | 2.79 |
| personal_median | 2.89 | 3.56 | 2.79 |
| **hierarchical_v1** | **2.06** | 2.83 | **1.88** |
| skip_generative_v2 | 2.07 | 2.78 | 1.91 |
| skip_genpoisson_v2.1 | 2.16 | 2.81 | 2.00 |

**The headline finding:** unlike on FedCycle (where users have ~12 cycles and the
Bayesian model barely edges the baselines), on mcPHASES the hierarchical models
**beat the point baselines decisively** — MAE 2.06 vs 2.90, cold-start 1.88 vs 2.79.
The reason is exactly the mechanism we built for: with only ~2 cycles per user, the
population prior does the heavy lifting that a per-user mean can't. **This is the
real-world regime** — a new app user with one or two logged cycles — and it's where
partial pooling pays off most. mcPHASES also gives LH-verified ovulation (median day
16), the substrate for the future signal-rich model (M5).

## Caveats

- FedCycle is small and unusually regular (NFP users) — absolute MAEs here are lower
  than the ~3.45 d reported on the 2M-cycle Clue set and are **not comparable across
  datasets**. The value is the *relative* ordering and the calibration behavior.
- The v1 model does **not** yet handle skipped-cycle artifacts (rare in FedCycle).
  The skip-aware generative extension (PLAN.md M3 v2) needs a noisier dataset to
  demonstrate its advantage.
