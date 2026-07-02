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

**Known limitation (→ v2.1):** v2's predictive *intervals* over-cover on clean data
(80% → 98%) because a plain **Poisson** likelihood is over-dispersed for cycle
lengths (Poisson sd ≈ √29 ≈ 5.4 d, but real within-user sd ≈ 2.9 d — cycle lengths
are *under*-dispersed). This is exactly why Urteaga et al. use the **Generalized
Poisson** (which fits mean and variance separately). Swapping the likelihood is the
planned v2.1 calibration fix; v2's point-prediction skip-robustness (the headline)
already holds.

## Caveats

- FedCycle is small and unusually regular (NFP users) — absolute MAEs here are lower
  than the ~3.45 d reported on the 2M-cycle Clue set and are **not comparable across
  datasets**. The value is the *relative* ordering and the calibration behavior.
- The v1 model does **not** yet handle skipped-cycle artifacts (rare in FedCycle).
  The skip-aware generative extension (PLAN.md M3 v2) needs a noisier dataset to
  demonstrate its advantage.
