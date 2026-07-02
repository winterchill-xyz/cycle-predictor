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

## Caveats

- FedCycle is small and unusually regular (NFP users) — absolute MAEs here are lower
  than the ~3.45 d reported on the 2M-cycle Clue set and are **not comparable across
  datasets**. The value is the *relative* ordering and the calibration behavior.
- The v1 model does **not** yet handle skipped-cycle artifacts (rare in FedCycle).
  The skip-aware generative extension (PLAN.md M3 v2) needs a noisier dataset to
  demonstrate its advantage.
