# PLAN.md — building the cycle predictor

Roadmap from empty repo → a calibrated next-period predictor. Ordering follows the
evidence in [`research/RESEARCH.md`](research/RESEARCH.md): **generative/Bayesian
first, neural nets as baselines.**

## Problem statement

Given a user's history of cycles (and optional biosignals up to "today"), output a
**predictive distribution over the next period-start date** — hence the next cycle
length in days. Secondary targets: fertile window / ovulation day when BBT/LH/
wearable signals are available.

Primary metric: **MAE (days)** on the next cycle length, plus **calibration**
(does the 80% predictive interval cover ~80%?). We care about calibration because
the literature shows point MAE saturates (~3.4 d) across model families.

## Canonical schema (target of every dataset adapter)

One row per `(user_id, cycle_number)`:

| column | type | notes |
|--------|------|-------|
| `user_id` | str | stable per person |
| `cycle_number` | int | 1..N in chronological order |
| `cycle_start_date` | date | first day of menses |
| `cycle_length_days` | int | start→next start; the primary label |
| `period_length_days` | int | menses duration (optional) |
| `age`, `bmi` | float | optional covariates |
| `bbt_series`, `lh_positive_day`, `symptoms` | optional | for signal-rich datasets |

Adapters: `src/cycle_predictor/data/adapters/{fedcycle,mcphases,...}.py`.
Raw files stay immutable in `data/raw/`; adapters write tidy parquet to
`data/processed/`.

## Milestones

### M0 — Scaffold & data access  ✅ (this pass)
- Repo, docs, downloaders, catalogues. Fetch FedCycle; apply for mcPHASES.

### M1 — Data pipeline & EDA
- FedCycle adapter → canonical schema; notebook EDA (cycle-length distribution,
  per-user variance, regular vs irregular split, obvious skip artifacts as
  bimodal length peaks near 2×).
- Define the **irregular** flag and a **PCOS/irregular** holdout for later.

### M2 — Baselines + evaluation harness
- Baselines: 28-day constant, last-cycle, personal mean, personal median, rolling
  mean (window k).
- Harness: time-ordered per-user splits; **user-holdout** split for cold-start;
  metrics = MAE, RMSE, within-±1/±2-day, interval coverage. Report overall +
  regular/irregular subgroups. This harness is the contract every model plugs into.

### M3 — Bayesian hierarchical generative model (the target)
- Partial-pooling model: population prior → per-user `λ_i` (typical length),
  per-user variability; **latent skipped-cycle count** so an observed length can be
  explained as `(1+s)` true cycles (Truncated-Geometric skip prior, marginalized).
- Implement in PyMC (NumPyro backend for speed). Posterior predictive → calibrated
  intervals. Cold-start users fall back to the population prior.
- Reproduce the qualitative Urteaga/Li result: match baselines on MAE, **win on
  calibration** and early (6–8 day-ahead) accuracy.

### M4 — Baselines to beat/borrow: sequence + GBM
- LSTM/GRU over the cycle-length sequence (+ covariates); LightGBM on engineered
  features (rolling stats, variability, recent deltas, age/BMI). Expected: not
  better than M3 on MAE — used to bound the problem and for ablations.

### M5 — Signal-rich model (optional, needs mcPHASES/BBT)
- State-space / phase-latent model on BBT (Fukaya-style) or HSMM for phase
  labeling (Symul & Holmes). Only once a temperature/hormone dataset is in hand.

### M6 — Packaging
- `predict_next_period(history) -> distribution`; model card documenting metrics
  per subgroup, calibration plots, and **limitations** (PCOS, cold-start, licensing).

## Non-goals / cautions
- Not a medical device; predictions are estimates with uncertainty, never
  contraception guidance.
- **Licensing:** FedCycle reuse terms are unestablished — fine for research,
  verify before any product. Proprietary datasets need signed agreements.
- Privacy: menstrual data is sensitive; keep raw data local and gitignored.

## Immediate next actions
1. `scripts/fetch_datasets.py --only fedcycle` → `data/raw/fedcycle/`.
2. Write the FedCycle adapter + an EDA notebook (M1).
3. Stand up the evaluation harness with baselines (M2) before any fancy model.
