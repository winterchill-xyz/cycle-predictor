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

### M1 — Data pipeline & EDA  ✅
- FedCycle adapter → canonical schema (`src/cycle_predictor/data/fedcycle.py`).
- EDA (`scripts/eda_fedcycle.py`): regular NFP cohort (within-user SD median 2.1 d),
  **19% of users have ≤3 cycles** (cold-start), skip artifacts rare (0.1%).
  Conclusion: point MAE is easy here → **calibration is the goal**.
- TODO: an explicit irregular/PCOS holdout once a more variable dataset is in hand.

### M2 — Baselines + evaluation harness  ✅
- Baselines (`models/baselines.py`): 28-day constant, last-cycle, personal
  mean/median, rolling mean.
- Harness (`eval.py`): user-holdout `split`, point `evaluate` (+ `max_history`
  cold-start slice), probabilistic `evaluate_prob` (interval coverage + sharpness).

### M3 — Hierarchical Bayesian model (v1)  ✅
- Normal-Normal partial pooling (`models/hierarchical.py`): PyMC fits population
  hyperparameters; closed-form conjugate posterior-predictive per user; cold-start
  falls back to the population prior. See RESULTS.md.
- Reproduces the qualitative Urteaga/Li story: **ties/beats baselines on MAE, wins
  on cold-start, and gives ~calibrated intervals** (80% → 85% coverage).
### M3 v2 — skip-aware generative model  ✅
- `models/generative.py`: observed length ~ Poisson((1+s)·λ_i), latent skip count
  `s` ~ Geometric(π) marginalized out; PyMC fits (μ_log, τ, π), per-user 1-D grid
  posterior predicts. Explains a ~2× cycle as a forgotten log instead of a real one.
- Proven by `scripts/demo_skip_robustness.py`: under injected skips v2 holds ~2.3 d
  MAE while v1/naive degrade to 10–12 d at p=0.3. On clean FedCycle v2 ≈ v1.
- **v2.1 (next):** swap the Poisson likelihood for the **Generalized Poisson**
  (Urteaga) — Poisson is over-dispersed for cycle lengths, so v2's intervals
  over-cover (80% → 98%). GP fits mean and variance separately → calibrated.

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
1. **M3 v2.1:** Generalized-Poisson likelihood to fix v2's interval over-coverage.
2. Acquire a more variable dataset (mcPHASES credentialing, or a request to Clue /
   Natural Cycles / Sympto-Kindara) to exercise irregular/PCOS + real skip handling.
3. **M4:** LSTM/GRU + LightGBM baselines (needs torch/lightgbm on the 3.12 venv) to
   confirm they don't beat the generative model on point MAE (per the literature).
4. Per-user skip probability π_i (currently population-level) for v2.
