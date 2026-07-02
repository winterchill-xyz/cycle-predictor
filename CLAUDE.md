# CLAUDE.md — cycle-predictor

Working brief for AI-assisted development of an **ML menstrual cycle predictor**.
Read this first; it captures decisions that are not obvious from the code.

## Goal

Predict a user's **next menstrual period start date** (primary target) and, where
biosignals allow, the **fertile window / ovulation day**, from their history of
past cycles + optional signals — outputting a **calibrated predictive
distribution**, not only a point estimate.

## What the research says (drives our modeling) — full survey in `research/RESEARCH.md`

- **SOTA = hierarchical Bayesian *generative* models** that explicitly model
  self-tracking artifacts (a user skipping/forgetting a period log inflates the
  recorded cycle length). Li/Urteaga et al. (JAMIA 2022; arXiv 2102.12439),
  Urteaga et al. (PMLR v149, 2021), and SkipTrack (arXiv 2508.05845, 2025).
- **Neural nets are not better on point accuracy.** On ~2M Clue cycles: generative
  Poisson/Generalized-Poisson ≈ **3.45 d MAE**, vs LSTM 3.63, RNN 3.85, CNN 4.38.
  The generative win is **calibration** + accuracy **6–8 days earlier**. ⇒ build
  Bayesian/generative first; LSTM/GBM are baselines.
- **Complementary families:** state-space models on basal body temperature (BBT)
  — Fukaya 2017 (arXiv 1606.02536), de Paula Oliveira 2021 (PMC8379295); hidden
  semi-Markov models for phase/event labeling — Symul & Holmes 2022 (PMID 34495854).
- **Predictive signals:** BBT and LH (+ urinary E3G/PdG); wearable temperature /
  heart rate / sleep / respiratory / stress (mcPHASES/Fitbit); sympto-thermal
  self-reports (cervical mucus, cervix position); cycle-length history; age/BMI.
- **Hard parts (name them in eval):** cycle irregularity, PCOS, cold-start (few/no
  logged cycles), data sparsity, and self-report/skip artifacts.

## Modeling order (see `PLAN.md`)

1. Baselines: 28-day constant, last-cycle, personal mean/median, rolling mean.
2. Bayesian hierarchical model with partial pooling (population prior → per-user),
   handling skipped cycles as a latent count. Cold-start falls back to the prior.
3. Baselines to beat/borrow from: LSTM/GRU over cycle sequences; LightGBM on
   engineered features.
4. Optional: state-space / BBT model when temperature data is present.

## Canonical data schema

Every dataset is normalized by an adapter to **one row per `(user_id, cycle_number)`**:
`cycle_start_date, cycle_length_days, period_length_days`, plus optional covariates
(`age, bmi, bbt_series, lh_positive_day, symptoms…`). Adapters live in
`src/cycle_predictor/data/adapters/`. Keep raw files immutable in `data/raw/`.

## Data (details + licensing in `data/DATASETS.md`)

- **Openly downloadable:** FedCycle/Marquette CSV (`epublications.marquette.edu/data_nfp/7`,
  ~290 KB, ~80 vars) — ⚠ reuse licensing *unestablished*, verify before productizing.
  mcPHASES (`physionet.org/content/mcphases`) — multimodal, hormone-verified, but needs
  free PhysioNet **credentialing + data use agreement** (not a plain download).
- **Access-restricted / proprietary:** Clue (~2M cycles), Natural Cycles (~600k),
  Apple Women's Health Study (~664k), Sympto/Kindara (~2.7M). Request-only.

## Downloading papers & datasets

Credentials live in **`../winterchill/.env`** (`BRIGHTDATA_*` Web Unlocker,
`OPENROUTER_API_KEY`). Copy into `.env` or pass `--env ../winterchill/.env`.

- `scripts/fetch_pdfs.py` — resolves open-access PDFs via arXiv / Europe PMC /
  OpenAlex / Unpaywall, downloads through **BrightData Web Unlocker**, verifies the
  `%PDF` magic, writes to `research/papers/` + a manifest. Falls back to
  **OpenRouter/Perplexity** to discover a PDF link when deterministic routes miss.
- `scripts/fetch_datasets.py` — downloads openly-available datasets into `data/raw/`.

### ⚠ Two gotchas that will waste your time otherwise

1. **BrightData proxy needs the runner's egress IP whitelisted** in the
   `web_unlocker1` zone. Symptom: HTTP 407 `ip_forbidden` naming the exact IP —
   add that IP at brightdata.com → zone → Allowed IPs.
2. **Run downloaders OUTSIDE the tool sandbox.** The sandbox blocks the proxy
   CONNECT tunnel and file writes (symptom: HTTP 000 / empty files that silently
   don't appear). With Bash, set `dangerouslyDisableSandbox: true` for fetches.

## Environment (important)

System Python here is **3.14**, which lacks wheels for the heavy libs. The project
venv is **Python 3.12** (full wheel support), created with `uv`:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy scipy pandas pymc arviz pytest ruff
.venv/bin/python scripts/eval_models.py   # PyMC fit; --fast for method-of-moments
.venv/bin/python -m pytest -q             # 14 tests (incl a PyMC smoke test)
```

The stdlib-only parts (data adapter, eval harness, baselines, downloaders) run on
any Python; only the PyMC fit needs the venv (it falls back to a method-of-moments
fit if PyMC is absent). Current results live in `RESULTS.md`.

## Conventions

- Library code under `src/cycle_predictor`, importable as `cycle_predictor`
  (`pyproject.toml` sets `pythonpath=src` for pytest; `pip install -e .` also works).
- **Never commit** `.env`, raw data, or PDFs — all gitignored. Commit the
  *catalogues* (`research/papers.csv`, `data/DATASETS.md`) and *fetch scripts* so
  downloads are reproducible.
- **Commit small and often; push after every commit.** Remote:
  `winterchill-xyz/cycle-predictor` over HTTPS, authenticated as gh account
  `viatsko` (the SSH key maps to `rayanatrades`, which is denied — use https).
