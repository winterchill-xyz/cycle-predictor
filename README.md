# cycle-predictor

Machine-learning predictor for the **menstrual cycle**: forecast the next period
start date (and, where signals allow, the fertile window / ovulation day) from a
user's history of past cycles plus optional biosignals — with **calibrated
uncertainty**, not just a point guess.

> Status: bootstrapping. See [`PLAN.md`](PLAN.md) for the roadmap and
> [`research/RESEARCH.md`](research/RESEARCH.md) for the literature survey that
> informs the modeling choices.

## Why the design looks like it does

The literature is clear (see the survey): the state of the art for next-cycle
prediction is **hierarchical Bayesian *generative* models** that explicitly model
self-tracking artifacts — users forgetting to log a period, which silently
doubles a recorded cycle length. On the largest benchmark (Urteaga et al. 2021,
~2M Clue cycles) neural nets give **no point-accuracy advantage** over these
generative models (~3.45 d MAE for both); the payoff is **calibration** and
**earlier** (6–8 days ahead) accuracy. So we build generative/Bayesian models
first and treat LSTMs as baselines, not the target.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Credentials for the downloaders (BrightData + OpenRouter) live in ../winterchill/.env
cp .env.example .env   # then fill in, or point the scripts at ../winterchill/.env

# Fetch the openly-downloadable training data
python scripts/fetch_datasets.py --only fedcycle

# Fetch open-access PDFs of the key papers
python scripts/fetch_pdfs.py --catalogue research/papers.csv
```

## Layout

| Path | What |
|------|------|
| `src/cycle_predictor/` | library: data schema, feature engineering, models, evaluation |
| `data/` | `raw/` downloaded datasets (gitignored), `processed/`, `DATASETS.md` catalogue |
| `research/` | `RESEARCH.md` survey, `papers.csv` catalogue, `papers/` PDFs (gitignored) |
| `scripts/` | `fetch_pdfs.py`, `fetch_datasets.py` — BrightData-backed downloaders |
| `notebooks/` | exploration |
| `PLAN.md` | modeling plan & milestones |
| `CLAUDE.md` | working notes / conventions for AI-assisted development |

## Data at a glance

**Openly downloadable:** FedCycle/Marquette (~290 KB CSV) and mcPHASES (PhysioNet,
free credentialing required). **Access-restricted:** Clue, Natural Cycles, Apple
Women's Health Study, Sympto/Kindara. Full provenance and licensing notes in
[`data/DATASETS.md`](data/DATASETS.md).
