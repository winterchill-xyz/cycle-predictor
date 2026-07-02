# Datasets catalogue

Provenance, scale, signals, and **licensing/access** for menstrual-cycle datasets.
Ordered by how easily a team can actually train on them today. Raw files live in
`data/raw/<name>/` (gitignored); reproduce downloads with `scripts/fetch_datasets.py`.

---

## ✅ Openly downloadable

### FedCycle / Marquette  — `data/raw/fedcycle/`
- **What:** ~80 cycle-level variables (SUBCODE, CYCLENUMBER, STDYGROUP, PEAKYES,
  MEANCYCLE, EDO, LLUTEAL, demographics, reproductive history, medication).
  Hundreds of cycles from a Natural Family Planning study (Fehring, 2012).
- **Scale:** small (~290 KB CSV). Good for pipeline/EDA/baselines, too small alone
  for a production model.
- **Landing page:** https://epublications.marquette.edu/data_nfp/7/
- **Files:** SPSS `.sav` (~329 KB) and `FedCycleData071012 (2).csv` (~290 KB) under
  "Additional Files".
- **License:** ⚠ **UNESTABLISHED.** A claim that participants consented to reuse was
  *refuted* in our verification. Fine for private research; **verify terms before any
  product/redistribution.** Do not commit the raw file.
- **Fetch:** `python scripts/fetch_datasets.py --only fedcycle`

### mcPHASES  — `data/raw/mcphases/`
- **What:** multimodal, **hormone-verified** ground truth: daily urinary LH, E3G,
  PdG (Mira Plus) + Fitbit Sense (HR, skin temperature, sleep, respiratory rate,
  activity, stress) + partial Dexcom G6 CGM. Four-phase labels. Built to support ML
  ovulation/fertile-window prediction and non-invasive hormonal-state predictors.
- **Scale:** modest cohort, dense per-subject multimodal time series.
- **Landing page:** https://physionet.org/content/mcphases/  (v1.0.0, DOI 10.13026/zx6a-2c81)
- **Companion paper:** https://www.nature.com/articles/s41597-026-06805-3
- **Access:** **Open but CREDENTIALED** — requires a free PhysioNet account,
  credentialing, and signing a Data Use Agreement. **Not a plain wget.**
- **Fetch:** semi-automated — `scripts/fetch_datasets.py --only mcphases` will use
  your PhysioNet login if provided; otherwise it prints the manual steps.

---

## 🔒 Access-restricted / proprietary (request or partnership)

| Dataset | Scale | Signals | How to get it |
|---------|-------|---------|---------------|
| **Clue** (BioWink) | 186k users / ~2M cycles | cycle length, symptoms | Research collaboration w/ BioWink; see Li/Urteaga papers |
| **Natural Cycles** | 124k users / 612k ovulatory cycles / 17.4M BBT | BBT, LH, age, BMI | Proprietary app data; partnership only |
| **Apple Women's Health Study** | ~664k cycles / 43.7k people | cycle logs + wearable | Access-restricted cohort (Harvard Chan / Apple) |
| **Sympto / Kindara** (Symul 2019) | 212,967 users / 2.7M cycles / 33.7M obs-days | BBT, cervical mucus, cervix position, vaginal sensation, bleeding | From authors **upon request with Sympto & Kindara permission**. Aggregated figure values are public: https://lasy.github.io/FAM-Public-Repo/ |

---

## Adapters

Each dataset gets an adapter under `src/cycle_predictor/data/adapters/` that
normalizes it to the canonical schema in [`../PLAN.md`](../PLAN.md)
(one row per `(user_id, cycle_number)`), writing tidy parquet to `data/processed/`.

## Recommendation

Start on **FedCycle** for the pipeline + baselines now; **apply for mcPHASES
credentialing in parallel** for a signal-rich multimodal set. Plan a request to
one large corpus (Clue / Natural Cycles / Sympto-Kindara) if scaling to production.
