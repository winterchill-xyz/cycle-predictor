# Literature survey — ML for menstrual cycle & ovulation prediction

*Synthesized from a multi-agent deep-research pass (6 search angles → 25 sources
fetched → 121 claims → 25 adversarially verified, 23 confirmed / 2 refuted).
Every claim below carries its source URL; see `papers.csv` for the download
catalogue. Metrics are **self-reported per paper against its own baselines** — no
shared public leaderboard exists, so cross-paper numbers are not apples-to-apples.*

## Executive summary

The frontier for **next-cycle-length / period-start-date** prediction is
**hierarchical Bayesian *generative* models that explicitly model self-tracking
artifacts** — a user forgetting to log a period silently inflates the recorded
cycle length, so the model treats an observed cycle as the sum of `(1 + skipped)`
true cycles. These match or beat RNN/LSTM/CNN on point accuracy while adding
**calibrated uncertainty** and **earlier** prediction. Complementary families:
**state-space models** on basal body temperature (BBT) for onset forecasting, and
**hidden semi-Markov models** for phase/event labeling. The biggest practical
constraint is **data access**: the four largest, most predictive corpora (Clue,
Natural Cycles, Apple WHS, Sympto/Kindara) are proprietary/request-only; only
**mcPHASES** (PhysioNet, credentialed) and the small **FedCycle/Marquette** set
are openly downloadable.

## 1. Models & the key benchmark

### The decisive result — neural nets give no point-accuracy edge
Urteaga et al. 2021 (PMLR v149, *ML for Healthcare*), on the real-world **Clue**
dataset (186,106 users / 2,047,166 cycles), Table 2 point-estimate **MAE (days)**:

| Model | MAE (days) |
|-------|-----------:|
| Poisson generative (Li et al., mode) | **3.451** |
| Generalized-Poisson generative (Proposed, mode) | **3.459** |
| LSTM | 3.626 |
| RNN | 3.846 |
| CNN | 4.379 |

> "black-box neural network architectures do not provide any prediction accuracy
> advantage." The generative model's edge is **calibration** (R² ≈ 0.873 vs 0.803)
> and accuracy **6–8 days before** the next cycle starts.
> PDF: https://proceedings.mlr.press/v149/urteaga21a/urteaga21a.pdf

### State-of-the-art generative / Bayesian hierarchical
- **Li, Urteaga, Shea, Vitzthum, Wiggins, Elhadad — "A predictive model for next
  cycle start date that accounts for adherence in menstrual self-tracking"**,
  JAMIA 2022;29(1):3–11. Latent variables: per-user typical length `λ_i`, skip
  probability `π_i`, skipped-cycle count `s_{i,c}`; observed length = sum of
  `(1+s_{i,c})` true cycles. Supports online updating within a cycle.
  - arXiv: https://arxiv.org/abs/2102.12439 · JAMIA: https://academic.oup.com/jamia/article/29/1/3/6371799 · PMC: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8714275
- **Urteaga et al. — Generalized-Poisson generative model.** Draws skipped-cycle
  count from a Truncated Geometric and marginalizes analytically; the Generalized
  Poisson's two parameters fit mean and variance separately
  (μ=λ/(1−ξ), σ²=λ/(1−ξ)³) → under-dispersed, better-calibrated posteriors.
  - https://proceedings.mlr.press/v149/urteaga21a/urteaga21a.pdf
- **SkipTrack — Duttweiler et al. 2025 (Harvard Chan / Apple).** Bayesian
  hierarchical model of cycle length **and regularity** jointly, accounting for
  skipped bleeding-day logging. Applied to the **Apple Women's Health Study**
  (664,461 cycles / 43,683 individuals after exclusions).
  - https://arxiv.org/abs/2508.05845

### State-space models (BBT-driven onset forecasting)
- **Fukaya, Kawamori, Osada, Kitazawa, Ishiguro — Statistics in Medicine 2017.**
  Menstrual *phase* as a latent state; sequential Bayesian filtering → predictive
  distribution for onset from daily BBT + self-reported menstruation.
  - arXiv: https://arxiv.org/abs/1606.02536 · journal: https://onlinelibrary.wiley.com/doi/full/10.1002/sim.7345
- **de Paula Oliveira, Bruinvels, Pedlar, Moore, Newell — Scientific Reports 2021.**
  Mixed-effect state-space model (random-walk trend + ARMA + covariates), Bayesian
  forecasting; ~**1.64-day in-sample RMSE** (accuracy 0.987) on an *athlete* cohort
  — held-out RMSE can be ~2× (authors' caveat).
  - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8379295/

### Hidden semi-Markov models (phase / event labeling)
- **Symul & Holmes — IEEE J. Biomedical & Health Informatics 2022 (medRxiv 2021).**
  HSMM that adapts to changing tracking behavior, handles state-dependent
  missingness and mixed variable types, quantifies uncertainty. Labeling accuracy
  ~98% (clean) / ~90% (realistic missingness) / **~93% on real partially-labeled data**.
  - PubMed: https://pubmed.ncbi.nlm.nih.gov/34495854/ · preprint: https://www.medrxiv.org/content/10.1101/2021.01.11.21249605v1.full

### Wearable-signal ML (fertile window / menses)
- **Goodale et al. 2019 (Ava bracelet), JMIR** — nightly wearable physiology,
  n=237. https://www.jmir.org/2019/4/e13404/
- **BBT + heart-rate ML, Reproductive Biology & Endocrinology 2022** — fertile
  window ~87% accuracy in regular menstruators; degrades on irregular cycles.
  https://rbej.biomedcentral.com/articles/10.1186/s12958-022-00993-4
- **Li et al. 2020, npj Digital Medicine** — physiological & symptomatic variation
  across cycles (Clue). https://www.nature.com/articles/s41746-020-0269-8

## 2. Datasets (access & licensing)

| Dataset | Scale | Signals | Access | URL |
|---------|-------|---------|--------|-----|
| **FedCycle / Marquette** | small (~80 vars, hundreds of cycles) | cycle metrics, demographics, repro history | **Open download** (CSV ~290 KB). ⚠ reuse licensing *unestablished* | https://epublications.marquette.edu/data_nfp/7/ |
| **mcPHASES** (PhysioNet v1.0.0) | modest cohort, multimodal | LH/E3G/PdG hormones (Mira), Fitbit HR/skin-temp/sleep/resp/stress, Dexcom CGM; hormone-verified 4-phase labels | **Open but credentialed** (PhysioNet account + DUA) | https://physionet.org/content/mcphases/ |
| **Clue** (BioWink) | 186k users / ~2M cycles | cycle lengths, symptoms | Proprietary / request | (paper) https://arxiv.org/abs/2102.12439 |
| **Natural Cycles** | 124k users / 612k ovulatory cycles / 17.4M BBT | BBT, LH, age, BMI | Proprietary | https://www.nature.com/articles/s41746-019-0152-7 · PMC6710244 |
| **Apple Women's Health Study** | ~664k cycles / 43.7k people | cycle logs + wearable | Access-restricted | https://arxiv.org/abs/2508.05845 |
| **Sympto / Kindara** (Symul 2019) | 212,967 users / 2.7M cycles / 33.7M obs-days | 5 sympto-thermal signs (BBT, cervical mucus, cervix position, vaginal sensation, bleeding) | Request (app-vendor permission); aggregated figures public | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6635432 · https://lasy.github.io/FAM-Public-Repo/ |

**Bottom line for a training set today:** start with **FedCycle** (immediate, but
verify licensing before any product use) and apply for **mcPHASES** credentialing
in parallel for a richer multimodal set. Everything larger is request-gated.

## 3. Predictive signals / features (recurring across studies)
- **BBT** — the single most common biosignal (Fukaya, Natural Cycles, Sympto/Kindara, mcPHASES).
- **LH**, plus urinary **E3G / PdG** where hormone-verified ground truth exists (mcPHASES, Natural Cycles).
- **Wearables** — heart rate, skin/wearable temperature, sleep, respiratory rate, activity, stress, CGM glucose (mcPHASES/Fitbit). *(HRV appears only in intended-use framing, not a confirmed feature list.)*
- **Sympto-thermal self-reports** — cervical mucus, cervix position, vaginal sensation, bleeding (Sympto/Kindara).
- **Context** — age, BMI, and the **observed cycle-length history itself**.

## 4. Evaluation & known challenges
- **Metrics:** MAE / RMSE in days; within-±1 / ±2-day hit rate; **calibration**
  (predicted-interval coverage) — the differentiator, since point MAE saturates.
- **Splits:** time-ordered (predict future from past); **user-holdout** to measure
  **cold-start**. Report **regular vs irregular** subgroups separately.
- **Challenges (name them explicitly):** cycle irregularity, **PCOS**, cold-start
  (few/zero logged cycles), data sparsity, and **skip/self-report artifacts** —
  the last is exactly what the generative SOTA models target.

## 5. Open questions (from the survey's gaps)
1. **PCOS / highly-irregular / cold-start** — no confirmed benchmark stratifies on these, despite being core failure modes. Needs targeted evaluation.
2. **Survival analysis, Gaussian processes, gradient boosting** — requested but produced no surviving confirmed claims; a focused follow-up search is warranted (note: Bortot et al. 2006, *Biostatistics*, is a classic Bayesian hierarchical cycle-length reference — https://academic.oup.com/biostatistics/article-abstract/7/1/100/243078).
3. **Wearable-only lift** — does non-invasive (HRV/skin-temp/HR) prediction match BBT+LH, and does an mcPHASES-trained model generalize beyond its small cohort?
4. **Concrete reuse terms** for FedCycle (the consent-for-reuse claim was *refuted*), and realistic request/approval pathways + latency for the proprietary sets.

## Refuted / not-established (do not rely on)
- A "10-state HMM detects ovulation to within ±1.5 days" claim attributed to Symul/Kindara — **refuted (1–2)**.
- "FedCycle is explicitly licensed for reuse with participant consent" — **refuted (0–3)**; treat FedCycle licensing as unknown.
