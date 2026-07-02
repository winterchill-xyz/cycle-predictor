# Porting the predictor to other languages

**TL;DR — don't build a C++ core.** This model is ~4 population constants and a page
of elementary math, with no heavy runtime. The most portable, private, offline-
capable shape is: **one params file + a small native inference kernel per language,
locked by shared golden vectors.** C++/FFI or WASM would be *more* work and *less*
portable here (Swift interop, emscripten, JNI, pybind — a build matrix for
arithmetic). Reach for a compiled core only if the model later grows heavy
(neural nets, big matmuls).

## The three artifacts (the contract)

| Artifact | What | Regenerate |
|----------|------|-----------|
| `artifacts/model.json` | the shipped parameters (single source of truth) | `scripts/export_model.py` |
| `artifacts/test_vectors.json` | input → expected `{mode, cycle_length, sd}` | `scripts/gen_test_vectors.py` |
| `src/cycle_predictor/portable.py` | numpy-free **reference kernel** to port from | — |

`portable.py` is byte-for-byte equivalent to the numpy research models
(`tests/test_portable.py` proves it), so a port validated against these vectors is
validated against the real model.

## What to port (all of `portable.py`, ~150 lines)

1. **`backbone_predict(bp, history)`** — the only non-trivial piece. A grid over
   `log λ` (size `bp.grid`), a Normal prior `N(mu_log, tau)` on `log λ`, and a
   skip-marginalized Generalized-Poisson likelihood per past cycle length; returns
   `(mean, sd)` of the next cycle length. Needs only `log`, `exp`, and a numerically
   stable `logsumexp`.
2. **`detect_thermal_shift(temp_by_day, ts)`** — a few comparisons (coverline rule).
3. **`predict(params, history, lh_by_day, temp_by_day)`** — routing: LH surge →
   two-phase (`ovulation + luteal_mean`, sd `luteal_sd`); else thermal shift →
   two-phase; else backbone. Trivial once (1) is done.

Signals are keyed by **day-of-cycle** (int, 0 = menses onset), so the kernel is
date-library-agnostic.

## The thin date wrapper (per language, trivial)

`predict_next_period` in Python (`cycle_predictor/api.py`) is the reference for the
wrapper each client adds around the kernel:

1. `history = diffs(sorted(period_starts))`; `onset = last(period_starts)`.
2. Convert any `{date: LH}` / `{date: temp}` to `{(date - onset).days: value}` for
   offsets ≥ 0.
3. `r = predict(params, history, lh_by_day, temp_by_day)`.
4. `predicted = onset + round(r.cycle_length)` days.
   `half = z(confidence) * r.sd` (z from the normal quantile; 0.8 → 1.2816).
   `interval = [onset + round(r.cycle_length - half), onset + round(r.cycle_length + half)]`.

## Validating a port

Load `model.json`, run every vector in `test_vectors.json` through your `predict`,
and assert `mode` matches and `|cycle_length - expected|`, `|sd - expected|` are
within `tolerance` (1e-4). If all 12 pass, your kernel is correct.

## Recommended targets

- **Swift** (iOS): native struct + `Foundation`; on-device, offline, private —
  menstrual data never leaves the phone.
- **TypeScript** (web frontend **and** Node backends): one port covers both.
- **Python** backend: already have it (`portable.py`, no numpy needed).

Same `model.json`, same vectors, everywhere. Bump `version` in `portable.py` when the
params change and regenerate both artifacts.
