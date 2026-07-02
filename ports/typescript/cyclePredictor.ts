// Cycle predictor — TypeScript port of the reference kernel (src/cycle_predictor/portable.py).
// Dependency-free; runs in browsers and Node. Validated against artifacts/test_vectors.json
// (see validate.ts). Load params from artifacts/model.json.

export interface BackboneParams {
  type?: string;
  mu_log: number; tau: number; pi: number; xi: number;
  s_max: number; lam_min: number; lam_max: number; grid: number;
}
export interface Params {
  version: string;
  backbone: BackboneParams;
  twophase: { luteal_mean: number; luteal_sd: number };
  lh_surge: number;
  thermal_shift: { baseline: number; run: number; delta: number; search: [number, number] };
}
export type Mode = "history" | "two_phase_lh" | "two_phase_wearable";
export interface KernelResult { cycleLength: number; sd: number; mode: Mode; }

function logsumexp(vals: number[]): number {
  let m = -Infinity;
  for (const v of vals) if (v > m) m = v;
  if (m === -Infinity) return -Infinity;
  let s = 0;
  for (const v of vals) s += Math.exp(v - m);
  return m + Math.log(s);
}

/** Generalized-Poisson posterior-predictive [mean, sd] for the next cycle length. */
export function backbonePredict(bp: BackboneParams, history: number[]): [number, number] {
  const G = bp.grid;
  const lo = Math.log(bp.lam_min), hi = Math.log(bp.lam_max);
  const loglam: number[] = [], lam: number[] = [];
  for (let i = 0; i < G; i++) { const x = lo + (hi - lo) * i / (G - 1); loglam.push(x); lam.push(Math.exp(x)); }

  const { xi, pi, s_max: sMax, mu_log: muLog, tau } = bp;
  const S: number[] = []; for (let s = 0; s <= sMax; s++) S.push(s);
  const logwUn = S.map((s) => Math.log(1 - pi) + s * Math.log(pi));
  const zz = logsumexp(logwUn);
  const logw = logwUn.map((w) => w - zz);
  const mult = S.map((s) => 1 + s);

  const logpost = loglam.map((x) => -0.5 * ((x - muLog) / tau) ** 2 - Math.log(tau) - 0.5 * Math.log(2 * Math.PI));
  for (const d of history) {
    const di = Math.max(1, Math.round(d));
    for (let gi = 0; gi < G; gi++) {
      const terms: number[] = [];
      for (let si = 0; si < S.length; si++) {
        const a = mult[si] * lam[gi];
        const arg = a + di * xi;
        terms.push(arg > 0 ? logw[si] + Math.log(a) + (di - 1) * Math.log(arg) - arg : -Infinity);
      }
      logpost[gi] += logsumexp(terms);
    }
  }
  const z2 = logsumexp(logpost);
  let elam = 0, elam2 = 0;
  for (let i = 0; i < G; i++) { const p = Math.exp(logpost[i] - z2); elam += p * lam[i]; elam2 += p * lam[i] * lam[i]; }
  const varLam = Math.max(elam2 - elam * elam, 0);
  const c1 = 1 / (1 - xi), c3 = 1 / (1 - xi) ** 3;
  return [elam * c1, Math.sqrt(elam * c3 + varLam * c1 * c1)];
}

function toDayMap(obj: Record<string | number, number> | null | undefined): Map<number, number> {
  const m = new Map<number, number>();
  if (obj) for (const [k, v] of Object.entries(obj)) if (v != null) m.set(Number(k), v);
  return m;
}

export function detectThermalShift(temp: Map<number, number>, ts: Params["thermal_shift"]): number | null {
  const { baseline, run, delta } = ts; const [lo, hi] = ts.search;
  if (temp.size < baseline + run) return null;
  for (let d = lo; d <= hi; d++) {
    const base: number[] = []; for (let k = d - baseline; k < d; k++) if (temp.has(k)) base.push(temp.get(k)!);
    const fut: number[] = []; for (let j = 0; j < run; j++) if (temp.has(d + j)) fut.push(temp.get(d + j)!);
    if (base.length < baseline - 1 || fut.length < run) continue;
    const cover = base.reduce((a, b) => a + b, 0) / base.length + delta;
    if (fut.every((t) => t > cover)) return Math.max(d - 1, 0);
  }
  return null;
}

/** Numpy-free equivalent of UnifiedPredictor.predict. Signals keyed by day-of-cycle. */
export function predict(
  params: Params, history: number[],
  lhByDay?: Record<string | number, number> | null,
  tempByDay?: Record<string | number, number> | null,
): KernelResult {
  const lh = toDayMap(lhByDay);
  if (lh.size) {
    const surge = [...lh.entries()].filter(([, v]) => v >= params.lh_surge).map(([d]) => d).sort((a, b) => a - b);
    if (surge.length) return { cycleLength: surge[0] + params.twophase.luteal_mean, sd: params.twophase.luteal_sd, mode: "two_phase_lh" };
  }
  const temp = toDayMap(tempByDay);
  if (temp.size) {
    const est = detectThermalShift(temp, params.thermal_shift);
    if (est !== null) return { cycleLength: est + params.twophase.luteal_mean, sd: params.twophase.luteal_sd, mode: "two_phase_wearable" };
  }
  const [mean, sd] = backbonePredict(params.backbone, history);
  return { cycleLength: mean, sd, mode: "history" };
}

// --- date wrapper (mirrors cycle_predictor/api.py; not covered by the kernel vectors) ---
export interface UserLog {
  periodStarts: (string | Date)[];
  lhTests?: Record<string, number>;
  wearableTemp?: Record<string, number>;
  today?: string | Date;
}
export interface PeriodForecast {
  predictedStart: Date; earliest: Date; latest: Date; confidence: number;
  daysUntil: number; cycleLengthDays: number; sdDays: number; mode: Mode; nHistory: number;
}

const DAY = 86400000;
const asDate = (d: string | Date): Date => (d instanceof Date ? d : new Date(d + "T00:00:00Z"));
const dayDiff = (a: Date, b: Date) => Math.round((a.getTime() - b.getTime()) / DAY);
const addDays = (d: Date, n: number) => new Date(d.getTime() + n * DAY);

// Inverse standard-normal CDF (Acklam) — for the confidence interval only.
function invNormCdf(p: number): number {
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.383577518672690e2, -3.066479806614716e1, 2.506628277459239];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
  const pl = 0.02425;
  if (p < pl) { const q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  if (p <= 1 - pl) { const q = p - 0.5, r = q * q; return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1); }
  const q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
}

export function predictNextPeriod(log: UserLog, params: Params, confidence = 0.8): PeriodForecast {
  const starts = log.periodStarts.map(asDate).sort((a, b) => a.getTime() - b.getTime());
  if (!starts.length) throw new Error("need at least one period start date");
  const onset = starts[starts.length - 1];
  const history: number[] = [];
  for (let i = 1; i < starts.length; i++) history.push(dayDiff(starts[i], starts[i - 1]));
  const today = log.today ? asDate(log.today) : onset;

  const offsets = (m?: Record<string, number>) => {
    if (!m) return null;
    const out: Record<number, number> = {};
    for (const [d, v] of Object.entries(m)) { const off = dayDiff(asDate(d), onset); if (off >= 0) out[off] = v; }
    return out;
  };
  const r = predict(params, history, offsets(log.lhTests), offsets(log.wearableTemp));
  const z = invNormCdf(0.5 + confidence / 2);
  const half = z * r.sd;
  return {
    predictedStart: addDays(onset, Math.round(r.cycleLength)),
    earliest: addDays(onset, Math.round(r.cycleLength - half)),
    latest: addDays(onset, Math.round(r.cycleLength + half)),
    confidence,
    daysUntil: dayDiff(addDays(onset, Math.round(r.cycleLength)), today),
    cycleLengthDays: Math.round(r.cycleLength * 10) / 10,
    sdDays: Math.round(r.sd * 10) / 10,
    mode: r.mode,
    nHistory: history.length,
  };
}
