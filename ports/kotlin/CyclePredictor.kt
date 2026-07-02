// Cycle predictor — Kotlin port of the reference kernel (src/cycle_predictor/portable.py).
// Pure Kotlin stdlib (+ java.time for the date wrapper); suits Android and JVM backends.
// Validated against artifacts/test_vectors.json in CI (see ports/kotlin/Validate.kt).
import java.time.LocalDate
import java.time.temporal.ChronoUnit
import kotlin.math.PI
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sqrt

data class Backbone(
    val muLog: Double, val tau: Double, val pi: Double, val xi: Double,
    val sMax: Int, val lamMin: Double, val lamMax: Double, val grid: Int,
)
data class TwoPhase(val lutealMean: Double, val lutealSd: Double)
data class ThermalShift(val baseline: Int, val run: Int, val delta: Double, val search: List<Int>)
data class Params(
    val version: String, val backbone: Backbone, val twophase: TwoPhase,
    val lhSurge: Double, val thermalShift: ThermalShift,
)
data class KernelResult(val cycleLength: Double, val sd: Double, val mode: String)

private fun logsumexp(vals: List<Double>): Double {
    var m = Double.NEGATIVE_INFINITY
    for (v in vals) if (v > m) m = v
    if (m == Double.NEGATIVE_INFINITY) return m
    var s = 0.0
    for (v in vals) s += exp(v - m)
    return m + ln(s)
}

/** Generalized-Poisson posterior-predictive (mean, sd) for the next cycle length. */
fun backbonePredict(bp: Backbone, history: List<Double>): Pair<Double, Double> {
    val g = bp.grid
    val lo = ln(bp.lamMin); val hi = ln(bp.lamMax)
    val loglam = DoubleArray(g); val lam = DoubleArray(g)
    for (i in 0 until g) { val x = lo + (hi - lo) * i / (g - 1); loglam[i] = x; lam[i] = exp(x) }

    val xi = bp.xi; val pi = bp.pi
    val s = (0..bp.sMax).toList()
    val logwUn = s.map { ln(1 - pi) + it * ln(pi) }
    val zz = logsumexp(logwUn)
    val logw = logwUn.map { it - zz }
    val mult = s.map { (1 + it).toDouble() }

    val logpost = DoubleArray(g) { -0.5 * ((loglam[it] - bp.muLog) / bp.tau).pow(2) - ln(bp.tau) - 0.5 * ln(2 * PI) }
    for (d in history) {
        val di = maxOf(1, d.roundToInt()).toDouble()
        for (gi in 0 until g) {
            val terms = ArrayList<Double>(s.size)
            for (si in s.indices) {
                val a = mult[si] * lam[gi]
                val arg = a + di * xi
                terms.add(if (arg > 0) logw[si] + ln(a) + (di - 1) * ln(arg) - arg else Double.NEGATIVE_INFINITY)
            }
            logpost[gi] += logsumexp(terms)
        }
    }
    val z2 = logsumexp(logpost.toList())
    var elam = 0.0; var elam2 = 0.0
    for (i in 0 until g) { val p = exp(logpost[i] - z2); elam += p * lam[i]; elam2 += p * lam[i] * lam[i] }
    val varLam = maxOf(elam2 - elam * elam, 0.0)
    val c1 = 1.0 / (1.0 - xi); val c3 = 1.0 / (1.0 - xi).pow(3)
    return Pair(elam * c1, sqrt(elam * c3 + varLam * c1 * c1))
}

fun detectThermalShift(temp: Map<Int, Double>, ts: ThermalShift): Int? {
    val baseline = ts.baseline; val run = ts.run
    val lo = ts.search[0]; val hi = ts.search[1]
    if (temp.size < baseline + run) return null
    for (d in lo..hi) {
        val base = ArrayList<Double>(); for (k in (d - baseline) until d) temp[k]?.let { base.add(it) }
        val fut = ArrayList<Double>(); for (j in 0 until run) temp[d + j]?.let { fut.add(it) }
        if (base.size < baseline - 1 || fut.size < run) continue
        val cover = base.sum() / base.size + ts.delta
        if (fut.all { it > cover }) return maxOf(d - 1, 0)
    }
    return null
}

/** Numpy-free equivalent of UnifiedPredictor.predict. Signals keyed by day-of-cycle. */
fun predict(
    params: Params, history: List<Double>,
    lhByDay: Map<Int, Double>? = null, tempByDay: Map<Int, Double>? = null,
): KernelResult {
    if (!lhByDay.isNullOrEmpty()) {
        val surge = lhByDay.filter { it.value >= params.lhSurge }.keys.sorted()
        if (surge.isNotEmpty())
            return KernelResult(surge[0] + params.twophase.lutealMean, params.twophase.lutealSd, "two_phase_lh")
    }
    if (!tempByDay.isNullOrEmpty()) {
        val est = detectThermalShift(tempByDay, params.thermalShift)
        if (est != null)
            return KernelResult(est + params.twophase.lutealMean, params.twophase.lutealSd, "two_phase_wearable")
    }
    val (mean, sd) = backbonePredict(params.backbone, history)
    return KernelResult(mean, sd, "history")
}

// --- date wrapper (app-facing; mirrors cycle_predictor/api.py, not in the kernel vectors) ---
data class UserLog(
    val periodStarts: List<String>,
    val lhTests: Map<String, Double>? = null,
    val wearableTemp: Map<String, Double>? = null,
    val today: String? = null,
)
data class PeriodForecast(
    val predictedStart: LocalDate, val earliest: LocalDate, val latest: LocalDate,
    val confidence: Double, val daysUntil: Int, val nHistory: Int,
    val cycleLengthDays: Double, val sdDays: Double, val mode: String,
)

// Inverse standard-normal CDF (Acklam) — for the confidence interval only.
fun invNormCdf(p: Double): Double {
    val a = doubleArrayOf(-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.383577518672690e2, -3.066479806614716e1, 2.506628277459239)
    val b = doubleArrayOf(-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1)
    val c = doubleArrayOf(-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783)
    val d = doubleArrayOf(7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416)
    val pl = 0.02425
    return when {
        p < pl -> { val q = sqrt(-2 * ln(p)); (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1) }
        p <= 1 - pl -> { val q = p - 0.5; val r = q * q; (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1) }
        else -> { val q = sqrt(-2 * ln(1 - p)); -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1) }
    }
}

fun predictNextPeriod(log: UserLog, params: Params, confidence: Double = 0.8): PeriodForecast {
    val starts = log.periodStarts.map { LocalDate.parse(it) }.sorted()
    require(starts.isNotEmpty()) { "need at least one period start date" }
    val onset = starts.last()
    val history = (1 until starts.size).map { ChronoUnit.DAYS.between(starts[it - 1], starts[it]).toDouble() }
    val today = log.today?.let { LocalDate.parse(it) } ?: onset

    fun offsets(m: Map<String, Double>?): Map<Int, Double>? {
        if (m == null) return null
        val out = HashMap<Int, Double>()
        for ((k, v) in m) { val off = ChronoUnit.DAYS.between(onset, LocalDate.parse(k)).toInt(); if (off >= 0) out[off] = v }
        return out
    }
    val r = predict(params, history, offsets(log.lhTests), offsets(log.wearableTemp))
    val half = invNormCdf(0.5 + confidence / 2) * r.sd
    val predicted = onset.plusDays(r.cycleLength.roundToInt().toLong())
    return PeriodForecast(
        predictedStart = predicted,
        earliest = onset.plusDays((r.cycleLength - half).roundToInt().toLong()),
        latest = onset.plusDays((r.cycleLength + half).roundToInt().toLong()),
        confidence = confidence,
        daysUntil = ChronoUnit.DAYS.between(today, predicted).toInt(),
        nHistory = history.size,
        cycleLengthDays = (r.cycleLength * 10).roundToInt() / 10.0,
        sdDays = (r.sd * 10).roundToInt() / 10.0,
        mode = r.mode,
    )
}
