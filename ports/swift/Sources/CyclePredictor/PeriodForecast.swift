// Date wrapper — the app-facing API (mirrors cycle_predictor/api.py). Not covered by
// the kernel golden vectors (dates are language-native), but mirrors the validated ports.
import Foundation

public struct UserLog {
    public let periodStarts: [String]        // "yyyy-MM-dd"
    public let lhTests: [String: Double]?
    public let wearableTemp: [String: Double]?
    public let today: String?
    public init(periodStarts: [String], lhTests: [String: Double]? = nil,
                wearableTemp: [String: Double]? = nil, today: String? = nil) {
        self.periodStarts = periodStarts; self.lhTests = lhTests
        self.wearableTemp = wearableTemp; self.today = today
    }
}

public struct PeriodForecast {
    public let predictedStart, earliest, latest: Date
    public let confidence: Double
    public let daysUntil, nHistory: Int
    public let cycleLengthDays, sdDays: Double
    public let mode: String
}

private let isoFormatter: DateFormatter = {
    let f = DateFormatter()
    f.dateFormat = "yyyy-MM-dd"
    f.timeZone = TimeZone(identifier: "UTC")
    f.locale = Locale(identifier: "en_US_POSIX")
    f.calendar = Calendar(identifier: .gregorian)
    return f
}()

private func asDate(_ s: String) -> Date { isoFormatter.date(from: s)! }
private func dayDiff(_ a: Date, _ b: Date) -> Int {
    Int(((a.timeIntervalSince1970 - b.timeIntervalSince1970) / 86400).rounded())
}
private func addDays(_ d: Date, _ n: Int) -> Date {
    Date(timeIntervalSince1970: d.timeIntervalSince1970 + Double(n) * 86400)
}

// Inverse standard-normal CDF (Acklam) — for the confidence interval only.
func invNormCdf(_ p: Double) -> Double {
    let a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.383577518672690e2, -3.066479806614716e1, 2.506628277459239]
    let b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1]
    let c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783]
    let d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416]
    let pl = 0.02425
    if p < pl {
        let q = (-2 * log(p)).squareRoot()
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    } else if p <= 1 - pl {
        let q = p - 0.5, r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    }
    let q = (-2 * log(1 - p)).squareRoot()
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
}

public func predictNextPeriod(_ log: UserLog, _ params: Params, confidence: Double = 0.8) -> PeriodForecast {
    let starts = log.periodStarts.map(asDate).sorted()
    precondition(!starts.isEmpty, "need at least one period start date")
    let onset = starts.last!
    var history = [Double]()
    for i in 1..<starts.count { history.append(Double(dayDiff(starts[i], starts[i - 1]))) }
    let today = log.today.map(asDate) ?? onset

    func offsets(_ m: [String: Double]?) -> [Int: Double]? {
        guard let m = m else { return nil }
        var out = [Int: Double]()
        for (k, v) in m { let off = dayDiff(asDate(k), onset); if off >= 0 { out[off] = v } }
        return out
    }
    let r = predict(params, history, lhByDay: offsets(log.lhTests), tempByDay: offsets(log.wearableTemp))
    let half = invNormCdf(0.5 + confidence / 2) * r.sd
    let predicted = addDays(onset, Int(r.cycleLength.rounded()))
    return PeriodForecast(
        predictedStart: predicted,
        earliest: addDays(onset, Int((r.cycleLength - half).rounded())),
        latest: addDays(onset, Int((r.cycleLength + half).rounded())),
        confidence: confidence,
        daysUntil: dayDiff(predicted, today),
        nHistory: history.count,
        cycleLengthDays: (r.cycleLength * 10).rounded() / 10,
        sdDays: (r.sd * 10).rounded() / 10,
        mode: r.mode)
}
