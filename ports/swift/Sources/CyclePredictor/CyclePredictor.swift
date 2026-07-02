// Cycle predictor — Swift port of the reference kernel (src/cycle_predictor/portable.py).
// Pure Foundation; on-device, offline. Validated against artifacts/test_vectors.json.
import Foundation

public struct Backbone: Codable {
    public let muLog, tau, pi, xi, lamMin, lamMax: Double
    public let sMax, grid: Int
}
public struct TwoPhase: Codable { public let lutealMean, lutealSd: Double }
public struct ThermalShift: Codable {
    public let baseline, run: Int
    public let delta: Double
    public let search: [Int]
}
public struct Params: Codable {
    public let version: String
    public let backbone: Backbone
    public let twophase: TwoPhase
    public let lhSurge: Double
    public let thermalShift: ThermalShift

    public static func load(from url: URL) throws -> Params {
        let dec = JSONDecoder()
        dec.keyDecodingStrategy = .convertFromSnakeCase
        return try dec.decode(Params.self, from: Data(contentsOf: url))
    }
}

public struct KernelResult { public let cycleLength, sd: Double; public let mode: String }

@inline(__always) func logsumexp(_ vals: [Double]) -> Double {
    var m = -Double.infinity
    for v in vals where v > m { m = v }
    if m == -Double.infinity { return m }
    var s = 0.0
    for v in vals { s += exp(v - m) }
    return m + log(s)
}

/// Generalized-Poisson posterior-predictive (mean, sd) for the next cycle length.
public func backbonePredict(_ bp: Backbone, _ history: [Double]) -> (Double, Double) {
    let G = bp.grid
    let lo = log(bp.lamMin), hi = log(bp.lamMax)
    var loglam = [Double](repeating: 0, count: G), lam = [Double](repeating: 0, count: G)
    for i in 0..<G { let x = lo + (hi - lo) * Double(i) / Double(G - 1); loglam[i] = x; lam[i] = exp(x) }

    let xi = bp.xi, pi = bp.pi
    let S = Array(0...bp.sMax)
    let logwUn = S.map { log(1 - pi) + Double($0) * log(pi) }
    let zz = logsumexp(logwUn)
    let logw = logwUn.map { $0 - zz }
    let mult = S.map { Double(1 + $0) }

    var logpost = loglam.map { -0.5 * pow(($0 - bp.muLog) / bp.tau, 2) - log(bp.tau) - 0.5 * log(2 * Double.pi) }
    for d in history {
        let di = Double(max(1, Int(d.rounded())))
        for gi in 0..<G {
            var terms = [Double](repeating: 0, count: S.count)
            for si in 0..<S.count {
                let a = mult[si] * lam[gi]
                let arg = a + di * xi
                terms[si] = arg > 0 ? logw[si] + log(a) + (di - 1) * log(arg) - arg : -Double.infinity
            }
            logpost[gi] += logsumexp(terms)
        }
    }
    let z2 = logsumexp(logpost)
    var elam = 0.0, elam2 = 0.0
    for i in 0..<G { let p = exp(logpost[i] - z2); elam += p * lam[i]; elam2 += p * lam[i] * lam[i] }
    let varLam = max(elam2 - elam * elam, 0)
    let c1 = 1 / (1 - xi), c3 = 1 / pow(1 - xi, 3)
    return (elam * c1, (elam * c3 + varLam * c1 * c1).squareRoot())
}

public func detectThermalShift(_ temp: [Int: Double], _ ts: ThermalShift) -> Int? {
    let baseline = ts.baseline, run = ts.run
    let lo = ts.search[0], hi = ts.search[1]
    if temp.count < baseline + run { return nil }
    for d in lo...hi {
        var base = [Double](), fut = [Double]()
        for k in (d - baseline)..<d { if let v = temp[k] { base.append(v) } }
        for j in 0..<run { if let v = temp[d + j] { fut.append(v) } }
        if base.count < baseline - 1 || fut.count < run { continue }
        let cover = base.reduce(0, +) / Double(base.count) + ts.delta
        if fut.allSatisfy({ $0 > cover }) { return max(d - 1, 0) }
    }
    return nil
}

/// Numpy-free equivalent of UnifiedPredictor.predict. Signals keyed by day-of-cycle.
public func predict(_ params: Params, _ history: [Double],
                    lhByDay: [Int: Double]? = nil, tempByDay: [Int: Double]? = nil) -> KernelResult {
    if let lh = lhByDay, !lh.isEmpty {
        let surge = lh.filter { $0.value >= params.lhSurge }.keys.sorted()
        if let first = surge.first {
            return KernelResult(cycleLength: Double(first) + params.twophase.lutealMean,
                                sd: params.twophase.lutealSd, mode: "two_phase_lh")
        }
    }
    if let temp = tempByDay, !temp.isEmpty, let est = detectThermalShift(temp, params.thermalShift) {
        return KernelResult(cycleLength: Double(est) + params.twophase.lutealMean,
                            sd: params.twophase.lutealSd, mode: "two_phase_wearable")
    }
    let (mean, sd) = backbonePredict(params.backbone, history)
    return KernelResult(cycleLength: mean, sd: sd, mode: "history")
}
