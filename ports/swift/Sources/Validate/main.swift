// Validate the Swift port against the shared golden vectors.
//   swift run --package-path ports/swift Validate <artifacts-dir>
import Foundation
import CyclePredictor

let artDir = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "../../artifacts"
let params = try Params.load(from: URL(fileURLWithPath: artDir + "/model.json"))

let vecData = try Data(contentsOf: URL(fileURLWithPath: artDir + "/test_vectors.json"))
let suite = try JSONSerialization.jsonObject(with: vecData) as! [String: Any]
let tol = (suite["tolerance"] as! NSNumber).doubleValue
let vectors = suite["vectors"] as! [[String: Any]]

var pass = 0
var failures = [String]()
for v in vectors {
    let name = v["name"] as! String
    let input = v["input"] as! [String: Any]
    let history = (input["history"] as! [Any]).map { ($0 as! NSNumber).doubleValue }
    func dayMap(_ key: String) -> [Int: Double]? {
        guard let d = input[key] as? [String: Any] else { return nil }
        var out = [Int: Double]()
        for (k, val) in d { out[Int(k)!] = (val as! NSNumber).doubleValue }
        return out
    }
    let r = predict(params, history, lhByDay: dayMap("lh_by_day"), tempByDay: dayMap("temp_by_day"))
    let e = v["expected"] as! [String: Any]
    let mode = e["mode"] as! String
    let cl = (e["cycle_length"] as! NSNumber).doubleValue
    let sd = (e["sd"] as! NSNumber).doubleValue
    if r.mode == mode && abs(r.cycleLength - cl) < tol && abs(r.sd - sd) < tol {
        pass += 1
    } else {
        failures.append("  FAIL \(name): got (\(r.mode), \(r.cycleLength), \(r.sd)) want (\(mode), \(cl), \(sd))")
    }
}
print("Swift port: \(pass)/\(vectors.count) golden vectors pass")
if !failures.isEmpty { print(failures.joined(separator: "\n")); exit(1) }
