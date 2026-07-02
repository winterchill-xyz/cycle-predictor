// Validate the Kotlin port against the shared golden vectors (dependency-free JSON reader).
//   kotlinc CyclePredictor.kt Validate.kt -include-runtime -d app.jar
//   java -cp app.jar ValidateKt <artifacts-dir>
import java.io.File
import kotlin.math.abs
import kotlin.system.exitProcess

// --- minimal recursive-descent JSON parser (enough for our artifacts) ---
private class Json(val s: String) {
    var i = 0
    fun parse(): Any? { val v = value(); return v }
    private fun ws() { while (i < s.length && s[i].isWhitespace()) i++ }
    private fun value(): Any? {
        ws()
        return when (s[i]) {
            '{' -> obj(); '[' -> arr(); '"' -> str()
            't' -> { i += 4; true }; 'f' -> { i += 5; false }; 'n' -> { i += 4; null }
            else -> num()
        }
    }
    private fun obj(): Map<String, Any?> {
        val m = LinkedHashMap<String, Any?>(); i++; ws(); if (s[i] == '}') { i++; return m }
        while (true) { ws(); val k = str(); ws(); i++ /* : */; m[k] = value(); ws(); if (s[i] == ',') i++ else { i++; break } }
        return m
    }
    private fun arr(): List<Any?> {
        val a = ArrayList<Any?>(); i++; ws(); if (s[i] == ']') { i++; return a }
        while (true) { a.add(value()); ws(); if (s[i] == ',') i++ else { i++; break } }
        return a
    }
    private fun str(): String {
        val sb = StringBuilder(); i++
        while (s[i] != '"') {
            if (s[i] == '\\') { i++; sb.append(when (s[i]) { 'n' -> '\n'; 't' -> '\t'; else -> s[i] }) } else sb.append(s[i]); i++
        }
        i++; return sb.toString()
    }
    private fun num(): Double { val start = i; while (i < s.length && (s[i].isDigit() || s[i] in "-+.eE")) i++; return s.substring(start, i).toDouble() }
}

@Suppress("UNCHECKED_CAST")
fun main(args: Array<String>) {
    val dir = if (args.isNotEmpty()) args[0] else "../../artifacts"
    val model = Json(File("$dir/model.json").readText()).parse() as Map<String, Any?>
    val bp = model["backbone"] as Map<String, Any?>
    val backbone = Backbone(
        bp["mu_log"] as Double, bp["tau"] as Double, bp["pi"] as Double, bp["xi"] as Double,
        (bp["s_max"] as Double).toInt(), bp["lam_min"] as Double, bp["lam_max"] as Double, (bp["grid"] as Double).toInt(),
    )
    val tp = model["twophase"] as Map<String, Any?>
    val ts = model["thermal_shift"] as Map<String, Any?>
    val params = Params(
        model["version"] as String, backbone,
        TwoPhase(tp["luteal_mean"] as Double, tp["luteal_sd"] as Double),
        model["lh_surge"] as Double,
        ThermalShift((ts["baseline"] as Double).toInt(), (ts["run"] as Double).toInt(),
            ts["delta"] as Double, (ts["search"] as List<Any?>).map { (it as Double).toInt() }),
    )

    val suite = Json(File("$dir/test_vectors.json").readText()).parse() as Map<String, Any?>
    val tol = suite["tolerance"] as Double
    val vectors = suite["vectors"] as List<Map<String, Any?>>
    var pass = 0
    val failures = ArrayList<String>()
    for (v in vectors) {
        val name = v["name"] as String
        val input = v["input"] as Map<String, Any?>
        val history = (input["history"] as List<Any?>).map { it as Double }
        fun dayMap(key: String): Map<Int, Double>? {
            val d = input[key] as? Map<String, Any?> ?: return null
            return d.entries.associate { it.key.toInt() to (it.value as Double) }
        }
        val r = predict(params, history, dayMap("lh_by_day"), dayMap("temp_by_day"))
        val e = v["expected"] as Map<String, Any?>
        val ok = r.mode == e["mode"] && abs(r.cycleLength - (e["cycle_length"] as Double)) < tol &&
            abs(r.sd - (e["sd"] as Double)) < tol
        if (ok) pass++ else failures.add("  FAIL $name: got $r want $e")
    }
    println("Kotlin port: $pass/${vectors.size} golden vectors pass")
    if (failures.isNotEmpty()) { println(failures.joinToString("\n")); exitProcess(1) }
}
