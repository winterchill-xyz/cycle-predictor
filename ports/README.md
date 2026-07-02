# Language ports

Native, dependency-free implementations of the inference kernel for on-device,
offline, private prediction. Each loads [`../artifacts/model.json`](../artifacts/model.json)
and must reproduce every vector in
[`../artifacts/test_vectors.json`](../artifacts/test_vectors.json). The canonical
spec is [`../src/cycle_predictor/portable.py`](../src/cycle_predictor/portable.py);
see [`../PORTING.md`](../PORTING.md).

| Port | Target | Validate locally | CI |
|------|--------|------------------|----|
| `typescript/` | web frontend + Node backends | `node --experimental-strip-types ports/typescript/validate.ts` | ✅ |
| `swift/` | iOS (Foundation) | `swift run --package-path ports/swift Validate "$PWD/artifacts"` | ✅ |
| `kotlin/` | Android + JVM backends | `cd ports/kotlin && kotlinc *.kt -include-runtime -d app.jar && java -cp app.jar ValidateKt ../../artifacts` | ✅ |

All three are validated in GitHub Actions against the shared vectors (Kotlin is
validated in CI — no JVM was available where it was written). Each port exposes both
the kernel (`predict`) and the date wrapper (`predictNextPeriod`, dates in → date +
interval out), mirroring `cycle_predictor/api.py`.

When the model changes: bump `version` in `portable.py`, regenerate the artifacts
(`scripts/export_model.py`, `scripts/gen_test_vectors.py`), and CI re-checks every
port automatically.
