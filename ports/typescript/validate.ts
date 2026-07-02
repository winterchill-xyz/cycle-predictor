// Validate the TypeScript port against the shared golden vectors.
//   node --experimental-strip-types ports/typescript/validate.ts
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { predict, type Params } from "./cyclePredictor.ts";

const artifacts = fileURLToPath(new URL("../../artifacts/", import.meta.url));
const params = JSON.parse(readFileSync(artifacts + "model.json", "utf8")) as Params;
const suite = JSON.parse(readFileSync(artifacts + "test_vectors.json", "utf8"));

let pass = 0;
const failures: string[] = [];
for (const v of suite.vectors) {
  const r = predict(params, v.input.history, v.input.lh_by_day, v.input.temp_by_day);
  const e = v.expected;
  const ok = r.mode === e.mode
    && Math.abs(r.cycleLength - e.cycle_length) < suite.tolerance
    && Math.abs(r.sd - e.sd) < suite.tolerance;
  if (ok) pass++;
  else failures.push(`  FAIL ${v.name}: got ${JSON.stringify(r)} want ${JSON.stringify(e)}`);
}
console.log(`TypeScript port: ${pass}/${suite.vectors.length} golden vectors pass`);
if (failures.length) { console.log(failures.join("\n")); process.exit(1); }
