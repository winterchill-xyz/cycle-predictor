#!/usr/bin/env python3
"""Generate artifacts/test_vectors.json — the cross-language conformance suite.

Each vector is a kernel-level (day-of-cycle) input and the expected {mode,
cycle_length, sd} from the reference kernel. A port passes if it reproduces every
vector within a small tolerance (see PORTING.md). Signals are keyed by day-of-cycle
(strings, as JSON requires) so the vectors are date-library-agnostic.

    .venv/bin/python scripts/gen_test_vectors.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cycle_predictor import portable  # noqa: E402

OUT = ROOT / "artifacts" / "test_vectors.json"

# (name, history, lh_by_day, temp_by_day)
_THERMAL = {**{d: 36.40 for d in range(0, 14)}, **{d: 36.75 for d in range(14, 24)}}
SCENARIOS = [
    ("cold_start_no_history", [], None, None),
    ("single_cycle", [28], None, None),
    ("regular_history", [28, 29, 28, 30, 29], None, None),
    ("short_cycles", [25, 24, 26], None, None),
    ("long_cycles", [34, 33, 35], None, None),
    ("history_with_skip_artifact", [28, 29, 56, 27, 30], None, None),
    ("lh_surge_day15", [28, 29], {15: 42.0}, None),
    ("lh_surge_day12", [30], {12: 30.0, 13: 55.0}, None),
    ("lh_no_surge_falls_back", [29, 30], {14: 8.0, 15: 12.0}, None),
    ("wearable_thermal_shift", [29, 30], None, _THERMAL),
    ("lh_beats_wearable", [28], {15: 40.0}, _THERMAL),
    ("flat_temp_falls_back", [29, 30], None, {d: 36.5 for d in range(0, 25)}),
]


def main() -> int:
    vectors = []
    for name, history, lh, temp in SCENARIOS:
        out = portable.predict(portable.DEFAULT_PARAMS, history, lh_by_day=lh, temp_by_day=temp)
        vectors.append({
            "name": name,
            "input": {
                "history": history,
                "lh_by_day": {str(k): v for k, v in lh.items()} if lh else None,
                "temp_by_day": {str(k): v for k, v in temp.items()} if temp else None,
            },
            "expected": {
                "mode": out["mode"],
                "cycle_length": round(out["cycle_length"], 6),
                "sd": round(out["sd"], 6),
            },
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"model_version": portable.DEFAULT_PARAMS["version"],
         "tolerance": 1e-4, "vectors": vectors}, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} — {len(vectors)} vectors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
