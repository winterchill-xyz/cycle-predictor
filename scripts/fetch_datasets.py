#!/usr/bin/env python3
"""Download openly-available menstrual-cycle datasets into data/raw/.

Registry-driven. Each dataset either has an automated resolver (FedCycle) or prints
manual access steps (mcPHASES needs PhysioNet credentialing + a DUA). Fetches go
direct first, then via BrightData Web Unlocker (for bot-blocked landing pages).

Usage:
    python scripts/fetch_datasets.py --list
    python scripts/fetch_datasets.py --only fedcycle
    python scripts/fetch_datasets.py --env ../winterchill/.env
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, load_env, smart_fetch  # noqa: E402

RAW = ROOT / "data" / "raw"
HREF_RE = re.compile(r'href=["\']([^"\']+)["\']', re.I)

# Header tokens that identify the real FedCycle CSV (guards against saving an HTML error page).
FEDCYCLE_MARKERS = ("CYCLENUMBER", "MeanCycleLength", "LengthofCycle", "SUBCODE", "ClientID")


def _abs(base: str, href: str) -> str:
    if href.startswith("http"):
        return href
    if href.startswith("/"):
        return "https://epublications.marquette.edu" + href
    return base.rstrip("/") + "/" + href


def fetch_fedcycle() -> dict:
    """Resolve the FedCycle CSV from the Marquette e-Publications landing page."""
    out_dir = RAW / "fedcycle"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "FedCycleData071012.csv"
    if dest.exists() and dest.stat().st_size > 50_000:
        return {"dataset": "fedcycle", "status": "cached",
                "path": str(dest.relative_to(ROOT)), "bytes": dest.stat().st_size}

    landing = "https://epublications.marquette.edu/data_nfp/7/"
    got = smart_fetch(landing)
    candidates: list[str] = []
    if got:
        html = got[0].decode("utf-8", "replace")
        for href in HREF_RE.findall(html):
            low = href.lower()
            if "viewcontent.cgi" in low and ("csv" in low or "additional" in low):
                candidates.append(_abs(landing, href))
            elif low.endswith(".csv"):
                candidates.append(_abs(landing, href))
    # De-dup, keep order; add a couple of known fallbacks last.
    seen, ordered = set(), []
    for u in candidates + [
        "https://epublications.marquette.edu/cgi/viewcontent.cgi?filename=0&article=1006&context=data_nfp&type=additional",
    ]:
        if u not in seen:
            seen.add(u); ordered.append(u)

    for url in ordered:
        blob = smart_fetch(url)
        if not blob:
            continue
        text = blob[0].decode("utf-8", "replace")
        if any(marker in text for marker in FEDCYCLE_MARKERS):
            dest.write_bytes(blob[0])
            return {"dataset": "fedcycle", "status": "ok", "url": url,
                    "path": str(dest.relative_to(ROOT)), "bytes": len(blob[0])}

    return {"dataset": "fedcycle", "status": "failed", "tried": ordered,
            "landing": landing,
            "hint": "Open the landing page and grab the .csv under 'Additional Files' manually."}


def fetch_mcphases() -> dict:
    """mcPHASES needs PhysioNet credentialing + DUA; automate only if creds present."""
    base = "https://physionet.org/files/mcphases/1.0.0/"
    user = os.environ.get("PHYSIONET_USER")
    pw = os.environ.get("PHYSIONET_PASS")
    out_dir = RAW / "mcphases"
    if not (user and pw):
        return {
            "dataset": "mcphases", "status": "manual",
            "steps": [
                "1. Create a free account at https://physionet.org/",
                "2. Get credentialed (training + ID verification) and sign the mcPHASES DUA",
                "3. Download from https://physionet.org/content/mcphases/ (or the files/ tree)",
                "4. Or set PHYSIONET_USER / PHYSIONET_PASS and re-run to pull via wget",
            ],
            "files_base": base,
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    # PhysioNet supports authenticated recursive download via wget; shell out to it.
    import subprocess
    cmd = ["wget", "-r", "-N", "-c", "-np", "-nH", "--cut-dirs=3",
           f"--user={user}", f"--password={pw}", "-P", str(out_dir), base]
    rc = subprocess.call(cmd)
    return {"dataset": "mcphases", "status": "ok" if rc == 0 else "failed",
            "path": str(out_dir.relative_to(ROOT))}


REGISTRY = {
    "fedcycle": ("Marquette FedCycle CSV (~290 KB, open; licensing unverified)", fetch_fedcycle),
    "mcphases": ("PhysioNet mcPHASES (multimodal; needs credentialing + DUA)", fetch_mcphases),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", help="dataset keys to fetch (default: all)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--env", default=None)
    args = ap.parse_args()

    if args.list:
        for k, (desc, _) in REGISTRY.items():
            print(f"  {k:12s} {desc}")
        return 0

    load_env(args.env)
    keys = args.only or list(REGISTRY)
    results = []
    for k in keys:
        if k not in REGISTRY:
            print(f"unknown dataset: {k} (see --list)"); continue
        print(f"[{k}] {REGISTRY[k][0]}")
        res = REGISTRY[k][1]()
        print(f"    -> {res['status']} {res.get('bytes', res.get('path',''))}")
        results.append(res)

    manifest = RAW / "download_manifest.json"
    manifest.write_text(json.dumps(results, indent=2))
    print(f"\nManifest: {manifest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
