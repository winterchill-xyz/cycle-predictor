#!/usr/bin/env python3
"""Fetch arXiv papers as HTML (and optionally LaTeX source) for math-friendly parsing.

PDFs store math as positioned glyphs, which is hard to parse. The LaTeX source has
the *exact* equations, so we fetch it by default; HTML (MathML) is a rendered
fallback for reading. For each arXiv row in the catalogue this fetches:
    - LaTeX source tarball  https://arxiv.org/e-print/<id>   (DEFAULT — exact math)
    - native arXiv HTML     https://arxiv.org/html/<id>      (LaTeX submissions since ~Dec 2023)
    - ar5iv HTML fallback    https://ar5iv.org/abs/<id>       (covers older papers)

Saves research/papers/<key>.tar.gz (+ <key>.html). These are gitignored like PDFs.

    python scripts/fetch_arxiv_html.py --env ../winterchill/.env
    python scripts/fetch_arxiv_html.py --no-source        # HTML only, skip LaTeX
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, load_env, smart_fetch  # noqa: E402

CATALOGUE = ROOT / "research" / "papers.csv"
OUT_DIR = ROOT / "research" / "papers"
MANIFEST = OUT_DIR / "html_manifest.json"
ARXIV_RE = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?")

# Markers that the response is a real rendered article (not a 404 stub / landing page).
HTML_MARKERS = ("<math", "mathml", "ltx_Math", "ltx_equation", "MathJax", "ltx_para")


def arxiv_id(row: dict) -> str | None:
    blob = f"{row.get('url','')} {row.get('doi','')} {row.get('pdf_url','')}"
    if "arxiv" not in blob.lower():
        return None
    m = ARXIV_RE.search(blob)
    return m.group(1) if m else None


def fetch_html(arxiv: str) -> tuple[str, bytes] | None:
    for source, url in (
        ("arxiv-native", f"https://arxiv.org/html/{arxiv}"),
        ("ar5iv", f"https://ar5iv.org/abs/{arxiv}"),
        ("ar5iv-labs", f"https://ar5iv.labs.arxiv.org/html/{arxiv}"),
    ):
        got = smart_fetch(url)
        if not got or len(got[0]) < 20_000:
            continue
        text = got[0].decode("utf-8", "replace")
        if "<html" in text.lower() and any(m in text for m in HTML_MARKERS):
            return source, got[0]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalogue", default=str(CATALOGUE))
    ap.add_argument("--env", default=None)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--no-source", action="store_true",
                    help="skip the LaTeX source tarball (fetch HTML only)")
    args = ap.parse_args()
    fetch_source = not args.no_source

    load_env(args.env)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(args.catalogue, newline="")))
    if args.only:
        rows = [r for r in rows if r["key"] in set(args.only)]

    results = []
    for row in rows:
        arxiv = arxiv_id(row)
        if not arxiv:
            continue
        key = row["key"]
        print(f"[{key}] arXiv:{arxiv}")
        rec = {"key": key, "arxiv_id": arxiv}

        dest = OUT_DIR / f"{key}.html"
        if dest.exists() and dest.stat().st_size > 20_000:
            rec["html"] = {"status": "cached", "bytes": dest.stat().st_size}
        else:
            got = fetch_html(arxiv)
            if got:
                dest.write_bytes(got[1])
                rec["html"] = {"status": "ok", "source": got[0], "bytes": len(got[1])}
                print(f"    ✓ html via {got[0]} ({len(got[1])} bytes)")
            else:
                rec["html"] = {"status": "failed"}
                print("    ✗ no HTML available (older/scanned paper — use PDF)")

        if fetch_source:
            tar = OUT_DIR / f"{key}.tar.gz"
            if tar.exists() and tar.stat().st_size > 1000:
                rec["source"] = {"status": "cached", "bytes": tar.stat().st_size}
            else:
                got = smart_fetch(f"https://arxiv.org/e-print/{arxiv}")
                if got and len(got[0]) > 1000:
                    tar.write_bytes(got[0])
                    rec["source"] = {"status": "ok", "bytes": len(got[0])}
                    print(f"    ✓ LaTeX source ({len(got[0])} bytes)")
                else:
                    rec["source"] = {"status": "failed"}
        results.append(rec)

    MANIFEST.write_text(json.dumps(results, indent=2))
    ok = sum(r.get("html", {}).get("status") in ("ok", "cached") for r in results)
    print(f"\n{ok}/{len(results)} arXiv papers available as HTML. Manifest: {MANIFEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
