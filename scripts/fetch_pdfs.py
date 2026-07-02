#!/usr/bin/env python3
"""Download open-access PDFs for the paper catalogue.

Pipeline per paper: build candidate PDF URLs from deterministic open-access routes
(explicit pdf_url → arXiv → Europe PMC → Unpaywall → OpenAlex → direct .pdf), fetch
each (direct first, then BrightData Web Unlocker), keep the first response whose
bytes start with %PDF. If every route misses, ask OpenRouter/Perplexity for a
direct PDF link and try that. Writes PDFs to research/papers/<key>.pdf and a
manifest.json with the outcome per paper.

Usage:
    python scripts/fetch_pdfs.py                         # all rows in research/papers.csv
    python scripts/fetch_pdfs.py --only li2022_adherence urteaga2021_genpoisson
    python scripts/fetch_pdfs.py --env ../winterchill/.env --no-perplexity
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    ROOT,
    UA,
    is_pdf,
    load_env,
    smart_fetch,
)

CATALOGUE = ROOT / "research" / "papers.csv"
OUT_DIR = ROOT / "research" / "papers"
MANIFEST = OUT_DIR / "manifest.json"

ARXIV_RE = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?")
PMC_RE = re.compile(r"(PMC\d+)", re.I)
URL_RE = re.compile(r"https?://[^\s\"'<>)]+")


# --------------------------------------------------------- open-access resolvers
def _json_get(url: str, timeout: float = 30) -> dict | None:
    try:
        req = Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except Exception:
        return None


def candidates_arxiv(row: dict) -> list[str]:
    blob = f"{row.get('url','')} {row.get('doi','')} {row.get('pdf_url','')}"
    if "arxiv.org" not in blob.lower() and "arxiv" not in row.get("doi", "").lower():
        return []
    m = ARXIV_RE.search(blob)
    return [f"https://arxiv.org/pdf/{m.group(1)}"] if m else []


def candidates_pmc(row: dict) -> list[str]:
    m = PMC_RE.search(row.get("url", ""))
    if not m:
        return []
    pmc = m.group(1).upper()
    return [
        # Europe PMC renders an open full-text PDF for OA articles:
        f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc}/fullTextPDF",
        f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc}/pdf/",
    ]


def candidates_unpaywall(row: dict, email: str) -> list[str]:
    doi = (row.get("doi") or "").strip()
    if not doi or doi.lower().startswith("10.48550"):  # arXiv DOIs handled elsewhere
        return []
    data = _json_get(f"https://api.unpaywall.org/v2/{quote(doi)}?email={quote(email)}")
    if not data:
        return []
    out = []
    best = data.get("best_oa_location") or {}
    for loc in [best, *(data.get("oa_locations") or [])]:
        for key in ("url_for_pdf", "url"):
            u = loc.get(key)
            if u:
                out.append(u)
    return out


def candidates_openalex(row: dict) -> list[str]:
    doi = (row.get("doi") or "").strip()
    data = None
    if doi and not doi.lower().startswith("10.48550"):
        data = _json_get(f"https://api.openalex.org/works/doi:{quote(doi)}")
    if not data and row.get("title"):
        srch = _json_get(
            "https://api.openalex.org/works?filter=title.search:"
            + quote(row["title"]) + "&per-page=1"
        )
        results = (srch or {}).get("results") or []
        data = results[0] if results else None
    if not data:
        return []
    out = []
    for loc in [data.get("best_oa_location") or {}, data.get("primary_location") or {}]:
        u = loc.get("pdf_url")
        if u:
            out.append(u)
    return out


def candidates_explicit(row: dict) -> list[str]:
    out = []
    if row.get("pdf_url"):
        out.append(row["pdf_url"])
    u = row.get("url", "")
    if u.lower().endswith(".pdf"):
        out.append(u)
    return out


def perplexity_find_pdf(row: dict, api_key: str) -> list[str]:
    """Ask OpenRouter/Perplexity for a direct PDF URL. Returns any URLs it names."""
    title = row.get("title", "")
    doi = row.get("doi", "")
    prompt = (
        "Find a direct, publicly accessible PDF download URL for this academic paper. "
        "Prefer the publisher OA PDF, arXiv, PubMed Central, or an institutional repository. "
        f"Title: \"{title}\". DOI: {doi or 'n/a'}. "
        "Reply with ONLY the direct PDF URL (ending in .pdf or a known PDF endpoint), nothing else."
    )
    payload = json.dumps({
        "model": "perplexity/sonar",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
    }).encode()
    try:
        req = Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": UA,
            },
            method="POST",
        )
        with urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        content = data["choices"][0]["message"]["content"]
        return URL_RE.findall(content)
    except Exception as exc:
        print(f"    perplexity error: {exc}")
        return []


# ------------------------------------------------------------------------ driver
def fetch_one(row: dict, email: str, perplexity_key: str | None) -> dict:
    key = row["key"]
    dest = OUT_DIR / f"{key}.pdf"
    if dest.exists() and dest.stat().st_size > 10_000:
        return {"key": key, "status": "cached", "path": str(dest.relative_to(ROOT)),
                "bytes": dest.stat().st_size}

    routes = [
        ("explicit", candidates_explicit(row)),
        ("arxiv", candidates_arxiv(row)),
        ("pmc", candidates_pmc(row)),
        ("unpaywall", candidates_unpaywall(row, email)),
        ("openalex", candidates_openalex(row)),
    ]
    tried = []
    for route, urls in routes:
        for url in urls:
            tried.append(url)
            got = smart_fetch(url, want_pdf=True)
            if got:
                dest.write_bytes(got[0])
                return {"key": key, "status": "ok", "route": route, "url": url,
                        "path": str(dest.relative_to(ROOT)), "bytes": len(got[0])}

    # last resort: Perplexity link discovery
    if perplexity_key:
        for url in perplexity_find_pdf(row, perplexity_key):
            if url in tried:
                continue
            got = smart_fetch(url, want_pdf=True)
            if got:
                dest.write_bytes(got[0])
                return {"key": key, "status": "ok", "route": "perplexity", "url": url,
                        "path": str(dest.relative_to(ROOT)), "bytes": len(got[0])}

    return {"key": key, "status": "failed", "tried": tried,
            "landing": row.get("url", "")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalogue", default=str(CATALOGUE))
    ap.add_argument("--env", default=None, help="path to an env file with creds")
    ap.add_argument("--only", nargs="*", help="only these catalogue keys")
    ap.add_argument("--no-perplexity", action="store_true")
    args = ap.parse_args()

    load_env(args.env)
    import os
    email = os.environ.get("UNPAYWALL_EMAIL") or "valera.yatsko@gmail.com"
    perplexity_key = None if args.no_perplexity else os.environ.get("OPENROUTER_API_KEY")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(args.catalogue, newline="")))
    if args.only:
        rows = [r for r in rows if r["key"] in set(args.only)]

    results = []
    for row in rows:
        print(f"[{row['key']}] {row.get('title','')[:70]}…")
        res = fetch_one(row, email, perplexity_key)
        icon = {"ok": "✓", "cached": "•", "failed": "✗"}.get(res["status"], "?")
        extra = res.get("route", res.get("status"))
        print(f"    {icon} {extra} {res.get('bytes','')}")
        results.append(res)
        time.sleep(0.5)

    MANIFEST.write_text(json.dumps(results, indent=2))
    ok = sum(r["status"] in ("ok", "cached") for r in results)
    print(f"\n{ok}/{len(results)} available. Manifest: {MANIFEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
