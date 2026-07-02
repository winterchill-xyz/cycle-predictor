# research/papers/

Downloaded sources for the papers catalogued in [`../papers.csv`](../papers.csv).
The **binaries are gitignored** (large + license-restricted); only the small
JSON manifests and this README are tracked. Reproduce the downloads with the
scripts below.

## What gets downloaded here

| Format | File | For | Coverage |
|--------|------|-----|----------|
| PDF | `<key>.pdf` | reading | **14/14** papers |
| LaTeX source | `<key>.tar.gz` | **exact math / equations** (best for parsing) | 4/4 arXiv papers |
| HTML | `<key>.html` | rendered MathML fallback | 4/4 arXiv papers |

Non-arXiv papers (PMC / Nature / JMIR / OUP) are PDF-only here — their publishers
don't expose LaTeX source.

## Reproduce

```bash
# PDFs (open-access routes → BrightData Web Unlocker → Perplexity fallback)
python scripts/fetch_pdfs.py --env ../winterchill/.env

# LaTeX source (default) + HTML for the arXiv papers
python scripts/fetch_arxiv_html.py --env ../winterchill/.env
```

## Manifests (tracked)

- `manifest.json` — per-paper PDF outcome + which route resolved it
  (`explicit` / `arxiv` / `pmc` / `unpaywall` / `openalex` / `perplexity`).
- `html_manifest.json` — per-arXiv-paper HTML + LaTeX-source outcome.

## Notes

- PDFs are for **private research reference**. Redistribution rights vary by
  publisher — do not re-host. Keeping them gitignored is deliberate.
- The two hardest PDFs (`depaulaoliveira2021`, `bortot2006`) were resolved by the
  **Perplexity link-discovery fallback**, not deterministic OA routes — expect that
  path to be needed for older/paywalled work.
