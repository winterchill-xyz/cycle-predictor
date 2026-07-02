"""Shared helpers for the downloader scripts (stdlib-only, no pip install needed).

Env loading, BrightData Web Unlocker fetch (proxy + API mode), PDF detection.
Credentials are read from an env file (default: ./.env then ../winterchill/.env).
"""
from __future__ import annotations

import json
import os
import ssl
from pathlib import Path
from urllib.request import (
    Request,
    ProxyHandler,
    HTTPSHandler,
    build_opener,
    urlopen,
)

ROOT = Path(__file__).resolve().parents[1]
UA = "cycle-predictor-fetcher/1.0 (research; contact valera.yatsko@gmail.com)"


# --------------------------------------------------------------------------- env
def load_env(explicit: str | None = None) -> dict:
    """Merge KEY=VALUE pairs from the first existing env file into os.environ
    (without overriding already-set vars). Returns the merged mapping.

    Search order: explicit path → ./.env → ../winterchill/.env (creds live there).
    """
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates += [ROOT / ".env", ROOT.parent / "winterchill" / ".env"]

    loaded: dict[str, str] = {}
    for path in candidates:
        if not path.is_file():
            continue
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in loaded:
                loaded[key] = val
        # first existing file wins for a given key; keep scanning for missing keys
    for k, v in loaded.items():
        os.environ.setdefault(k, v)
    return {k: os.environ.get(k, "") for k in loaded}


# --------------------------------------------------------------------------- pdf
def is_pdf(data: bytes, content_type: str = "") -> bool:
    return data[:5] == b"%PDF-" or "application/pdf" in (content_type or "").lower()


# -------------------------------------------------------------------- fetch (raw)
def _opener(proxy: str | None):
    handlers = []
    if proxy:
        handlers.append(ProxyHandler({"http": proxy, "https": proxy}))
        # BrightData Web Unlocker terminates TLS with its own cert → don't verify.
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        handlers.append(HTTPSHandler(context=ctx))
    return build_opener(*handlers)


def fetch_direct(url: str, timeout: float = 60, accept: str = "*/*") -> tuple[bytes, str] | None:
    """Plain fetch, no proxy. Returns (bytes, content_type) or None."""
    try:
        req = Request(url, headers={"User-Agent": UA, "Accept": accept})
        with urlopen(req, timeout=timeout) as resp:
            return resp.read(), resp.headers.get("Content-Type", "")
    except Exception:
        return None


def brightdata_proxy_url() -> str | None:
    u = os.environ.get("BRIGHTDATA_WU_USERNAME")
    p = os.environ.get("BRIGHTDATA_WU_PASSWORD")
    h = os.environ.get("BRIGHTDATA_WU_HOST")
    port = os.environ.get("BRIGHTDATA_WU_PORT")
    if all([u, p, h, port]):
        return f"http://{u}:{p}@{h}:{port}"
    return None


def fetch_via_brightdata_proxy(url: str, timeout: float = 120, accept: str = "*/*") -> tuple[bytes, str] | None:
    """Fetch through the BrightData Web Unlocker proxy. Requires the runner's
    egress IP to be whitelisted in the zone, and to run OUTSIDE the tool sandbox."""
    proxy = brightdata_proxy_url()
    if not proxy:
        return None
    try:
        req = Request(url, headers={"User-Agent": UA, "Accept": accept})
        with _opener(proxy).open(req, timeout=timeout) as resp:
            return resp.read(), resp.headers.get("Content-Type", "")
    except Exception:
        return None


def fetch_via_brightdata_api(url: str, timeout: float = 120) -> tuple[bytes, str] | None:
    """Fetch through the BrightData Web Unlocker API (token auth, no IP whitelist)."""
    token = os.environ.get("BRIGHTDATA_API_KEY")
    zone = os.environ.get("BRIGHTDATA_WU_ZONE", "web_unlocker1")
    if not token:
        return None
    payload = json.dumps({"zone": zone, "url": url, "format": "raw"}).encode()
    try:
        req = Request(
            "https://api.brightdata.com/request",
            data=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "User-Agent": UA,
            },
            method="POST",
        )
        with urlopen(req, timeout=timeout) as resp:
            return resp.read(), resp.headers.get("Content-Type", "")
    except Exception:
        return None


def smart_fetch(url: str, want_pdf: bool = False, timeout: float = 120) -> tuple[bytes, str] | None:
    """Try direct first (free/fast); fall back to BrightData proxy then API.
    If want_pdf, only accept responses that look like a PDF."""
    for fetcher in (fetch_direct, fetch_via_brightdata_proxy, fetch_via_brightdata_api):
        got = fetcher(url, timeout=timeout)
        if not got:
            continue
        data, ctype = got
        if not data:
            continue
        if want_pdf and not is_pdf(data, ctype):
            continue
        return data, ctype
    return None
