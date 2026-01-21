#!/usr/bin/env python3
"""scrape_swimcloud_50free.py

Augment an existing roster CSV with best 50 Freestyle times (high-school/club history)
from publicly accessible SwimCloud pages.

Important limitations:
- SwimCloud may throttle or block automated requests; use --sleep and --max to test gradually.
- Athlete identity matching is heuristic (name + optional hometown/state). Review low-confidence matches.

Usage:
  python scrape_swimcloud_50free.py --in waterpolo_rosters.csv --out waterpolo_rosters_with_50free.csv
  python scrape_swimcloud_50free.py --in waterpolo_rosters.csv --out out.csv --sleep 2.0 --max 200

The script adds/updates:
  best_50free, best_50free_source, best_50free_confidence
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote_plus, urljoin

import requests
from bs4 import BeautifulSoup

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

TIME_RE = re.compile(r"\b(\d{1,2}:\d{2}\.\d{2}|\d{2}\.\d{2}|\d{1}\.\d{2})\b")
# Swim: 50 Free times are commonly ~20-30s SCY for strong athletes.
def _time_to_seconds(t: str) -> Optional[float]:
    t = t.strip()
    if not t:
        return None
    if ":" in t:
        m, s = t.split(":", 1)
        try:
            return float(m) * 60.0 + float(s)
        except Exception:
            return None
    try:
        return float(t)
    except Exception:
        return None

def fetch(url: str, *, timeout: int = 30) -> str:
    r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
    r.raise_for_status()
    return r.text

def cache_get(cache_dir: Path, url: str) -> Optional[str]:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()
    fp = cache_dir / f"{h}.html"
    if fp.exists():
        return fp.read_text(encoding="utf-8", errors="ignore")
    return None

def cache_set(cache_dir: Path, url: str, html: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()
    fp = cache_dir / f"{h}.html"
    fp.write_text(html, encoding="utf-8")

@dataclass
class Candidate:
    url: str
    label: str
    score: float

def parse_swimcloud_search(html: str) -> List[Candidate]:
    soup = BeautifulSoup(html, "html.parser")
    cands: List[Candidate] = []
    # Heuristic: look for swimmer links
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/swimmer/" in href:
            label = a.get_text(" ", strip=True)
            if not label:
                continue
            url = urljoin("https://www.swimcloud.com/", href)
            cands.append(Candidate(url=url, label=label, score=0.0))
    # Deduplicate by URL
    seen = set()
    uniq: List[Candidate] = []
    for c in cands:
        if c.url in seen:
            continue
        seen.add(c.url)
        uniq.append(c)
    return uniq[:25]

def score_candidate(label: str, *, name: str, hometown: str) -> float:
    """Very simple scoring: exact-ish name match + hometown token overlap."""
    s = 0.0
    ln = label.lower()
    n = name.lower().strip()
    if n and n in ln:
        s += 2.0
    # match last name strongly
    parts = [p for p in re.split(r"\s+", n) if p]
    if parts:
        last = parts[-1]
        if last and last in ln:
            s += 1.5
    ht = (hometown or "").lower()
    # take city/state tokens
    tokens = [t.strip() for t in re.split(r"[,/\-]", ht) if t.strip()]
    for tok in tokens[:2]:
        if tok and tok in ln:
            s += 0.5
    return s

def extract_best_50free(html: str) -> Optional[str]:
    """Extract best 50 Free time (prefers SCY if course appears)."""
    soup = BeautifulSoup(html, "html.parser")
    # 1) Try to find rows containing "50 Free"
    best: Tuple[Optional[float], Optional[str]] = (None, None)

    # Search all table rows
    for tr in soup.find_all("tr"):
        txt = tr.get_text(" ", strip=True)
        if not txt:
            continue
        if re.search(r"\b50\s*Free\b", txt, flags=re.IGNORECASE):
            # Prefer SCY rows if present
            is_scy = bool(re.search(r"\bSCY\b", txt))
            # pull all time-like tokens
            times = TIME_RE.findall(txt)
            for t in times:
                sec = _time_to_seconds(t)
                if sec is None:
                    continue
                # discard implausible times
                if sec < 15.0 or sec > 60.0:
                    continue
                bonus = -0.2 if is_scy else 0.0
                cand = sec + bonus
                if best[0] is None or cand < best[0]:
                    best = (cand, t)

    # 2) Fallback: regex scan near 50 Free
    if best[1] is None:
        raw = soup.get_text("\n")
        # windowed search: lines near "50 Free"
        lines = raw.splitlines()
        for i, line in enumerate(lines):
            if re.search(r"\b50\s*Free\b", line, flags=re.IGNORECASE):
                window = " ".join(lines[i:i+5])
                times = TIME_RE.findall(window)
                for t in times:
                    sec = _time_to_seconds(t)
                    if sec is None or sec < 15.0 or sec > 60.0:
                        continue
                    if best[0] is None or sec < best[0]:
                        best = (sec, t)

    return best[1]

def lookup_swimcloud_50free(name: str, hometown: str, cache_dir: Path, sleep_s: float) -> Tuple[Optional[str], Optional[str], float]:
    """Returns (time, source_url, confidence)."""
    if not name or len(name.strip()) < 3:
        return None, None, 0.0

    q = quote_plus(name.strip())
    search_url = f"https://www.swimcloud.com/search/?q={q}"

    html = cache_get(cache_dir, search_url)
    if html is None:
        html = fetch(search_url)
        cache_set(cache_dir, search_url, html)
        time.sleep(sleep_s)

    cands = parse_swimcloud_search(html)
    if not cands:
        return None, None, 0.0

    for c in cands:
        c.score = score_candidate(c.label, name=name, hometown=hometown)

    cands.sort(key=lambda c: c.score, reverse=True)
    top = cands[:5]

    # Try candidates in order until we find a plausible 50 free
    for idx, c in enumerate(top):
        page = cache_get(cache_dir, c.url)
        if page is None:
            try:
                page = fetch(c.url)
                cache_set(cache_dir, c.url, page)
                time.sleep(sleep_s)
            except Exception:
                continue
        t = extract_best_50free(page)
        if t:
            # Confidence: score + small bonus for being 1st/2nd result
            conf = min(1.0, 0.2 + 0.15 * c.score + (0.1 if idx == 0 else 0.0))
            return t, c.url, conf

    return None, None, 0.0

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input roster CSV")
    ap.add_argument("--out", dest="out", required=True, help="Output CSV")
    ap.add_argument("--sleep", type=float, default=1.5, help="Sleep between HTTP requests")
    ap.add_argument("--max", type=int, default=10_000, help="Max rows to process (for testing)")
    ap.add_argument("--cache", default=".cache_swimcloud", help="Cache directory (HTML)")
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    cache_dir = Path(args.cache)

    with inp.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("No rows found in input CSV")

    fieldnames = list(rows[0].keys())
    for col in ["best_50free", "best_50free_source", "best_50free_confidence"]:
        if col not in fieldnames:
            fieldnames.append(col)

    processed = 0
    for r in rows:
        if processed >= args.max:
            break
        name = (r.get("player_name") or "").strip()
        hometown = (r.get("hometown") or "").strip()
        existing = (r.get("best_50free") or "").strip()
        if existing:
            continue
        t, src, conf = lookup_swimcloud_50free(name, hometown, cache_dir=cache_dir, sleep_s=args.sleep)
        if t:
            r["best_50free"] = t
            r["best_50free_source"] = src or ""
            r["best_50free_confidence"] = f"{conf:.2f}"
        processed += 1

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[DONE] Wrote {len(rows)} rows -> {out}")

if __name__ == "__main__":
    main()
