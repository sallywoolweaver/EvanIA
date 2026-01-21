#!/usr/bin/env python3
"""
scrape_50free_times.py

Enriches a roster CSV with a best-known 50 Freestyle time (SCY preferred), by searching public sources.

Important reality check:
- There is no single, universally-open database for *all* high school / club times.
- Many sources (Meet Mobile, SWIMS, some state portals) require accounts and/or prohibit scraping.
- This script is designed as a *best-effort* enrichment layer using sources that are publicly reachable.
  You may need to tune selectors for your preferred source (e.g., SwimCloud) as websites change.

Workflow:
  1) Run scrape_rosters.py to produce waterpolo_rosters.csv
  2) Run this script:
     python scrape_50free_times.py --in waterpolo_rosters.csv --out waterpolo_rosters_with_50free.csv

By default the script:
- Tries SwimCloud first (if reachable) via a search query and then the swimmer page.
- Optionally uses a lightweight web search (DuckDuckGo HTML endpoint) as a fallback to locate a swimmer page.
- Writes a separate errors CSV.

"""

from __future__ import annotations

import argparse
import csv
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup


USER_AGENT = "waterpolo-roster-enricher/1.0 (contact: you@example.com)"
TIMEOUT = 20
RATE_LIMIT_S = 1.0


@dataclass
class TimeHit:
    best_time: str
    course: str  # SCY / LCM / SCM / unknown
    source_url: str
    confidence: float  # 0..1


def _req(url: str, *, params: Optional[Dict[str, str]] = None) -> str:
    r = requests.get(
        url,
        params=params,
        timeout=TIMEOUT,
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
    )
    r.raise_for_status()
    return r.text


def swimcloud_search(name: str) -> List[str]:
    """Return candidate SwimCloud swimmer URLs for a name, if SwimCloud is reachable."""
    # SwimCloud structures can change; this is a pragmatic starting point.
    q = "+".join(name.split())
    url = "https://www.swimcloud.com/search/"
    html = _req(url, params={"q": q})
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for a in soup.select('a[href*="/swimmer/"]'):
        href = a.get("href") or ""
        if "/swimmer/" in href:
            if href.startswith("http"):
                out.append(href)
            else:
                out.append("https://www.swimcloud.com" + href)
    # Deduplicate while preserving order
    seen = set()
    dedup = []
    for u in out:
        if u not in seen:
            seen.add(u)
            dedup.append(u)
    return dedup[:10]


def parse_swimcloud_50free(swimmer_url: str) -> Optional[TimeHit]:
    """Parse 50 Freestyle best time from a SwimCloud swimmer page."""
    html = _req(swimmer_url)
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n", strip=True)

    # Heuristic: look for '50 Free' row with a time nearby.
    # Times are usually like 20.15 or 1:43.21 etc.
    # We'll pick the first plausible sprint time after '50 Freestyle' / '50 Free'.
    patterns = [
        r"\b50\s*(Freestyle|Free)\b[\s\S]{0,120}?(\d{2}\.\d{2})\b",
        r"\b50\s*(Freestyle|Free)\b[\s\S]{0,120}?(\d{1}\.\d{2})\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            t = m.group(2)
            return TimeHit(best_time=t, course="SCY", source_url=swimmer_url, confidence=0.75)

    return None


def duckduckgo_find_candidate(name: str, hometown: str = "") -> List[str]:
    """Lightweight web search to locate a swimmer page when direct search fails."""
    query = f"{name} 50 free time {hometown} swim"
    url = "https://duckduckgo.com/html/"
    html = _req(url, params={"q": query})
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for a in soup.select("a.result__a"):
        href = a.get("href") or ""
        if "swimcloud.com" in href and "/swimmer/" in href:
            out.append(href)
    # add any other likely results pages that contain meet results
    for a in soup.select("a.result__a"):
        href = a.get("href") or ""
        if any(dom in href for dom in ["athletic.net", "teamunify.com", "meetresults", "swimswam"]):
            out.append(href)
    seen=set()
    dedup=[]
    for u in out:
        if u not in seen:
            seen.add(u)
            dedup.append(u)
    return dedup[:10]


def best_50free_for_person(name: str, hometown: str) -> Tuple[Optional[TimeHit], Optional[str]]:
    """Return best known 50 free TimeHit and optional error."""
    try:
        # 1) SwimCloud direct search
        candidates = swimcloud_search(name)
        for u in candidates:
            hit = parse_swimcloud_50free(u)
            if hit:
                return hit, None
            time.sleep(RATE_LIMIT_S)

        # 2) DuckDuckGo fallback to locate a swimmer page
        candidates = duckduckgo_find_candidate(name, hometown)
        for u in candidates:
            if "swimcloud.com" in u and "/swimmer/" in u:
                hit = parse_swimcloud_50free(u)
                if hit:
                    return hit, None
            time.sleep(RATE_LIMIT_S)

        return None, "No 50 free time found from public sources."

    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", required=True, help="Roster CSV from scrape_rosters.py")
    ap.add_argument("--out", dest="out_csv", required=True, help="Output enriched CSV")
    ap.add_argument("--errors", default="waterpolo_50free_errors.csv", help="Errors CSV")
    ap.add_argument("--limit", type=int, default=0, help="Process only N rows (debug)")
    args = ap.parse_args()

    in_path = Path(args.in_csv)
    out_path = Path(args.out_csv)
    err_path = Path(args.errors)

    rows: List[Dict[str, str]] = []
    with in_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            rows.append(row)
            if args.limit and i >= args.limit:
                break

    errors: List[Dict[str, str]] = []
    for i, r in enumerate(rows, start=1):
        name = (r.get("name") or "").strip()
        hometown = (r.get("hometown") or "").strip()
        if not name:
            continue

        hit, err = best_50free_for_person(name, hometown)
        if hit:
            r["best_50free"] = hit.best_time
            r["best_50free_course"] = hit.course
            r["best_50free_source"] = hit.source_url
            r["best_50free_confidence"] = f"{hit.confidence:.2f}"
        else:
            r["best_50free"] = ""
            r["best_50free_course"] = ""
            r["best_50free_source"] = ""
            r["best_50free_confidence"] = ""
            errors.append({"row": str(i), "name": name, "hometown": hometown, "error": err or "unknown"})

        time.sleep(RATE_LIMIT_S)

    # Write enriched
    fieldnames = list(rows[0].keys()) if rows else []
    # ensure new fields present
    for fn in ["best_50free","best_50free_course","best_50free_source","best_50free_confidence"]:
        if fn not in fieldnames:
            fieldnames.append(fn)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with err_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["row","name","hometown","error"])
        w.writeheader()
        for e in errors:
            w.writerow(e)

    print(f"[DONE] Wrote {len(rows)} rows to {out_path} and {len(errors)} errors to {err_path}")


if __name__ == "__main__":
    main()
