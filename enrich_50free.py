#!/usr/bin/env python3
"""enrich_50free.py

Enriches a water polo roster CSV with 50 Freestyle swim times.

Reality check:
- There is no universally-public, scrape-friendly database of swim times for water polo rosters.
- Most “reliable” sources (e.g., SWIMS/USA Swimming) are not openly indexable.

Therefore this script supports two practical workflows:

A) Merge-in from a separate times CSV you provide (recommended)
   You obtain times from whatever source you legally have access to (SwimCloud export,
   meet results, state databases, etc.), then merge by (name + hometown/state).

B) Optional: parse from a per-athlete profile URL column (best-effort)
   If your roster CSV includes a column named `swim_profile_url`, the script will fetch
   that URL and attempt to extract a 50 Free best time. This is intentionally conservative
   and may fail depending on the site.

Usage:
  python enrich_50free.py \
    --rosters waterpolo_rosters.csv \
    --out waterpolo_rosters_with_50free.csv \
    --times-csv times_50free.csv

  python enrich_50free.py \
    --rosters waterpolo_rosters.csv \
    --out waterpolo_rosters_with_50free.csv \
    --use-profile-urls

Expected roster columns (from scrape_rosters.py):
  school, division, roster_url, jersey, name, position, year, height, hometown

Expected times CSV columns (case-insensitive; we match the first that exists):
  - name (required)
  - hometown OR state (optional but strongly recommended)
  - fifty_free OR 50_free OR time_50_free OR "50 Free" (required)

Output adds:
  time_50_free, time_50_free_source, time_50_free_confidence
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

TIMEOUT = 25


def norm_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z\s\-']+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def extract_state_from_hometown(h: str) -> str:
    # Common patterns: "Houston, Texas" or "Newport Beach, Calif." or "Budapest, Hungary"
    parts = [p.strip() for p in (h or "").split(",") if p.strip()]
    if len(parts) >= 2:
        return parts[-1].lower()
    return ""


def _request_get(url: str) -> str:
    r = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.text


def parse_50free_from_profile_html(html: str) -> Optional[Tuple[str, str]]:
    """Best-effort parser. Returns (time_str, source_hint) or None."""
    soup = BeautifulSoup(html, "lxml")

    # Try to find any row mentioning 50 Free / 50 Freestyle.
    text = soup.get_text("\n", strip=True)
    if not text:
        return None

    # Focus on lines containing '50' and 'Free'
    lines = [ln for ln in text.splitlines() if "50" in ln and ("free" in ln.lower() or "freestyle" in ln.lower())]
    if not lines:
        return None

    # Extract candidate times like 19.88, 20.04, 21.5 (2 decimals typical)
    times = []
    for ln in lines:
        for m in re.finditer(r"\b(\d{2}\.\d{1,2})\b", ln):
            try:
                t = float(m.group(1))
            except ValueError:
                continue
            # Filter obviously wrong
            if 15.0 <= t <= 40.0:
                times.append((t, m.group(1), ln))

    if not times:
        return None

    # Prefer SCY if hinted
    def score(item):
        t, s, ln = item
        ln_l = ln.lower()
        scy = 0 if "scy" in ln_l else 1
        scm = 0 if "scm" in ln_l else 1
        lcm = 0 if "lcm" in ln_l else 1
        # Lower is better: prefer scy over scm over lcm, then faster time
        return (scy, scm, lcm, t)

    best = sorted(times, key=score)[0]
    return best[1], "profile_parse"


@dataclass
class TimeRow:
    time_str: str
    source: str
    confidence: float


def load_times_csv(path: Path) -> Tuple[Dict[Tuple[str, str], TimeRow], Dict[str, TimeRow]]:
    """Returns:
    - keyed by (norm_name, norm_state)
    - keyed by norm_name only (fallback)
    """
    by_name_state: Dict[Tuple[str, str], TimeRow] = {}
    by_name: Dict[str, TimeRow] = {}

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = [h.lower().strip() for h in (reader.fieldnames or [])]
        if not headers:
            raise ValueError("times CSV has no headers")

        def pick_col(cands):
            for c in cands:
                if c.lower() in headers:
                    return c.lower()
            return None

        name_col = pick_col(["name"])
        if not name_col:
            raise ValueError("times CSV must include a 'name' column")

        hometown_col = pick_col(["hometown"])
        state_col = pick_col(["state"])
        time_col = pick_col(["fifty_free", "50_free", "time_50_free", "50 free", "50 freestyle"])
        if not time_col:
            raise ValueError("times CSV must include a 50 free time column (e.g., '50_free' or 'fifty_free')")

        for row in reader:
            name = norm_name(row.get(name_col, ""))
            if not name:
                continue

            t = (row.get(time_col, "") or "").strip()
            if not t:
                continue

            state = ""
            if state_col:
                state = (row.get(state_col, "") or "").strip().lower()
            elif hometown_col:
                state = extract_state_from_hometown(row.get(hometown_col, "") or "")

            tr = TimeRow(time_str=t, source=str(path), confidence=0.9 if state else 0.6)
            if state:
                by_name_state[(name, state)] = tr
            # keep last occurrence; callers can curate the input file
            by_name[name] = tr

    return by_name_state, by_name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rosters", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--times-csv", type=Path, default=None, help="Optional: merge-in times from this CSV")
    ap.add_argument("--use-profile-urls", action="store_true", help="Optional: parse 50 free from 'swim_profile_url' column")
    ap.add_argument("--sleep", type=float, default=0.6, help="Sleep between profile URL requests")
    args = ap.parse_args()

    by_name_state: Dict[Tuple[str, str], TimeRow] = {}
    by_name: Dict[str, TimeRow] = {}
    if args.times_csv:
        by_name_state, by_name = load_times_csv(args.times_csv)

    with args.rosters.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError("rosters CSV has no headers")

        # Ensure output columns exist
        out_fields = fieldnames[:]
        for col in ["time_50_free", "time_50_free_source", "time_50_free_confidence"]:
            if col not in out_fields:
                out_fields.append(col)

        rows = list(reader)

    for r in rows:
        name = norm_name(r.get("name", ""))
        state = extract_state_from_hometown(r.get("hometown", ""))

        best: Optional[TimeRow] = None

        # 1) Prefer exact name+state match from times CSV
        if by_name_state and state:
            best = by_name_state.get((name, state))

        # 2) Fallback to name-only match from times CSV
        if best is None and by_name:
            best = by_name.get(name)

        # 3) Optional: profile URL scrape
        if best is None and args.use_profile_urls:
            url = (r.get("swim_profile_url") or "").strip()
            if url:
                try:
                    html = _request_get(url)
                    parsed = parse_50free_from_profile_html(html)
                    if parsed:
                        t, hint = parsed
                        best = TimeRow(time_str=t, source=url, confidence=0.4)
                except Exception:
                    # silent; we keep output blank
                    best = None
                time.sleep(max(0.0, args.sleep))

        if best is not None:
            r["time_50_free"] = best.time_str
            r["time_50_free_source"] = best.source
            r["time_50_free_confidence"] = f"{best.confidence:.2f}"
        else:
            r.setdefault("time_50_free", "")
            r.setdefault("time_50_free_source", "")
            r.setdefault("time_50_free_confidence", "")

    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        w.writerows(rows)

    print(f"[DONE] Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
