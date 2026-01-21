#!/usr/bin/env python3
"""
enrich_swimcloud_hs_50free.py

Enriches a water polo roster CSV with each athlete's *high-school-era* SwimCloud best time
for the 50 Freestyle (SCY by default).

Key design choices:
- SwimCloud typically aggregates times from high school + college; to approximate "HS time",
  we select the athlete's best time with a swim date <= HS_CUTOFF (default: Aug 1 of class_year).
  Example: class_year=2024 => cutoff=2024-08-01.

- SwimCloud search pages may be JS-heavy; this script tries multiple extraction strategies:
  1) HTML anchors (/swimmer/<id>/)
  2) Regex over raw HTML for /swimmer/<id>/
  3) (Optional) Playwright rendering if --use-playwright is enabled and Playwright is installed.

Outputs:
Adds columns:
- best_50free_hs_scy
- best_50free_hs_date
- swimcloud_swimmer_url
- swimcloud_match_score
- swimcloud_note

Usage:
  python enrich_swimcloud_hs_50free.py --in waterpolo_rosters.csv --out waterpolo_rosters_with_50free.csv
  python enrich_swimcloud_hs_50free.py --in waterpolo_rosters.csv --out out.csv --use-playwright

Notes:
- Be considerate with request rate. Default delay is 0.8s between HTTP calls.
- If you have many rows, consider running with --limit for testing, then full run overnight.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

SWIMCLOUD_BASE = "https://www.swimcloud.com"

TIME_RE = re.compile(r"^\s*(\d{1,2}:)?\d{1,2}\.\d{1,2}\s*$")  # e.g., 21.34 or 1:45.67
SWIMMER_URL_RE = re.compile(r"/swimmer/(\d+)/")

MONTHS = {m: i for i, m in enumerate(
    ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], start=1
)}

def parse_time_to_seconds(s: str) -> Optional[float]:
    s = (s or "").strip()
    if not TIME_RE.match(s):
        return None
    if ":" in s:
        mm, rest = s.split(":", 1)
        try:
            return int(mm) * 60.0 + float(rest)
        except Exception:
            return None
    try:
        return float(s)
    except Exception:
        return None

def parse_us_date(s: str) -> Optional[dt.date]:
    """
    Handles typical SwimCloud date strings like "Feb 14, 2022" or "2/14/2022".
    """
    s = (s or "").strip()
    # 2/14/2022
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", s)
    if m:
        mo, da, yr = map(int, m.groups())
        try:
            return dt.date(yr, mo, da)
        except Exception:
            return None
    # Feb 14, 2022
    m = re.match(r"^([A-Za-z]{3,9})\s+(\d{1,2}),\s*(\d{4})$", s)
    if m:
        mon_s, da_s, yr_s = m.groups()
        mon3 = mon_s[:3].title()
        mo = MONTHS.get(mon3)
        if not mo:
            return None
        try:
            return dt.date(int(yr_s), mo, int(da_s))
        except Exception:
            return None
    return None

@dataclass
class Candidate:
    swimmer_url: str
    score: float
    note: str

def http_get(url: str, *, session: requests.Session, timeout: int = 25) -> str:
    r = session.get(url, timeout=timeout, headers={"User-Agent": UA})
    r.raise_for_status()
    return r.text

def extract_swimmer_urls_from_html(html: str) -> List[str]:
    urls: List[str] = []
    # 1) anchors
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        m = SWIMMER_URL_RE.search(href)
        if m:
            urls.append(f"{SWIMCLOUD_BASE}/swimmer/{m.group(1)}/")
    # 2) regex on raw html
    for m in SWIMMER_URL_RE.finditer(html):
        urls.append(f"{SWIMCLOUD_BASE}/swimmer/{m.group(1)}/")
    # de-dupe preserving order
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

def swimcloud_search_urls(query: str, *, session: requests.Session, use_playwright: bool) -> List[str]:
    """
    Returns a list of swimmer profile URLs from SwimCloud search.
    Tries both /search/ and /search/?q= styles.
    """
    urls: List[str] = []
    # Try plain requests first
    for url in [
        f"{SWIMCLOUD_BASE}/search/?q={requests.utils.quote(query)}",
        f"{SWIMCLOUD_BASE}/search/?q={requests.utils.quote(query)}&c=swimmers",
    ]:
        try:
            html = http_get(url, session=session)
            urls.extend(extract_swimmer_urls_from_html(html))
            if urls:
                return urls
        except Exception:
            pass

    if use_playwright:
        try:
            from playwright.sync_api import sync_playwright  # type: ignore
        except Exception:
            return urls

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(user_agent=UA)
                page.goto(f"{SWIMCLOUD_BASE}/search/?q={query}", wait_until="networkidle", timeout=45000)
                html = page.content()
                browser.close()
            urls = extract_swimmer_urls_from_html(html)
        except Exception:
            return urls

    return urls

def parse_swimmer_times(html: str) -> List[Dict[str, object]]:
    """
    Extracts rows from common SwimCloud times tables.
    This is intentionally robust and uses multiple strategies because the markup varies.

    Returns rows like:
      {"event": "50 Freestyle", "time": 21.34, "date": datetime.date(...), "course": "SCY"}
    """
    rows: List[Dict[str, object]] = []
    soup = BeautifulSoup(html, "html.parser")

    # Heuristic A: tables that include "Event" header and time strings.
    for table in soup.find_all("table"):
        # find headers
        headers = [th.get_text(" ", strip=True).lower() for th in table.find_all("th")]
        if not headers:
            continue
        # likely a times table if it has event and time-ish header
        if not any("event" in h for h in headers):
            continue
        if not any("time" in h or "result" in h for h in headers):
            continue

        for tr in table.find_all("tr"):
            tds = tr.find_all(["td", "th"])
            if len(tds) < 2:
                continue
            cells = [td.get_text(" ", strip=True) for td in tds]
            cell_join = " | ".join(cells)

            # event
            event = None
            for c in cells[:3]:
                if "freestyle" in c.lower() and "50" in c:
                    event = "50 Freestyle"
                    break
            if event != "50 Freestyle":
                continue

            # time: first token that looks like a time
            tval = None
            for c in cells:
                sec = parse_time_to_seconds(c)
                if sec is not None:
                    tval = sec
                    break

            if tval is None:
                continue

            # date (optional)
            dval = None
            for c in cells:
                d = parse_us_date(c)
                if d:
                    dval = d
                    break

            # course guess
            course = None
            low = cell_join.lower()
            if "scy" in low:
                course = "SCY"
            elif "lcm" in low:
                course = "LCM"
            elif "scm" in low:
                course = "SCM"

            rows.append({"event": event, "time": tval, "date": dval, "course": course})

    # Heuristic B: fallback regex near "50 Freestyle" if tables fail
    if not rows:
        # Find snippets containing "50 Freestyle" and nearby time.
        text = soup.get_text("\n", strip=True)
        for line in text.splitlines():
            if "50" in line and "freestyle" in line.lower():
                # try to capture a time in same line
                m = re.search(r"(\d{1,2}:\d{2}\.\d{2}|\d{2}\.\d{2})", line)
                if m:
                    sec = parse_time_to_seconds(m.group(1))
                    if sec is not None:
                        rows.append({"event": "50 Freestyle", "time": sec, "date": None, "course": None})

    return rows

def cutoff_date_for_class_year(class_year: Optional[int], cutoff_month: int, cutoff_day: int) -> Optional[dt.date]:
    if not class_year:
        return None
    try:
        return dt.date(int(class_year), cutoff_month, cutoff_day)
    except Exception:
        return None

def score_candidate_profile(html: str, name: str, hometown: str, class_year: Optional[int]) -> Tuple[float, str]:
    """
    Scores how likely this swimmer page belongs to the athlete.
    We look for:
    - name tokens
    - hometown substring
    - class year (if present)
    """
    soup = BeautifulSoup(html, "html.parser")
    page_text = soup.get_text(" ", strip=True).lower()

    score = 0.0
    note_parts = []

    # name match (token-based)
    tokens = [t for t in re.split(r"\s+", (name or "").strip()) if t]
    tokens_l = [t.lower() for t in tokens]
    n_hits = sum(1 for t in tokens_l if t in page_text)
    if tokens_l:
        score += 0.45 * (n_hits / max(1, len(tokens_l)))
        note_parts.append(f"name_hits={n_hits}/{len(tokens_l)}")

    # hometown
    if hometown:
        ht = hometown.lower()
        if ht in page_text:
            score += 0.35
            note_parts.append("hometown_match=1")
        else:
            # sometimes "City, ST" only; try city token
            city = hometown.split(",")[0].strip().lower()
            if city and city in page_text:
                score += 0.20
                note_parts.append("city_match=1")

    # class year match (approx)
    if class_year:
        if str(class_year) in page_text:
            score += 0.20
            note_parts.append("class_year_match=1")

    return score, ";".join(note_parts) if note_parts else "no_signals"

def best_hs_50free(rows: List[Dict[str, object]], cutoff: Optional[dt.date]) -> Tuple[Optional[str], Optional[str], str]:
    """
    From parsed rows, select best HS-era SCY 50 Free time.
    """
    # Prefer SCY rows when course present; otherwise accept unknown.
    filtered = []
    for r in rows:
        if r.get("event") != "50 Freestyle":
            continue
        t = r.get("time")
        if not isinstance(t, (int, float)):
            continue
        d = r.get("date")
        if cutoff and isinstance(d, dt.date) and d > cutoff:
            continue
        filtered.append(r)

    if not filtered:
        return None, None, "no_rows_after_hs_filter"

    # Prefer SCY
    scy = [r for r in filtered if (r.get("course") == "SCY")]
    pool = scy if scy else filtered

    best = min(pool, key=lambda r: float(r["time"]))  # type: ignore[index]
    best_time = float(best["time"])  # type: ignore[index]
    best_date = best.get("date")
    time_str = f"{best_time:.2f}"
    date_str = best_date.isoformat() if isinstance(best_date, dt.date) else ""
    note = "scy" if scy else "unknown_course"
    return time_str, date_str, note

def enrich_row(
    row: Dict[str, str],
    *,
    session: requests.Session,
    use_playwright: bool,
    delay_s: float,
    cutoff_month: int,
    cutoff_day: int,
) -> Dict[str, str]:
    name = (row.get("player_name") or "").strip()
    hometown = (row.get("hometown") or "").strip()
    class_year = row.get("class_year") or ""
    try:
        cy_int = int(class_year) if class_year else None
    except Exception:
        cy_int = None

    if not name:
        row["swimcloud_note"] = "missing_name"
        return row

    query = name
    if hometown:
        query = f"{name} {hometown}"

    # Search
    urls = swimcloud_search_urls(query, session=session, use_playwright=use_playwright)
    time.sleep(delay_s)

    if not urls:
        row["swimcloud_note"] = "no_search_results"
        return row

    cutoff = cutoff_date_for_class_year(cy_int, cutoff_month, cutoff_day)

    best_candidate: Optional[Candidate] = None
    best_time: Optional[str] = None
    best_date: Optional[str] = None

    # Evaluate up to first N candidates to keep it tractable
    for u in urls[:6]:
        try:
            html = http_get(u, session=session)
        except Exception:
            continue

        score, note = score_candidate_profile(html, name=name, hometown=hometown, class_year=cy_int)
        rows_times = parse_swimmer_times(html)
        t_str, d_str, t_note = best_hs_50free(rows_times, cutoff)
        if t_str is None:
            # Still keep candidate; maybe times section didn't parse.
            cand = Candidate(swimmer_url=u, score=score * 0.9, note=f"{note};{t_note}")
        else:
            # reward candidates that actually yield a time
            cand = Candidate(swimmer_url=u, score=score + 0.15, note=f"{note};{t_note}")

        if (best_candidate is None) or (cand.score > best_candidate.score):
            best_candidate = cand
            best_time = t_str
            best_date = d_str

        time.sleep(delay_s)

    if best_candidate is None:
        row["swimcloud_note"] = "candidates_unreachable"
        return row

    row["swimcloud_swimmer_url"] = best_candidate.swimmer_url
    row["swimcloud_match_score"] = f"{best_candidate.score:.3f}"
    row["swimcloud_note"] = best_candidate.note

    if best_time:
        row["best_50free_hs_scy"] = best_time
        row["best_50free_hs_date"] = best_date or ""

    return row

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", required=True, help="Input roster CSV")
    ap.add_argument("--out", dest="out_csv", required=True, help="Output CSV with HS 50 free times")
    ap.add_argument("--limit", type=int, default=0, help="Only process first N rows (debug)")
    ap.add_argument("--delay", type=float, default=0.8, help="Delay between HTTP calls (seconds)")
    ap.add_argument("--cutoff-mmdd", default="08-01", help="HS cutoff month-day (default Aug 1 of class_year)")
    ap.add_argument("--use-playwright", action="store_true", help="Render search pages with Playwright if needed")
    args = ap.parse_args()

    cutoff_mmdd = args.cutoff_mmdd.strip()
    m = re.match(r"^(\d{2})-(\d{2})$", cutoff_mmdd)
    if not m:
        raise SystemExit("--cutoff-mmdd must be MM-DD, e.g. 08-01")
    cutoff_month = int(m.group(1))
    cutoff_day = int(m.group(2))

    in_path = Path(args.in_csv)
    out_path = Path(args.out_csv)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    session = requests.Session()

    with in_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    # Ensure output fields
    fieldnames = list(rows[0].keys()) if rows else []
    for col in ["best_50free_hs_scy", "best_50free_hs_date", "swimcloud_swimmer_url", "swimcloud_match_score", "swimcloud_note"]:
        if col not in fieldnames:
            fieldnames.append(col)

    n = len(rows)
    for i, row in enumerate(rows, start=1):
        try:
            rows[i-1] = enrich_row(
                row,
                session=session,
                use_playwright=bool(args.use_playwright),
                delay_s=float(args.delay),
                cutoff_month=cutoff_month,
                cutoff_day=cutoff_day,
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            row["swimcloud_note"] = f"error:{type(e).__name__}:{e}"
        if i % 25 == 0 or i == n:
            print(f"[{i}/{n}] processed", file=sys.stderr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[DONE] wrote {len(rows)} rows -> {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
