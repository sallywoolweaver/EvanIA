#!/usr/bin/env python3
"""
scrape_rosters.py

Scrape NCAA men's water polo rosters for:
- school
- team record (from NCAA RPI page)
- player height
- hometown
- class + estimated graduation year

Outputs a CSV.

Usage:
  python scrape_rosters.py --out ncaa_mwp_rosters.csv --grad_base_year 2026
  python scrape_rosters.py --teams teams.yaml --out out.csv --sleep 0.75
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup


DEFAULT_RPI_URL = "https://www.ncaa.com/rankings/water-polo-men/nc/ncaa-mens-water-polo-rpi"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def parse_height_to_inches(h: str) -> Optional[int]:
    if not h:
        return None
    s = h.strip()
    # Common formats: 6-4, 6' 4'', 6'4", 6’4”, 6 4
    m = re.search(r"(\d)\s*[-'’]\s*(\d{1,2})", s)
    if not m:
        m = re.search(r"(\d)\s+(\d{1,2})", s)
    if not m:
        return None
    ft = int(m.group(1))
    inch = int(m.group(2))
    if ft < 4 or ft > 8 or inch < 0 or inch > 11:
        return None
    return ft * 12 + inch


def class_to_grad_year(class_raw: str, base_year: int) -> Optional[int]:
    if not class_raw:
        return None
    c = _norm(class_raw)
    # Normalize a few patterns:
    # r-fr, rs-fr, redshirt freshman, etc.
    c = c.replace("redshirt", "r").replace("rs", "r")
    if "grad" in c or c in {"gr", "gs", "g"}:
        return base_year
    if c.startswith("r-"):
        c = c[2:]
    # Handle typical class labels
    if c in {"fr", "freshman", "first-year", "fy"}:
        return base_year + 3
    if c in {"so", "sophomore"}:
        return base_year + 2
    if c in {"jr", "junior"}:
        return base_year + 1
    if c in {"sr", "senior"}:
        return base_year
    if "5" in c or "fifth" in c:
        return base_year
    return None


def load_teams(path: Path) -> List[Dict[str, str]]:
    obj = yaml.safe_load(path.read_text())
    teams = obj.get("teams", [])
    if not teams:
        raise ValueError(f"No teams found in {path}")
    return teams


def fetch_record_map(rpi_url: str, timeout_s: int = 20) -> Dict[str, str]:
    """
    Returns mapping: NCAA display team name -> record string (e.g., '23-5').
    """
    html = requests.get(rpi_url, timeout=timeout_s, headers={"User-Agent": "Mozilla/5.0"}).text
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table")
    if table is None:
        # Try pandas as fallback
        dfs = pd.read_html(html)
        if not dfs:
            return {}
        df = dfs[0]
        team_col = next((c for c in df.columns if _norm(str(c)) in {"team", "school"}), None)
        rec_col = next((c for c in df.columns if "record" in _norm(str(c))), None)
        if team_col and rec_col:
            return {str(t).strip(): str(r).strip() for t, r in zip(df[team_col], df[rec_col])}
        return {}

    headers = [_norm(th.get_text(" ", strip=True)) for th in table.find_all("th")]
    team_idx = None
    rec_idx = None
    for i, h in enumerate(headers):
        if h in {"team", "school"}:
            team_idx = i
        if "record" in h:
            rec_idx = i

    record_map: Dict[str, str] = {}
    for tr in table.find_all("tr"):
        tds = tr.find_all(["td", "th"])
        if not tds or rec_idx is None:
            continue
        cells = [td.get_text(" ", strip=True) for td in tds]

        if team_idx is None:
            if len(cells) >= 2:
                team = cells[1].strip()
            else:
                continue
        else:
            if len(cells) <= team_idx:
                continue
            team = cells[team_idx].strip()

        if len(cells) <= rec_idx:
            continue
        rec = cells[rec_idx].strip()
        if team and re.match(r"^\d+\s*-\s*\d+", rec):
            record_map[team] = rec.replace(" ", "")
    return record_map


def _extract_best_table(html: str) -> Optional[pd.DataFrame]:
    """
    Heuristic: choose the roster-ish table that contains 'Ht'/'Height' and 'Hometown' or 'Academic Year/Class'.
    """
    try:
        dfs = pd.read_html(html)
    except Exception:
        return None

    best = None
    best_score = -1
    for df in dfs:
        cols = [_norm(str(c)) for c in df.columns]
        score = 0
        if any(c == "ht" or c == "ht." or "height" in c for c in cols):
            score += 2
        if any("hometown" in c for c in cols):
            score += 2
        if any("academic year" in c or c == "class" or c == "yr" for c in cols):
            score += 1
        if any("name" in c for c in cols):
            score += 2
        if score > best_score and len(df) >= 8:
            best = df
            best_score = score
    return best


def scrape_roster_table(url: str, timeout_s: int = 25) -> Tuple[Optional[pd.DataFrame], str]:
    resp = requests.get(url, timeout=timeout_s, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    html = resp.text
    df = _extract_best_table(html)
    return df, html


def parse_roster_cards(html: str) -> List[Dict[str, str]]:
    """
    Fallback parser for SIDEARM-ish "List View" pages where table extraction fails.
    """
    soup = BeautifulSoup(html, "lxml")
    out: List[Dict[str, str]] = []

    for card in soup.select(".sidearm-roster-player, .sidearm-roster-player-container, .roster-card"):
        txt = card.get_text(" ", strip=True)
        if not txt or len(txt) < 10:
            continue

        name = None
        name_el = card.select_one(".sidearm-roster-player-name, .sidearm-roster-player-name a, .roster-name")
        if name_el:
            name = name_el.get_text(" ", strip=True)

        height = None
        ht_el = card.select_one(".sidearm-roster-player-height, .roster-height")
        if ht_el:
            height = ht_el.get_text(" ", strip=True)

        hometown = None
        htwn_el = card.select_one(".sidearm-roster-player-hometown, .roster-hometown")
        if htwn_el:
            hometown = htwn_el.get_text(" ", strip=True)

        cls = None
        cls_el = card.select_one(".sidearm-roster-player-academic-year, .sidearm-roster-player-year, .roster-year")
        if cls_el:
            cls = cls_el.get_text(" ", strip=True)

        if name:
            out.append({"Name": name, "Ht.": height or "", "Hometown": hometown or "", "Academic Year": cls or ""})

    return out


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    norm_map: Dict[str, str] = {}

    for c in cols:
        n = _norm(str(c))
        if n in {"name", "full name"} or "name" in n:
            norm_map[c] = "player"
        elif n in {"ht", "ht.", "height"} or "height" in n:
            norm_map[c] = "height_raw"
        elif "hometown" in n:
            norm_map[c] = "hometown"
        elif n in {"academic year", "class", "yr", "year"} or "academic year" in n:
            norm_map[c] = "class_raw"
        elif "pos" in n:
            norm_map[c] = "pos"
        elif "no" in n or "jersey" in n:
            norm_map[c] = "number"
        else:
            norm_map[c] = str(c)

    out = df.rename(columns=norm_map)

    for required in ["player", "height_raw", "hometown", "class_raw"]:
        if required not in out.columns:
            out[required] = ""
    return out


def clean_hometown(h: str) -> str:
    if not h:
        return ""
    s = h.strip()
    if "/" in s:
        s = s.split("/")[0].strip()
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teams", type=str, default="teams.yaml", help="Path to teams.yaml")
    ap.add_argument("--out", type=str, default="ncaa_mwp_rosters.csv", help="Output CSV path")
    ap.add_argument("--rpi_url", type=str, default=DEFAULT_RPI_URL, help="NCAA RPI URL used for team records")
    ap.add_argument("--grad_base_year", type=int, default=datetime.now().year, help="Base year for Sr graduation year (default: current year)")
    ap.add_argument("--sleep", type=float, default=0.75, help="Seconds to sleep between requests")
    ap.add_argument("--timeout", type=int, default=25, help="HTTP timeout seconds")
    args = ap.parse_args()

    teams_path = Path(args.teams)
    if not teams_path.exists():
        teams_path = Path(__file__).resolve().parent / args.teams

    teams = load_teams(teams_path)
    print(f"[info] Loaded {len(teams)} teams from {teams_path}")

    print(f"[info] Fetching records from NCAA RPI page: {args.rpi_url}")
    record_map = fetch_record_map(args.rpi_url, timeout_s=args.timeout)
    print(f"[info] Records found: {len(record_map)} teams")

    rows: List[Dict[str, Any]] = []
    failures: List[Tuple[str, str]] = []

    for t in teams:
        school = t["school"]
        ncaa_name = t.get("ncaa_name", school)
        roster_url = t["roster_url"]
        record = record_map.get(ncaa_name, "")

        print(f"[team] {school} | record='{record}' | roster={roster_url}")

        try:
            df, html = scrape_roster_table(roster_url, timeout_s=args.timeout)
            if df is None or len(df) == 0:
                card_rows = parse_roster_cards(html)
                if card_rows:
                    df = pd.DataFrame(card_rows)

            if df is None or len(df) == 0:
                raise RuntimeError("No roster table detected (table/card parsers failed).")

            df = normalize_columns(df)

            for _, r in df.iterrows():
                player = str(r.get("player", "")).strip()
                if not player or player.lower() in {"nan", "name"}:
                    continue

                height_raw = str(r.get("height_raw", "")).strip()
                hometown = clean_hometown(str(r.get("hometown", "")).strip())
                class_raw = str(r.get("class_raw", "")).strip()

                rows.append({
                    "school": school,
                    "record": record,
                    "player": player,
                    "height_raw": height_raw,
                    "height_in": parse_height_to_inches(height_raw),
                    "hometown": hometown,
                    "class_raw": class_raw,
                    "grad_year_est": class_to_grad_year(class_raw, args.grad_base_year),
                    "roster_url": roster_url,
                })

        except Exception as e:
            failures.append((school, f"{type(e).__name__}: {e}"))
            print(f"[warn] Failed {school}: {type(e).__name__}: {e}", file=sys.stderr)

        time.sleep(max(0.0, args.sleep))

    out_path = Path(args.out)
    out_df = pd.DataFrame(rows).sort_values(["school", "player"]).reset_index(drop=True)
    out_df.to_csv(out_path, index=False)
    print(f"[done] Wrote {len(out_df)} rows to {out_path}")

    if failures:
        fail_path = out_path.with_suffix(".failures.txt")
        fail_path.write_text("\n".join([f"{s}\t{err}" for s, err in failures]) + "\n")
        print(f"[warn] {len(failures)} teams failed. See: {fail_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
