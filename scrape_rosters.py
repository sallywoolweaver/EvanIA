#!/usr/bin/env python3
"""
scrape_rosters.py

Scrapes NCAA men's water polo roster pages from a curated list of team roster URLs
(typically SidearmSports/Presto sites) and outputs a CSV and/or Excel file with:

- school
- player_name
- position
- height
- hometown
- class_year
- record (team W-L from NCAA RPI page, best-effort)
- source_url (the roster URL used)

Notes:
- College roster pages are not standardized. This script supports common patterns
  (Sidearm "sidearm-roster-player", and PrestoSports "roster-player").
- Some athletics sites may rate-limit or block scraping. This script will continue
  on errors and write whatever data it could retrieve.

Usage:
  python scrape_rosters.py --out waterpolo_rosters.csv
  python scrape_rosters.py --out waterpolo_rosters.csv --xls waterpolo_rosters.xlsx
  python scrape_rosters.py --team UCLA --out ucla.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup

DEFAULT_TEAMS_YAML = Path(__file__).resolve().parent / "teams.yaml"

# Updated NCAA RPI page (your previous URL 404s)
DEFAULT_RPI_URL = "https://www.ncaa.com/rankings/waterpolo-men/nc/rpi"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/121.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

TIMEOUT = 20


@dataclass
class Team:
    school: str
    roster_url: str
    ncaa_name: Optional[str] = None  # name as it appears on NCAA page (optional)


def load_teams(path: Path) -> List[Team]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    teams = []
    for t in data.get("teams", []):
        teams.append(
            Team(
                school=str(t["school"]).strip(),
                roster_url=str(t["roster_url"]).strip(),
                ncaa_name=(str(t.get("ncaa_name")).strip() if t.get("ncaa_name") else None),
            )
        )
    if not teams:
        raise ValueError(f"No teams found in {path}")
    return teams


def _request_get(url: str, max_retries: int = 3, sleep_s: float = 1.2) -> requests.Response:
    last_exc: Optional[Exception] = None
    for i in range(max_retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            # Do not raise here; caller may want to handle non-200
            return r
        except Exception as e:
            last_exc = e
            if i < max_retries - 1:
                time.sleep(sleep_s * (i + 1))
    raise RuntimeError(f"Failed GET {url}: {last_exc}")


def fetch_ncaa_records(rpi_url: str) -> Dict[str, str]:
    """
    Returns a mapping {school_name_on_ncaa: 'W-L'} (best-effort).
    This page format can change; we try table parsing first, then a robust text scan.
    """
    r = _request_get(rpi_url)
    if r.status_code >= 400:
        # Best-effort: if NCAA page is blocked/changed, skip records
        return {}

    soup = BeautifulSoup(r.text, "lxml")

    # 1) Try HTML tables
    try:
        tables = pd.read_html(r.text)
        # Heuristic: pick the first table that has a "School" column and "Record"
        for df in tables:
            cols = [c.strip().lower() for c in df.columns.astype(str)]
            if any("school" == c for c in cols) and any("record" == c for c in cols):
                # normalize
                school_col = df.columns[cols.index("school")]
                record_col = df.columns[cols.index("record")]
                m = {}
                for _, row in df.iterrows():
                    sname = str(row[school_col]).strip()
                    rec = str(row[record_col]).strip()
                    if sname and re.match(r"^\d+\-\d+", rec):
                        m[sname] = rec
                if m:
                    return m
    except Exception:
        pass

    # 2) Fallback: scan visible text lines like: "1 UCLA 27-2 MPSF"
    text = soup.get_text("\n")
    records: Dict[str, str] = {}
    for line in text.splitlines():
        line = " ".join(line.split())
        if not line:
            continue
        # Must start with rank number, and contain a W-L token
        if not re.match(r"^\d+\s+", line):
            continue
        toks = line.split()
        if len(toks) < 3:
            continue
        # find first token that looks like record
        rec_i = None
        for j in range(1, min(len(toks), 8)):  # record tends to be early
            if re.fullmatch(r"\d+\-\d+", toks[j]):
                rec_i = j
                break
        if rec_i is None:
            continue
        rank = toks[0]
        record = toks[rec_i]
        school = " ".join(toks[1:rec_i]).strip()
        if school and record:
            records[school] = record
    return records


def normalize_height(h: str) -> str:
    h = (h or "").strip()
    if not h:
        return ""
    # Common forms: 6-4, 6'4", 6' 4", 6 ft 4 in
    m = re.match(r"^\s*(\d)\s*[\-'\s]\s*(\d{1,2})\s*(?:\"|in|)\s*$", h)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    m = re.search(r"(\d)\s*ft\s*(\d{1,2})\s*in", h, re.I)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return h


def parse_sidearm_roster(soup: BeautifulSoup) -> List[dict]:
    rows = []
    players = soup.select(".sidearm-roster-player")
    if not players:
        return rows

    for p in players:
        name = p.select_one(".sidearm-roster-player-name")
        if name:
            name = " ".join(name.get_text(" ").split())
        else:
            name = ""

        pos = p.select_one(".sidearm-roster-player-position-short")
        pos = " ".join(pos.get_text(" ").split()) if pos else ""

        height = p.select_one(".sidearm-roster-player-height")
        height = normalize_height(" ".join(height.get_text(" ").split())) if height else ""

        hometown = p.select_one(".sidearm-roster-player-hometown")
        hometown = " ".join(hometown.get_text(" ").split()) if hometown else ""

        year = p.select_one(".sidearm-roster-player-academic-year")
        year = " ".join(year.get_text(" ").split()) if year else ""

        if name:
            rows.append(
                dict(
                    player_name=name,
                    position=pos,
                    height=height,
                    hometown=hometown,
                    class_year=year,
                )
            )
    return rows


def parse_presto_roster(soup: BeautifulSoup) -> List[dict]:
    rows = []
    players = soup.select(".roster-player")
    if not players:
        # another common presto pattern: table rows
        table = soup.select_one("table.roster") or soup.select_one("table")
        if table:
            # try read_html for roster-like columns
            try:
                dfs = pd.read_html(str(table))
                for df in dfs:
                    cols = [c.strip().lower() for c in df.columns.astype(str)]
                    # Common columns: Name, Pos, Ht, Hometown, Yr
                    if any("name" in c for c in cols) and any(c in cols for c in ["pos", "position"]):
                        name_col = df.columns[[i for i,c in enumerate(cols) if "name" in c][0]]
                        pos_col = df.columns[[i for i,c in enumerate(cols) if c in ["pos", "position"]][0]]
                        ht_col = df.columns[[i for i,c in enumerate(cols) if c in ["ht", "height"]][0]] if any(c in cols for c in ["ht","height"]) else None
                        hometown_col = df.columns[[i for i,c in enumerate(cols) if "hometown" in c][0]] if any("hometown" in c for c in cols) else None
                        yr_col = df.columns[[i for i,c in enumerate(cols) if c in ["yr","year","class"]][0]] if any(c in cols for c in ["yr","year","class"]) else None
                        for _, r in df.iterrows():
                            nm = str(r[name_col]).strip()
                            if not nm or nm.lower() == "nan":
                                continue
                            rows.append(dict(
                                player_name=nm,
                                position=str(r[pos_col]).strip(),
                                height=normalize_height(str(r[ht_col]).strip()) if ht_col is not None else "",
                                hometown=str(r[hometown_col]).strip() if hometown_col is not None else "",
                                class_year=str(r[yr_col]).strip() if yr_col is not None else "",
                            ))
                        if rows:
                            return rows
            except Exception:
                pass
        return rows

    for p in players:
        name = p.select_one(".name") or p.select_one(".roster-name")
        name = " ".join(name.get_text(" ").split()) if name else ""

        pos = p.select_one(".position") or p.select_one(".pos")
        pos = " ".join(pos.get_text(" ").split()) if pos else ""

        height = p.select_one(".height") or p.select_one(".ht")
        height = normalize_height(" ".join(height.get_text(" ").split())) if height else ""

        hometown = p.select_one(".hometown") or p.select_one(".home-town")
        hometown = " ".join(hometown.get_text(" ").split()) if hometown else ""

        year = p.select_one(".year") or p.select_one(".class")
        year = " ".join(year.get_text(" ").split()) if year else ""

        if name:
            rows.append(
                dict(
                    player_name=name,
                    position=pos,
                    height=height,
                    hometown=hometown,
                    class_year=year,
                )
            )
    return rows


def scrape_team_roster(team: Team) -> Tuple[List[dict], Optional[str]]:
    """
    Returns (rows, error_message_if_any).
    """
    try:
        r = _request_get(team.roster_url)
        if r.status_code >= 400:
            return [], f"HTTP {r.status_code}"
        soup = BeautifulSoup(r.text, "lxml")

        rows = parse_sidearm_roster(soup)
        if not rows:
            rows = parse_presto_roster(soup)

        if not rows:
            return [], "No players parsed (site layout not recognized or content requires JS)"
        return rows, None
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teams", type=Path, default=DEFAULT_TEAMS_YAML, help="Path to teams.yaml")
    ap.add_argument("--out", type=Path, required=True, help="Output CSV path")
    ap.add_argument("--xls", type=Path, default=None, help="Optional Excel .xlsx output path")
    ap.add_argument("--rpi-url", type=str, default=DEFAULT_RPI_URL, help="NCAA RPI URL used for team records")
    ap.add_argument("--team", type=str, default=None, help="Only scrape one school (matches teams.yaml school field)")
    ap.add_argument("--sleep", type=float, default=0.5, help="Sleep seconds between team requests")
    args = ap.parse_args()

    teams = load_teams(args.teams)
    if args.team:
        teams = [t for t in teams if t.school.lower() == args.team.lower()]
        if not teams:
            raise SystemExit(f"No team matched --team {args.team!r}")

    # Fetch records best-effort
    records = fetch_ncaa_records(args.rpi_url)

    out_rows: List[dict] = []
    errors: List[dict] = []

    for idx, team in enumerate(teams, start=1):
        ncaa_key = team.ncaa_name or team.school
        team_record = records.get(ncaa_key, records.get(team.school, ""))

        rows, err = scrape_team_roster(team)
        if err:
            errors.append(
                dict(
                    school=team.school,
                    roster_url=team.roster_url,
                    error=err,
                )
            )
            print(f"[WARN] {team.school}: {err}", file=sys.stderr)
        else:
            for r in rows:
                out_rows.append(
                    dict(
                        school=team.school,
                        record=team_record,
                        source_url=team.roster_url,
                        **r,
                    )
                )
            print(f"[OK] {team.school}: {len(rows)} players")

        if idx < len(teams) and args.sleep > 0:
            time.sleep(args.sleep)

    # Always write outputs (even if empty), so you never end up with "nothing outputted"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(out_rows, columns=[
        "school", "record", "player_name", "position", "height", "hometown", "class_year", "source_url"
    ])
    df.to_csv(args.out, index=False)

    if args.xls:
        args.xls.parent.mkdir(parents=True, exist_ok=True)
        # Excel-friendly .xlsx (opens in Excel; avoids legacy .xls limitations)
        df.to_excel(args.xls, index=False)

    if errors:
        err_path = args.out.with_name(args.out.stem + "_errors.csv")
        pd.DataFrame(errors).to_csv(err_path, index=False)
        print(f"[DONE] Wrote {len(df)} rows to {args.out} and {len(errors)} errors to {err_path}")
    else:
        print(f"[DONE] Wrote {len(df)} rows to {args.out}")

    if args.xls:
        print(f"[DONE] Wrote Excel file to {args.xls}")


if __name__ == "__main__":
    main()