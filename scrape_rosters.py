from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from bs4 import BeautifulSoup

from wp_utils import (
    UA,
    absolute,
    clean_height,
    extract_kv_from_dl,
    extract_kv_from_sidearm_bio,
    fetch,
    find_stats_tables,
    guess_sidearm_profile_links,
    looks_like_goalie_stats,
    looks_like_player_stats,
    norm_name,
    norm_space,
)

try:
    from playwright.sync_api import sync_playwright
except Exception:
    sync_playwright = None


@dataclass
class Team:
    school: str
    season: int
    roster_url: str
    schedule_url: Optional[str] = None
    stats_url: Optional[str] = None
    use_playwright: bool = False
    ssl_verify: bool = True


def load_teams(path: str) -> List[Team]:
    raw = yaml.safe_load(Path(path).read_text())
    teams = []
    for t in raw.get("teams", []):
        teams.append(
            Team(
                school=str(t.get("school", "")).strip(),
                season=int(t.get("season", 2025)),
                roster_url=str(t.get("roster_url", "")).strip(),
                schedule_url=str(t.get("schedule_url", "")).strip() or None,
                stats_url=str(t.get("stats_url", "")).strip() or None,
                use_playwright=bool(t.get("use_playwright", False)),
                ssl_verify=bool(t.get("ssl_verify", True)),
            )
        )
    return [t for t in teams if t.school and t.roster_url]


def fetch_html(url: str, *, use_playwright: bool, ssl_verify: bool, timeout_s: int = 30) -> str:
    if not use_playwright:
        return fetch(url, timeout=timeout_s, ssl_verify=ssl_verify)

    if sync_playwright is None:
        raise RuntimeError("Playwright not installed. Run: pip install playwright && python -m playwright install chromium")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=UA)
        page.goto(url, wait_until="networkidle", timeout=timeout_s * 1000)
        html = page.content()
        browser.close()
        return html


def parse_roster(html: str, base_url: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, "lxml")
    players: List[Dict[str, Any]] = []

    cards = soup.select(".sidearm-roster-player, .sidearm-roster-player-container")
    for card in cards:
        name_el = card.select_one(".sidearm-roster-player-name, .sidearm-roster-player-name a, h3, h2")
        name = norm_name(name_el.get_text(" ", strip=True) if name_el else "")
        if not name:
            continue

        a = card.select_one('a[href*="/roster/"]')
        profile_url = absolute(base_url, a.get("href")) if a and a.get("href") else ""

        number_el = card.select_one(".sidearm-roster-player-jersey-number, .sidearm-roster-player-number")
        pos_el = card.select_one(".sidearm-roster-player-position, .sidearm-roster-player-pos")
        year_el = card.select_one(".sidearm-roster-player-academic-year, .sidearm-roster-player-class")
        ht_el = card.select_one(".sidearm-roster-player-height, .sidearm-roster-player-ht")
        hometown_el = card.select_one(".sidearm-roster-player-hometown")

        players.append(
            {
                "name": name,
                "number": norm_space(number_el.get_text(" ", strip=True)) if number_el else "",
                "position": norm_space(pos_el.get_text(" ", strip=True)) if pos_el else "",
                "class_year": norm_space(year_el.get_text(" ", strip=True)) if year_el else "",
                "height": clean_height(norm_space(ht_el.get_text(" ", strip=True)) if ht_el else ""),
                "hometown": norm_space(hometown_el.get_text(" ", strip=True)) if hometown_el else "",
                "profile_url": profile_url,
            }
        )

    if players:
        return players

    table = soup.select_one("table")
    if table:
        headers = [norm_space(th.get_text(" ", strip=True)).lower() for th in table.select("thead th")]
        for tr in table.select("tbody tr"):
            tds = [norm_space(td.get_text(" ", strip=True)) for td in tr.select("td")]
            if not tds or (headers and len(tds) != len(headers)):
                continue
            row = dict(zip(headers, tds)) if headers else {}
            link = tr.select_one('a[href*="/roster/"]')
            nm = norm_name(row.get("name") or row.get("student-athlete") or row.get("player") or (link.get_text(" ", strip=True) if link else ""))
            if not nm:
                continue
            players.append(
                {
                    "name": nm,
                    "number": row.get("#") or row.get("no.") or row.get("no") or "",
                    "position": row.get("pos") or row.get("position") or "",
                    "class_year": row.get("yr") or row.get("year") or row.get("class") or "",
                    "height": clean_height(row.get("ht") or row.get("height") or ""),
                    "hometown": row.get("hometown") or "",
                    "profile_url": absolute(base_url, link.get("href")) if link and link.get("href") else "",
                }
            )

    return players


def parse_profile_fields(html: str) -> Dict[str, str]:
    soup = BeautifulSoup(html, "lxml")
    kv: Dict[str, str] = {}
    kv.update(extract_kv_from_dl(soup))
    kv.update(extract_kv_from_sidearm_bio(soup))
    return {k.lower().strip().rstrip(":"): v for k, v in kv.items()}


def scrape_record_from_schedule(html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")
    record_text = ""
    rec_el = soup.select_one(".sidearm-schedule-record, .sidearm-schedule__record, .schedule-record, .record")
    if rec_el:
        record_text = norm_space(rec_el.get_text(" ", strip=True))

    w = l = t = 0
    for el in soup.select(".sidearm-schedule-game-result, .schedule-result"):
        txt = norm_space(el.get_text(" ", strip=True))
        if txt.startswith("W"):
            w += 1
        elif txt.startswith("L"):
            l += 1
        elif txt.startswith("T"):
            t += 1

    if not record_text and (w + l + t) > 0:
        record_text = f"{w}-{l}" + (f"-{t}" if t else "")

    return {"record_text": record_text, "record_w": w or None, "record_l": l or None, "record_t": t or None}


def scrape_player_stats(html: str) -> Dict[str, Dict[str, Any]]:
    player_stats: Dict[str, Dict[str, Any]] = {}
    for caption, headers, rows in find_stats_tables(html):
        if looks_like_player_stats(headers) or looks_like_goalie_stats(headers):
            hlow = [h.lower() for h in headers]
            name_idx = 0
            for key in ["name", "player", "student-athlete"]:
                if key in hlow:
                    name_idx = hlow.index(key)
                    break
            for r in rows:
                if name_idx >= len(r):
                    continue
                nm = norm_name(r[name_idx])
                if not nm:
                    continue
                d = player_stats.setdefault(nm, {})
                for i, h in enumerate(headers):
                    if i >= len(r):
                        continue
                    hk = norm_space(h).lower()
                    if hk in ("", "name", "player", "student-athlete"):
                        continue
                    prefix = "gk_" if looks_like_goalie_stats(headers) else "stat_"
                    d[prefix + hk] = r[i]
    return player_stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teams", default="teams.yaml")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--out", default="waterpolo_rosters.csv")
    ap.add_argument("--xlsx", default="")
    ap.add_argument("--errors", default="waterpolo_rosters_errors.csv")
    ap.add_argument("--use-playwright", action="store_true")
    ap.add_argument("--max-players", type=int, default=0)
    args = ap.parse_args()

    teams = [t for t in load_teams(args.teams) if int(t.season) == int(args.season)]
    if not teams:
        raise SystemExit(f"No teams found for season={args.season} in {args.teams}")

    rows: List[Dict[str, Any]] = []
    errs: List[Dict[str, Any]] = []

    for team in teams:
        try:
            roster_html = fetch_html(team.roster_url, use_playwright=(args.use_playwright or team.use_playwright), ssl_verify=team.ssl_verify)
            players = parse_roster(roster_html, team.roster_url)

            if not players:
                links = guess_sidearm_profile_links(roster_html, team.roster_url)
                players = [{"name": "", "profile_url": u} for u in links]

            if args.max_players and len(players) > args.max_players:
                players = players[: args.max_players]

            team_record = {}
            if team.schedule_url:
                try:
                    sch = fetch_html(team.schedule_url, use_playwright=False, ssl_verify=team.ssl_verify)
                    team_record = scrape_record_from_schedule(sch)
                except Exception as e:
                    errs.append({"school": team.school, "url": team.schedule_url, "stage": "schedule", "error": str(e)})

            stats_by_player = {}
            if team.stats_url:
                try:
                    st = fetch_html(team.stats_url, use_playwright=False, ssl_verify=team.ssl_verify)
                    stats_by_player = scrape_player_stats(st)
                except Exception as e:
                    errs.append({"school": team.school, "url": team.stats_url, "stage": "stats", "error": str(e)})

            for p in players:
                profile_url = p.get("profile_url", "")
                prof_kv: Dict[str, str] = {}
                if profile_url:
                    try:
                        prof_html = fetch_html(profile_url, use_playwright=(args.use_playwright or team.use_playwright), ssl_verify=team.ssl_verify)
                        prof_kv = parse_profile_fields(prof_html)
                        if not p.get("name"):
                            soup = BeautifulSoup(prof_html, "lxml")
                            h1 = soup.select_one("h1")
                            p["name"] = norm_name(h1.get_text(" ", strip=True) if h1 else "")
                    except Exception as e:
                        errs.append({"school": team.school, "url": profile_url, "stage": "profile", "error": str(e)})

                def pick(*keys: str) -> str:
                    for k in keys:
                        v = prof_kv.get(k.lower())
                        if v:
                            return v
                    return ""

                name = norm_name(p.get("name", ""))
                row = {
                    "season": team.season,
                    "school": team.school,
                    "player_name": name,
                    "jersey_number": p.get("number", ""),
                    "position": p.get("position", "") or pick("position", "pos"),
                    "class_year": p.get("class_year", "") or pick("academic year", "class", "year"),
                    "height": clean_height(p.get("height", "") or pick("height")),
                    "weight": pick("weight"),
                    "hometown": p.get("hometown", "") or pick("hometown"),
                    "high_school": pick("high school", "highschool"),
                    "profile_url": profile_url,
                    "roster_url": team.roster_url,
                    "team_record_2025": team_record.get("record_text", ""),
                    "team_w_2025": team_record.get("record_w", ""),
                    "team_l_2025": team_record.get("record_l", ""),
                    "team_t_2025": team_record.get("record_t", ""),
                }

                if name in stats_by_player:
                    row.update(stats_by_player[name])

                rows.append(row)

            print(f"[OK] {team.school}: {len(players)} players")
        except Exception as e:
            print(f"[WARN] {team.school}: {e}")
            errs.append({"school": team.school, "url": team.roster_url, "stage": "roster", "error": str(e)})

    df = pd.DataFrame(rows)
    if "player_name" in df.columns:
        df = df[df["player_name"].astype(str).str.strip() != ""]

    df.to_csv(args.out, index=False)
    if args.xlsx:
        df.to_excel(args.xlsx, index=False)

    pd.DataFrame(errs).to_csv(args.errors, index=False)
    print(f"[DONE] Wrote {len(df)} rows to {args.out} and {len(errs)} errors to {args.errors}")


if __name__ == "__main__":
    main()
