from __future__ import annotations

import argparse
import json
import re
import time
from typing import Any, Dict, Optional, Tuple
from urllib.parse import quote_plus

import pandas as pd
from rapidfuzz import fuzz

from wp_utils import UA, norm_name, norm_space

try:
    from playwright.sync_api import sync_playwright
except Exception:
    sync_playwright = None


def _extract_next_data(html: str) -> Optional[Dict[str, Any]]:
    m = re.search(r'<script[^>]+id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def _walk(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for it in obj:
            yield from _walk(it)


def _parse_time_to_seconds(t: str) -> Optional[float]:
    t = norm_space(t)
    if not t:
        return None
    if ":" in t:
        mm, ss = t.split(":", 1)
        try:
            return float(mm) * 60.0 + float(ss)
        except Exception:
            return None
    try:
        return float(t)
    except Exception:
        return None


def _pick_50free_from_next(next_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    best = None
    best_sec = None
    for d in _walk(next_data):
        ev = d.get("event") or d.get("eventName") or d.get("name") or d.get("event_name") or ""
        if isinstance(ev, str) and "50" in ev and "Free" in ev:
            t = d.get("time") or d.get("result") or d.get("mark") or d.get("bestTime") or d.get("value")
            if isinstance(t, str):
                sec = _parse_time_to_seconds(t)
                if sec is not None and (best_sec is None or sec < best_sec):
                    best_sec = sec
                    best = {
                        "event": ev,
                        "time": t,
                        "course": d.get("course") or d.get("courseName") or "",
                        "meet": d.get("meet") or d.get("meetName") or "",
                    }
    return best


def swimcloud_search_and_pick(page, name: str, hometown: str) -> Optional[str]:
    q = quote_plus(f"{name} {hometown}".strip())
    url = f"https://www.swimcloud.com/search/?q={q}"
    page.goto(url, wait_until="networkidle", timeout=60000)

    links = []
    for a in page.query_selector_all('a[href*="/swimmer/"]'):
        href = a.get_attribute("href") or ""
        if href.startswith("/swimmer/"):
            links.append("https://www.swimcloud.com" + href.split("?")[0])

    seen = set()
    uniq = []
    for u in links:
        if u not in seen:
            seen.add(u)
            uniq.append(u)

    best_u = None
    best_s = -1
    for u in uniq[:20]:
        s = fuzz.token_sort_ratio(u.split("/swimmer/")[-1].replace("-", " "), name)
        if hometown:
            s += 0.2 * fuzz.token_sort_ratio(u, hometown)
        if s > best_s:
            best_s = s
            best_u = u
    return best_u


def scrape_50free(page, swimmer_url: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    page.goto(swimmer_url, wait_until="networkidle", timeout=60000)
    html = page.content()

    next_data = _extract_next_data(html)
    if next_data:
        best = _pick_50free_from_next(next_data)
        if best:
            return best.get("time"), str(best.get("course") or ""), str(best.get("meet") or "")

    # DOM fallback
    try:
        for tr in page.query_selector_all("tr"):
            txt = norm_space(tr.inner_text())
            if "50" in txt and "Freestyle" in txt:
                m = re.search(r"\b(\d+:\d{2}\.\d{2}|\d{2}\.\d{2})\b", txt)
                if m:
                    course = "SCY" if ("SCY" in txt or "Yards" in txt) else ("LCM" if "LCM" in txt else "")
                    return m.group(1), course, ""
    except Exception:
        pass

    return None, None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--xlsx", default="")
    ap.add_argument("--sleep", type=float, default=1.25)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if sync_playwright is None:
        raise SystemExit("Playwright not installed. Run: pip install playwright && python -m playwright install chromium")

    df = pd.read_csv(args.inp)
    df_work = df.head(args.limit).copy() if args.limit else df.copy()
    df_rest = df.iloc[args.limit:].copy() if args.limit else df.iloc[0:0].copy()

    for col in ["swimcloud_url", "swimcloud_50free_time", "swimcloud_50free_course", "swimcloud_50free_meet"]:
        if col not in df_work.columns:
            df_work[col] = ""

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=UA)

        for i, row in df_work.iterrows():
            name = norm_name(str(row.get("player_name", "") or ""))
            hometown = norm_space(str(row.get("hometown", "") or ""))

            swimmer_url = norm_space(str(row.get("swimcloud_url", "") or ""))
            if not swimmer_url:
                try:
                    swimmer_url = swimcloud_search_and_pick(page, name, hometown) or ""
                except Exception:
                    swimmer_url = ""

            df_work.at[i, "swimcloud_url"] = swimmer_url

            t = course = meet = ""
            if swimmer_url:
                try:
                    tt, cc, mm = scrape_50free(page, swimmer_url)
                    t = tt or ""
                    course = cc or ""
                    meet = mm or ""
                except Exception:
                    pass

            df_work.at[i, "swimcloud_50free_time"] = t
            df_work.at[i, "swimcloud_50free_course"] = course
            df_work.at[i, "swimcloud_50free_meet"] = meet

            if args.sleep > 0:
                time.sleep(args.sleep)

        browser.close()

    out = pd.concat([df_work, df_rest], axis=0)
    out.to_csv(args.out, index=False)
    if args.xlsx:
        out.to_excel(args.xlsx, index=False)

    print(f"[DONE] Wrote {len(out)} rows to {args.out}")


if __name__ == "__main__":
    main()
