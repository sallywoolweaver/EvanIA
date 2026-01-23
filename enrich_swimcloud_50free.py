#!/usr/bin/env python3
"""
enrich_swimcloud_50free.py

Enrich a roster CSV with a SwimCloud *best* 50 Freestyle time (ideally HS times).

Key fix vs prior versions:
- SwimCloud search is a JS dropdown. The /search/?q=... page often embeds the dropdown
  candidates in:  #global-search[data-global-search-items="...json..."]
- We parse that attribute directly (fast + reliable) instead of waiting for a combobox.

Workflow:
- (Playwright) Complete onboarding gate once (gender + zip) to set cookies.
- For each row:
  - Clean name (remove "HS", dashes, trailing "Season", suffixes).
  - Load /search/?q=<name>
  - Parse candidates from #global-search data-global-search-items JSON
  - Score candidates: HS match > hometown match > exact name
  - Try top N candidates: open /swimmer/<id>/times/ and extract best 50 Free
  - If found, write:
      swimcloud_url
      swimcloud_50free_time
      swimcloud_50free_course
      swimcloud_50free_note
      swimcloud_matched_high_school

Usage (recommended):
  /home/compsci/Desktop/EvanIA/.venv/bin/python enrich_swimcloud_50free.py \
    --in waterpolo_rosters_2025.csv \
    --out waterpolo_rosters_2025_with_50free.csv \
    --playwright \
    --pw-headful \
    --pw-user-data /home/compsci/Desktop/EvanIA/.pw_profile_swimcloud \
    --debug-html /home/compsci/Desktop/EvanIA/debug_html \
    --timeout 25 \
    --sleep 0.6 \
    --min-grad-year 2019

If you want it faster once stable, drop --pw-headful.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import html as html_lib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None  # type: ignore

PW_AVAILABLE = False
try:
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError  # type: ignore

    PW_AVAILABLE = True
except Exception:
    PW_AVAILABLE = False

from urllib.parse import quote_plus

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
)

TIME_PAT = re.compile(r"\b(\d{1,2}:\d{2}\.\d{2}|\d{1,2}\.\d{2})\b")
COURSE_PAT = re.compile(r"\b(SCY|SCM|LCM)\b", re.IGNORECASE)

EVENT_KEYS = ["50 Free", "50 Freestyle", "50 Yard Freestyle", "50 Y Free", "50 FR"]

CLASS_OF_PATTERNS = [
    re.compile(r"\bClass of\s*(\d{4})\b", re.IGNORECASE),
    re.compile(r"\bGraduation\s*Year\s*[:\-]?\s*(\d{4})\b", re.IGNORECASE),
    re.compile(r"\bGrad(?:uation)?\s*[:\-]?\s*(\d{4})\b", re.IGNORECASE),
]


@dataclasses.dataclass
class EnrichResult:
    url: str = ""
    time: str = ""
    course: str = ""
    note: str = ""
    error: str = ""
    matched_high_school: str = ""


def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def norm_name(name: str) -> str:
    """
    Clean roster name strings that often contain garbage tokens from scraped rosters.
    - Remove suffixes (JR/SR/II/III)
    - Remove "HS", dashes, stray punctuation
    - Remove trailing 'Season' (common Stanford layout bug)
    """
    name = norm_space(name)
    # common junk tokens
    name = re.sub(r"\b(Season)\b$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\b(Season)\b", " ", name, flags=re.IGNORECASE)

    # remove "HS" when it appears as a token
    name = re.sub(r"\bHS\b", " ", name, flags=re.IGNORECASE)

    # remove suffixes
    name = re.sub(r"\b(JR\.?|SR\.?|FR\.?|SO\.?|II|III|IV)\b", " ", name, flags=re.IGNORECASE)

    # normalize punctuation
    name = name.replace("–", " ").replace("—", " ").replace("-", " ")
    name = re.sub(r"[^A-Za-z\s']", " ", name)

    name = norm_space(name)
    return name


def _token_set(s: str) -> set:
    toks = [t.strip().lower() for t in re.split(r"[,\s]+", (s or "")) if t.strip()]
    toks = [t for t in toks if len(t) >= 2]
    return set(toks)


def _name_similarity(a: str, b: str) -> int:
    """
    Cheap similarity:
    - exact full match gets big points
    - last name match gets moderate points
    """
    a = norm_name(a).lower()
    b = norm_name(b).lower()
    if not a or not b:
        return 0
    if a == b:
        return 20
    a_parts = a.split()
    b_parts = b.split()
    if len(a_parts) >= 2 and len(b_parts) >= 2 and a_parts[-1] == b_parts[-1]:
        # last name match
        score = 8
        # first name initial/partial bonus
        if a_parts[0][0] == b_parts[0][0]:
            score += 2
        return score
    return 0


def _overlap_score(a: str, b: str) -> int:
    sa = _token_set(a)
    sb = _token_set(b)
    if not sa or not sb:
        return 0
    inter = sa.intersection(sb)
    return min(10, len(inter))


def _parse_time_to_seconds(t: str) -> Optional[float]:
    if not t:
        return None
    t = t.strip()
    m = re.fullmatch(r"(\d+):(\d{2})\.(\d{1,2})", t)
    if m:
        mins = int(m.group(1))
        secs = int(m.group(2))
        hund = int(m.group(3).ljust(2, "0"))
        return mins * 60.0 + secs + hund / 100.0
    m = re.fullmatch(r"(\d{1,2})\.(\d{1,2})", t)
    if m:
        secs = int(m.group(1))
        hund = int(m.group(2).ljust(2, "0"))
        return secs + hund / 100.0
    return None


def _best_time_string(times: List[str]) -> Optional[str]:
    best = None
    best_s = None
    for t in times:
        s = _parse_time_to_seconds(t)
        if s is None:
            continue
        if best_s is None or s < best_s:
            best_s = s
            best = t
    return best


def _normalize_times_url(url: str) -> str:
    if not url:
        return url
    u = url.split("#", 1)[0].split("?", 1)[0].rstrip("/")
    if "/swimmer/" in u and not u.endswith("/times"):
        u = u + "/times"
    return u.rstrip("/") + "/"


def extract_50free(html_txt: str) -> Tuple[str, str, str]:
    low = (html_txt or "").lower()
    if "captcha" in low or "access denied" in low:
        return "", "", "blocked_or_captcha"

    if BeautifulSoup is not None:
        soup = BeautifulSoup(html_txt, "html.parser")
        for node in soup.find_all(["tr", "li", "div"]):
            txt = norm_space(node.get_text(" ", strip=True))
            if not txt:
                continue
            if not any(k.lower() in txt.lower() for k in EVENT_KEYS):
                continue
            m = TIME_PAT.search(txt)
            if not m:
                continue
            t = m.group(1)
            cm = COURSE_PAT.search(txt)
            course = cm.group(1).upper() if cm else ""
            return t, course, "soup_row_match"

    # regex scan
    for key in EVENT_KEYS:
        k = key.lower()
        start = 0
        while True:
            idx = low.find(k, start)
            if idx == -1:
                break
            window = html_txt[max(0, idx - 500) : idx + 1500]
            m = TIME_PAT.search(window)
            if m:
                t = m.group(1)
                cm = COURSE_PAT.search(window)
                course = cm.group(1).upper() if cm else ""
                return t, course, "regex_window_match"
            start = idx + len(k)

    return "", "", "not_found"


def extract_class_of_year(html_txt: str) -> Optional[int]:
    if not html_txt:
        return None
    txt = html_txt
    for pat in CLASS_OF_PATTERNS:
        m = pat.search(txt)
        if m:
            try:
                y = int(m.group(1))
                if 1900 <= y <= 2100:
                    return y
            except Exception:
                pass
    return None


def _debug_write(debug_dir: Optional[Path], filename: str, content: str) -> None:
    if not debug_dir:
        return
    try:
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / filename).write_text(content or "", encoding="utf-8", errors="ignore")
    except Exception:
        pass


def _maybe_complete_swimcloud_onboarding(page, debug_dir: Optional[Path], zip_code: str, timeout_ms: int) -> bool:
    """
    SwimCloud sometimes redirects to onboarding:
      "Start your college search! Select your gender ..."
    We complete it once so cookies persist in the persistent context.
    """
    try:
        page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
    except Exception:
        pass

    html0 = ""
    try:
        html0 = page.content()
    except Exception:
        html0 = ""

    low = (html0 or "").lower()
    if "start your college search" not in low and "select your gender" not in low:
        return False

    _debug_write(debug_dir, "pw_onboarding_gate.html", html0)

    # Click Male (any choice is fine)
    try:
        page.locator("text=Male").first.click(timeout=timeout_ms)
    except Exception:
        pass

    # Fill ZIP (try a few selectors)
    filled = False
    for sel in [
        'input[placeholder*="zip" i]',
        'input[name*="zip" i]',
        'input[type="tel"]',
        'input[type="text"]',
        "input",
    ]:
        try:
            loc = page.locator(sel).first
            if loc.count() > 0:
                loc.fill(zip_code, timeout=timeout_ms)
                filled = True
                break
        except Exception:
            continue

    if filled:
        try:
            page.keyboard.press("Enter")
        except Exception:
            pass

    # Click any reasonable continue button if present
    for label in ["Continue", "Next", "Submit", "Start", "Go"]:
        try:
            page.get_by_role("button", name=re.compile(label, re.I)).click(timeout=1500)
            break
        except Exception:
            continue

    # Let it transition
    try:
        page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
    except Exception:
        pass

    return True


def ensure_search_page_ready(page, debug_dir: Optional[Path], zip_code: str, timeout_ms: int) -> None:
    """
    Navigate to /search/ and ensure we're not stuck on onboarding.
    Also ensure #global-search exists.
    """
    page.set_default_timeout(timeout_ms)
    page.set_default_navigation_timeout(timeout_ms)

    page.goto("https://www.swimcloud.com/search/", wait_until="domcontentloaded")

    # If onboarding gate, complete it and land back on /search/
    _maybe_complete_swimcloud_onboarding(page, debug_dir, zip_code=zip_code, timeout_ms=timeout_ms)

    # Now ensure global-search root exists (it is server-side in the HTML)
    try:
        page.wait_for_selector("#global-search", timeout=timeout_ms)
    except Exception:
        _debug_write(debug_dir, "pw_search_no_global_search.html", page.content())
        raise RuntimeError("Search shell not present (#global-search missing).")


def swimcloud_search_candidates(page, name_query: str, debug_dir: Optional[Path], timeout_ms: int) -> List[Dict[str, Any]]:
    """
    Go to /search/?q=<query> and parse candidates from #global-search[data-global-search-items].
    This is the dropdown data.
    """
    q = norm_space(name_query)
    url = f"https://www.swimcloud.com/search/?q={quote_plus(q)}"

    page.goto(url, wait_until="domcontentloaded")

    # Wait until the data attribute is populated (or at least present).
    # The element exists server-side, but data-global-search-items can be empty briefly.
    def has_items() -> bool:
        try:
            val = page.locator("#global-search").get_attribute("data-global-search-items") or ""
            val = val.strip()
            return val.startswith("[") and len(val) > 5
        except Exception:
            return False

    ok = False
    t0 = time.time()
    while time.time() - t0 < (timeout_ms / 1000.0):
        if has_items():
            ok = True
            break
        try:
            page.wait_for_timeout(250)
        except Exception:
            pass

    html_now = page.content()
    if not ok:
        _debug_write(debug_dir, "pw_search_not_hydrated.html", html_now)
        return []

    raw = page.locator("#global-search").get_attribute("data-global-search-items") or ""
    raw = raw.strip()

    # attribute is HTML-escaped JSON
    try:
        items_json = html_lib.unescape(raw)
        items = json.loads(items_json)
        if isinstance(items, list):
            return [x for x in items if isinstance(x, dict)]
    except Exception:
        _debug_write(debug_dir, "pw_search_bad_items_json.html", html_now)

    return []


def score_candidate(item: Dict[str, Any], roster_name: str, roster_hometown: str, roster_hs: str) -> int:
    """
    Score SwimCloud candidate dict.
    item fields typically include: name, url, team (HS), location, source
    """
    cand_name = norm_name(str(item.get("name", "") or ""))
    cand_hs = norm_space(str(item.get("team", "") or ""))
    cand_loc = norm_space(str(item.get("location", "") or ""))

    score = 0
    score += _name_similarity(roster_name, cand_name)

    # HS is the strongest signal when present
    if roster_hs:
        score += 4 * _overlap_score(roster_hs, cand_hs)
        # bonus if substring match (handles “Lamar High School”)
        if roster_hs.lower() in cand_hs.lower() or cand_hs.lower() in roster_hs.lower():
            score += 20

    # hometown/location signal
    if roster_hometown:
        score += 2 * _overlap_score(roster_hometown, cand_loc)

    # favor swimmers source explicitly
    if str(item.get("source", "")).lower() == "swimmers":
        score += 5

    return score


def fetch_page_playwright(page, url: str, timeout_ms: int, debug_dir: Optional[Path], debug_name: str = "") -> str:
    page.set_default_timeout(timeout_ms)
    page.set_default_navigation_timeout(timeout_ms)

    page.goto(url, wait_until="domcontentloaded")

    # Small scroll to trigger lazy load
    try:
        page.mouse.wheel(0, 1800)
        page.wait_for_timeout(350)
    except Exception:
        pass

    try:
        return page.content()
    except Exception:
        html_now = ""
        try:
            html_now = page.content()
        except Exception:
            pass
        if debug_name:
            _debug_write(debug_dir, f"{debug_name}.html", html_now)
        return ""


def find_best_50free_for_row(
    page,
    roster_name: str,
    roster_hometown: str,
    roster_hs: str,
    debug_dir: Optional[Path],
    timeout_ms: int,
    max_candidates: int,
    min_grad_year: int,
) -> EnrichResult:
    roster_name_clean = norm_name(roster_name)
    roster_hometown_clean = norm_space(roster_hometown)
    roster_hs_clean = norm_space(roster_hs)

    if not roster_name_clean or len(roster_name_clean.split()) < 2:
        return EnrichResult(error="missing_or_bad_name")

    # IMPORTANT: only search by NAME (not “HS”, not dashes, not extra tokens)
    items = swimcloud_search_candidates(page, roster_name_clean, debug_dir, timeout_ms=timeout_ms)
    if not items:
        return EnrichResult(error="no_search_items", note="search_items_empty")

    swimmers = [it for it in items if str(it.get("source", "")).lower() == "swimmers" and "/swimmer/" in str(it.get("url", ""))]
    if not swimmers:
        return EnrichResult(error="no_swimmer_candidates", note="no_swimmers_in_items")

    # Score + sort
    scored: List[Tuple[int, Dict[str, Any]]] = []
    for it in swimmers:
        s = score_candidate(it, roster_name_clean, roster_hometown_clean, roster_hs_clean)
        scored.append((s, it))
    scored.sort(key=lambda x: x[0], reverse=True)

    # Try top N candidates; stop on first with a 50 free time + (optional) grad year filter
    tried = 0
    for s, it in scored[: max_candidates if max_candidates > 0 else 8]:
        tried += 1
        href = str(it.get("url", "") or "")
        if not href:
            continue
        if href.startswith("/"):
            href = "https://www.swimcloud.com" + href
        times_url = _normalize_times_url(href)

        html_times = fetch_page_playwright(page, times_url, timeout_ms=timeout_ms, debug_dir=debug_dir, debug_name="")
        if not html_times:
            continue

        # grad year sanity check (only if the page exposes it)
        class_of = extract_class_of_year(html_times)
        if class_of is not None and min_grad_year > 0 and class_of < min_grad_year:
            continue

        t, course, parse_note = extract_50free(html_times)
        if t:
            matched_hs = norm_space(str(it.get("team", "") or ""))
            note = f"ok;score={s};cand={it.get('name','')};loc={it.get('location','')};parse={parse_note};tried={tried}"
            return EnrichResult(
                url=times_url,
                time=t,
                course=course,
                note=note,
                matched_high_school=matched_hs,
            )

    # If we got here: candidates existed, but no times parsed
    # Save the top candidate times page for debugging
    best = scored[0][1]
    best_href = str(best.get("url", "") or "")
    if best_href.startswith("/"):
        best_href = "https://www.swimcloud.com" + best_href
    best_times_url = _normalize_times_url(best_href)
    html_best = fetch_page_playwright(page, best_times_url, timeout_ms=timeout_ms, debug_dir=debug_dir, debug_name="pw_best_candidate_times")
    _debug_write(debug_dir, "pw_best_candidate_times.html", html_best)

    return EnrichResult(
        url=best_times_url,
        error="no_50free_found",
        note=f"candidates={len(scored)} tried={min(len(scored), max_candidates)}",
        matched_high_school=norm_space(str(best.get("team", "") or "")),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.6)
    ap.add_argument("--timeout", type=float, default=25.0)
    ap.add_argument("--playwright", action="store_true")
    ap.add_argument("--pw-headful", action="store_true")
    ap.add_argument(
        "--pw-user-data",
        dest="pw_user_data",
        default=str(Path.home() / ".pw_profile_swimcloud"),
        help="Persistent user-data dir for Playwright context (cookies/cache).",
    )
    ap.add_argument("--pw-user-agent", dest="pw_user_agent", default=UA)
    ap.add_argument("--debug-html", dest="debug_html", default="")
    ap.add_argument("--xlsx", dest="xlsx_path", default="")
    ap.add_argument("--zip", dest="zip_code", default="77005")
    ap.add_argument("--max-candidates", type=int, default=8)
    ap.add_argument("--min-grad-year", type=int, default=0)
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    debug_dir = Path(args.debug_html) if args.debug_html else None
    xlsx_path = Path(args.xlsx_path) if args.xlsx_path else None

    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    df = pd.read_csv(in_path)

    # standardize name column
    if "name" not in df.columns and "player_name" in df.columns:
        df["name"] = df["player_name"]

    # ensure output columns exist
    for c in [
        "swimcloud_url",
        "swimcloud_50free_time",
        "swimcloud_50free_course",
        "swimcloud_50free_note",
        "swimcloud_matched_high_school",
    ]:
        if c not in df.columns:
            df[c] = ""

    # Try to locate hometown/high_school columns (your roster CSV varies by team)
    def pick_col(cands: List[str]) -> str:
        for c in cands:
            if c in df.columns:
                return c
        return ""

    col_hometown = pick_col(["hometown", "home_town", "location"])
    col_hs = pick_col(["high_school", "hs", "previous_school", "school_prev", "prep_school"])

    # limit handling
    if args.limit and args.limit > 0:
        df_work = df.head(args.limit).copy()
        df_rest = df.iloc[args.limit:].copy()
    else:
        df_work = df.copy()
        df_rest = None

    errors: List[Dict[str, str]] = []

    # Non-playwright mode is intentionally not supported for this UI anymore
    if not args.playwright:
        raise SystemExit("Use --playwright for SwimCloud (dropdown search UI).")

    if not PW_AVAILABLE:
        raise SystemExit(
            "Playwright not available. Install:\n"
            "  pip install playwright\n"
            "  python -m playwright install chromium\n"
        )

    pw_ctx = None
    pw = None
    browser = None
    page = None

    timeout_ms = int(float(args.timeout) * 1000)

    try:
        pw_ctx = sync_playwright()
        pw = pw_ctx.start()
        browser = pw.chromium.launch_persistent_context(
            args.pw_user_data,
            headless=not args.pw_headful,
            user_agent=args.pw_user_agent,
            viewport={"width": 1280, "height": 720},
            locale="en-US",
        )
        page = browser.pages[0] if browser.pages else browser.new_page()

        # Ensure search page works and onboarding is completed once
        ensure_search_page_ready(page, debug_dir, zip_code=str(args.zip_code), timeout_ms=timeout_ms)

        for i, row in df_work.iterrows():
            rdict = {k: ("" if pd.isna(v) else str(v)) for k, v in row.to_dict().items()}

            name_raw = rdict.get("name", "") or rdict.get("player_name", "")
            hometown_raw = rdict.get(col_hometown, "") if col_hometown else ""
            hs_raw = rdict.get(col_hs, "") if col_hs else ""

            res = find_best_50free_for_row(
                page=page,
                roster_name=name_raw,
                roster_hometown=hometown_raw,
                roster_hs=hs_raw,
                debug_dir=debug_dir,
                timeout_ms=timeout_ms,
                max_candidates=int(args.max_candidates),
                min_grad_year=int(args.min_grad_year),
            )

            if res.url:
                df_work.at[i, "swimcloud_url"] = res.url
            if res.time:
                df_work.at[i, "swimcloud_50free_time"] = res.time
            if res.course:
                df_work.at[i, "swimcloud_50free_course"] = res.course
            if res.note:
                df_work.at[i, "swimcloud_50free_note"] = res.note
            if res.matched_high_school:
                df_work.at[i, "swimcloud_matched_high_school"] = res.matched_high_school

            if res.error:
                errors.append(
                    {
                        "row_index": str(i),
                        "name": name_raw,
                        "hometown": hometown_raw,
                        "high_school": hs_raw,
                        "error": res.error,
                        "note": res.note,
                        "url": res.url,
                    }
                )

            time.sleep(max(0.0, float(args.sleep)))

    finally:
        if browser is not None:
            try:
                browser.close()
            except Exception:
                pass
        if pw_ctx is not None and pw is not None:
            try:
                pw_ctx.stop()
            except Exception:
                pass

    df_out = pd.concat([df_work, df_rest], ignore_index=False) if df_rest is not None else df_work
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)

    if xlsx_path:
        try:
            df_out.to_excel(xlsx_path, index=False)
        except Exception as e:
            print(f"[WARN] XLSX export failed: {e}", file=sys.stderr)

    if errors:
        err_path = out_path.with_name(out_path.stem + "_errors.csv")
        with open(err_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(errors[0].keys()))
            w.writeheader()
            w.writerows(errors)
        print(f"[WARN] {len(errors)} rows missing 50 Free. See: {err_path}")
    else:
        print("[OK] All rows enriched.")

    print(f"[DONE] Wrote: {out_path}")


if __name__ == "__main__":
    main()
