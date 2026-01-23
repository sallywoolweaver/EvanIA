#!/usr/bin/env python3
"""
enrich_swimcloud_50free.py

Enrich a roster CSV with a SwimCloud best 50 Freestyle time (typically HS times).

Key behavior:
- SwimCloud "search" is a GLOBAL dropdown/autocomplete, not a traditional results page.
- This script uses Playwright to:
  1) pass the onboarding gate once (gender + zip) to set cookies
  2) use the global search combobox
  3) select the best candidate based on HS/hometown
  4) navigate to /times/ and parse the 50 Free best time

Outputs added/updated:
  swimcloud_url
  swimcloud_50free_time
  swimcloud_50free_course
  swimcloud_50free_note
  swimcloud_high_school   (filled from SwimCloud if missing/empty)
  swimcloud_grad_year     (if discoverable)
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Optional: Playwright
PW_AVAILABLE = False
try:
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError  # type: ignore
    PW_AVAILABLE = True
except Exception:
    PW_AVAILABLE = False

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None  # type: ignore


UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
)

SWIMCLOUD_HOME = "https://www.swimcloud.com/"
SWIMCLOUD_SEARCH = "https://www.swimcloud.com/search/"

EVENT_KEYS = [
    "50 Free",
    "50 Freestyle",
    "50 Yard Freestyle",
    "50 Y Free",
    "50 FR",
    "50 Fr",
]

TIME_PAT = re.compile(r"\b(\d{1,2}:\d{2}\.\d{2}|\d{1,2}\.\d{2})\b")
COURSE_PAT = re.compile(r"\b(SCY|SCM|LCM)\b", re.IGNORECASE)

# ---------- utilities ----------

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def clean_name(raw: str) -> str:
    """
    Fix roster garbage like:
      - "First Last Season"
      - "First Last - Season"
      - suffixes, class-year tags, punctuation
    """
    s = norm_space(raw)
    # remove common garbage tokens
    s = re.sub(r"\bSeason\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bRoster\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(Fr|So|Jr|Sr)\.?\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\((?:Fr|So|Jr|Sr)\.?\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(JR\.?|SR\.?|II|III|IV)\b", "", s, flags=re.IGNORECASE)
    # drop stray hyphen separators
    s = re.sub(r"\s*[-–—]\s*", " ", s)
    # keep letters, spaces, apostrophes, hyphens
    s = re.sub(r"[^A-Za-z\-\s']", " ", s)
    s = norm_space(s)
    return s


def clean_high_school(raw: str) -> str:
    s = norm_space(raw)
    # remove "HS" standalone (but keep "High School" phrase)
    s = re.sub(r"\bHS\b\.?", "", s, flags=re.IGNORECASE)
    # remove weird punctuation/dashes
    s = re.sub(r"\s*[-–—]\s*", " ", s)
    s = norm_space(s)
    return s


def tokenize_place(s: str) -> List[str]:
    s = (s or "").lower()
    toks = re.split(r"[^a-z0-9]+", s)
    toks = [t for t in toks if len(t) >= 3]
    return toks


def last_name(name: str) -> str:
    parts = norm_space(name).split()
    return parts[-1].lower() if parts else ""


def _normalize_times_url(url: str) -> str:
    if not url:
        return url
    u = url.split("#", 1)[0].split("?", 1)[0].rstrip("/")
    if "/swimmer/" in u:
        if not u.endswith("/times"):
            u = u + "/times"
        u = u + "/"
    return u


def _parse_time_to_seconds(t: str) -> Optional[float]:
    t = (t or "").strip()
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


@dataclasses.dataclass
class EnrichResult:
    url: str = ""
    time: str = ""
    course: str = ""
    note: str = ""
    swimcloud_high_school: str = ""
    swimcloud_grad_year: str = ""
    error: str = ""


# ---------- SwimCloud page parsing ----------

def parse_swimmer_header_info(html: str) -> Tuple[str, str, str]:
    """
    Attempt to parse:
      displayed_name, high_school, grad_year
    from the swimmer profile (or times) page.
    This is heuristic and resilient to minor layout changes.
    """
    displayed = ""
    hs = ""
    grad = ""

    if not html:
        return displayed, hs, grad

    low = html.lower()

    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "html.parser")

        # name: common patterns are h1 / .c-title / etc.
        h1 = soup.find("h1")
        if h1:
            displayed = norm_space(h1.get_text(" ", strip=True))

        # try any obvious "High School" label
        txt = soup.get_text("\n", strip=True)
        m = re.search(r"High School\s*[:\-]?\s*([^\n]+)", txt, flags=re.IGNORECASE)
        if m:
            hs = norm_space(m.group(1))

        # grad year
        m = re.search(r"\bClass of\s*(20\d{2})\b", txt, flags=re.IGNORECASE)
        if m:
            grad = m.group(1)

        # Some pages include "Graduation: 2024" or "Grad Year 2024"
        if not grad:
            m = re.search(r"\bGrad(?:uation)?\s*(?:Year)?\s*[:\-]?\s*(20\d{2})\b", txt, flags=re.IGNORECASE)
            if m:
                grad = m.group(1)

    else:
        # fallback regex-only
        m = re.search(r"<h1[^>]*>(.*?)</h1>", html, flags=re.IGNORECASE | re.DOTALL)
        if m:
            displayed = norm_space(re.sub(r"<[^>]+>", " ", m.group(1)))
        m = re.search(r"High School\s*[:\-]?\s*([^\n<]+)", html, flags=re.IGNORECASE)
        if m:
            hs = norm_space(m.group(1))
        m = re.search(r"\bClass of\s*(20\d{2})\b", html, flags=re.IGNORECASE)
        if m:
            grad = m.group(1)

    # sanitize
    displayed = clean_name(displayed) if displayed else displayed
    hs = clean_high_school(hs)
    return displayed, hs, grad


def extract_50free_times(html_txt: str) -> Tuple[str, str, str]:
    """
    Return (best_time, course, note)
    Stronger than the earlier version:
      - collects multiple candidate times from the same event row
      - tries to avoid relay splits by requiring the event label match
    """
    low = (html_txt or "").lower()
    if "captcha" in low or "access denied" in low:
        return "", "", "blocked_or_captcha"

    times_found: List[str] = []
    course = ""

    # Prefer structured parse
    if BeautifulSoup is not None and html_txt:
        soup = BeautifulSoup(html_txt, "html.parser")

        # Scan rows/blocks; match event label tightly
        for row in soup.find_all(["tr", "li", "div"]):
            txt = norm_space(row.get_text(" ", strip=True))
            if not txt:
                continue
            # Must contain an event keyword
            if not any(re.search(rf"\b{re.escape(k)}\b", txt, flags=re.IGNORECASE) for k in EVENT_KEYS):
                continue
            # Avoid obvious relay/split rows
            if re.search(r"\brelay\b|\bsplit\b", txt, flags=re.IGNORECASE):
                continue

            # Collect all time-like strings in this row
            row_times = TIME_PAT.findall(txt)
            if row_times:
                times_found.extend(row_times)
                cm = COURSE_PAT.search(txt)
                if cm:
                    course = cm.group(1).upper()
                # If we found times in a clearly-matching row, we can stop scanning
                # after gathering a decent set.
                if len(times_found) >= 3:
                    break

    # Regex fallback: look near the event label
    if not times_found and html_txt:
        for key in EVENT_KEYS:
            k = key.lower()
            start = 0
            while True:
                idx = low.find(k, start)
                if idx == -1:
                    break
                window = html_txt[max(0, idx - 500) : idx + 2000]
                if re.search(r"\brelay\b|\bsplit\b", window, flags=re.IGNORECASE):
                    start = idx + len(k)
                    continue
                window_times = TIME_PAT.findall(window)
                if window_times:
                    times_found.extend(window_times)
                    cm = COURSE_PAT.search(window)
                    if cm:
                        course = cm.group(1).upper()
                    break
                start = idx + len(k)

    best = _best_time_string(times_found)
    if not best:
        return "", "", "not_found"

    return best, course, "ok"


# ---------- Playwright SwimCloud search (dropdown) ----------

def debug_dump(page, debug_dir: Optional[Path], fname: str) -> None:
    if not debug_dir:
        return
    try:
        debug_dir.mkdir(parents=True, exist_ok=True)
        html = page.content()
        (debug_dir / fname).write_text(html, encoding="utf-8", errors="ignore")
    except Exception:
        pass


def maybe_complete_onboarding(page, debug_dir: Optional[Path], zip_code: str, timeout_ms: int) -> bool:
    """
    SwimCloud sometimes forces an onboarding gate (gender + zip) before search works.
    Fill it once; cookies in persistent context usually prevent repeat gates.
    """
    try:
        page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
    except Exception:
        pass

    html = ""
    try:
        html = page.content()
    except Exception:
        html = ""
    low = html.lower()

    if ("start your college search" not in low) and ("select your gender" not in low):
        return False

    debug_dump(page, debug_dir, "pw_onboarding_gate.html")

    # click Male (or any)
    clicked = False
    for sel in ["text=Male", "text=Female"]:
        try:
            page.locator(sel).first.click(timeout=2500)
            clicked = True
            break
        except Exception:
            continue

    # fill zip
    filled = False
    for sel in [
        'input[placeholder*="zip" i]',
        'input[name*="zip" i]',
        'input[type="text"]',
        'input[type="tel"]',
        'input',
    ]:
        try:
            loc = page.locator(sel).first
            if loc.count() > 0:
                loc.fill(zip_code, timeout=2500)
                filled = True
                break
        except Exception:
            continue

    if filled:
        try:
            page.keyboard.press("Enter")
        except Exception:
            pass

    # click a continue-ish button if present
    for btn_text in ["Continue", "Next", "Submit", "Start", "Go"]:
        try:
            page.get_by_role("button", name=re.compile(btn_text, re.I)).click(timeout=2500)
            break
        except Exception:
            continue

    # Do NOT wait for networkidle (can hang). Just allow a short settle.
    try:
        page.wait_for_timeout(1200)
    except Exception:
        pass

    return True


def locate_global_search_input(page):
    """
    SwimCloud search UI varies; find a visible combobox/search input.
    """
    selectors = [
        "#global-search-select input",
        "input[role='combobox']",
        "input[aria-autocomplete='list']",
        "input[type='search']",
        "header input",
    ]
    for sel in selectors:
        try:
            loc = page.locator(sel).filter(has_not=page.locator("[disabled]"))
            # pick first visible
            for i in range(min(loc.count(), 8)):
                el = loc.nth(i)
                if el.is_visible():
                    return el
        except Exception:
            continue
    return None


def ensure_search_ready(page, debug_dir: Optional[Path], zip_code: str, timeout_ms: int) -> None:
    """
    Navigate to SwimCloud search and ensure the global search input exists.
    Avoid networkidle waits (often hang).
    """
    # Navigate to /search/
    try:
        page.goto(SWIMCLOUD_SEARCH, wait_until="domcontentloaded", timeout=timeout_ms)
    except Exception:
        page.goto(SWIMCLOUD_HOME, wait_until="domcontentloaded", timeout=timeout_ms)

    # If gate appears, complete it and go back to /search/
    maybe_complete_onboarding(page, debug_dir, zip_code=zip_code, timeout_ms=timeout_ms)
    try:
        page.goto(SWIMCLOUD_SEARCH, wait_until="domcontentloaded", timeout=timeout_ms)
    except Exception:
        pass

    # Wait for search input to appear
    t0 = time.time()
    while time.time() - t0 < (timeout_ms / 1000.0):
        inp = locate_global_search_input(page)
        if inp is not None:
            return
        try:
            page.wait_for_timeout(500)
        except Exception:
            pass

    debug_dump(page, debug_dir, "pw_search_not_hydrated.html")
    raise RuntimeError("Search UI not hydrated (could not locate search combobox).")


def collect_dropdown_candidates(page) -> List[Dict[str, str]]:
    """
    Return list of candidates from the open dropdown.
    Attempts to extract swimmer links and visible text.
    """
    candidates: List[Dict[str, str]] = []

    # Common patterns: listbox/options and/or anchors in a popup
    option_selectors = [
        "[role='listbox'] [role='option']",
        "[role='option']",
        ".Select-menu-outer .Select-option",  # old react-select
        "div[id^='react-select-']",
    ]

    for sel in option_selectors:
        try:
            opts = page.locator(sel)
            n = min(opts.count(), 12)
            if n <= 0:
                continue
            for i in range(n):
                o = opts.nth(i)
                if not o.is_visible():
                    continue
                txt = ""
                try:
                    txt = norm_space(o.inner_text(timeout=1500))
                except Exception:
                    txt = ""

                href = ""
                # try to find an anchor inside
                try:
                    a = o.locator("a[href*='/swimmer/']").first
                    if a.count() > 0:
                        href = a.get_attribute("href") or ""
                except Exception:
                    href = ""

                candidates.append({"text": txt, "href": href, "idx": str(i), "sel": sel})
            if candidates:
                break
        except Exception:
            continue

    return candidates


def score_candidate(text: str, target_name: str, hometown: str, high_school: str) -> int:
    """
    Heuristic score using:
      - last name presence
      - high school tokens match (strong)
      - hometown tokens match (medium)
    """
    t = (text or "").lower()
    score = 0

    ln = last_name(target_name)
    if ln and ln in t:
        score += 4

    hs_tokens = tokenize_place(high_school)
    ht_tokens = tokenize_place(hometown)

    # HS match dominates
    for tok in hs_tokens:
        if tok in t:
            score += 3

    # Hometown match if HS missing/weak
    for tok in ht_tokens:
        if tok in t:
            score += 1

    return score


def names_compatible(target: str, displayed: str) -> bool:
    """
    Require last names match and first initial match if possible.
    """
    target = clean_name(target)
    displayed = clean_name(displayed)
    tparts = target.split()
    dparts = displayed.split()
    if len(tparts) < 2 or len(dparts) < 2:
        return False
    if tparts[-1].lower() != dparts[-1].lower():
        return False
    return tparts[0][0].lower() == dparts[0][0].lower()


def dropdown_search_best_swimmer_url(
    page,
    name: str,
    hometown: str,
    high_school: str,
    *,
    debug_dir: Optional[Path],
    zip_code: str,
    timeout_ms: int,
    min_grad_year: int,
) -> Tuple[str, str, str, str]:
    """
    Uses the dropdown autocomplete to select the best swimmer candidate.

    Returns:
      (times_url, note, swimcloud_hs, swimcloud_grad_year)
    """
    ensure_search_ready(page, debug_dir, zip_code=zip_code, timeout_ms=timeout_ms)

    inp = locate_global_search_input(page)
    if inp is None:
        debug_dump(page, debug_dir, "pw_no_search_input.html")
        return "", "no_search_input", "", ""

    # Build query: NAME only. (Do NOT include 'HS' or extra roster junk.)
    qname = clean_name(name)

    # Clear + type
    try:
        inp.click(timeout=1500)
        # hard clear
        inp.fill("", timeout=1500)
        page.keyboard.press("Control+A")
        page.keyboard.press("Backspace")
    except Exception:
        pass

    try:
        # small delay typing triggers dropdown reliably
        inp.type(qname, delay=40)
    except Exception:
        try:
            inp.fill(qname)
        except Exception:
            return "", "search_input_fill_failed", "", ""

    # Wait for dropdown options
    t0 = time.time()
    candidates: List[Dict[str, str]] = []
    while time.time() - t0 < (timeout_ms / 1000.0):
        candidates = collect_dropdown_candidates(page)
        if candidates:
            break
        try:
            page.wait_for_timeout(250)
        except Exception:
            pass

    if not candidates:
        debug_dump(page, debug_dir, "pw_dropdown_empty.html")
        return "", "no_dropdown_candidates", "", ""

    # Score candidates
    scored = []
    for c in candidates:
        txt = c.get("text", "")
        s = score_candidate(txt, qname, hometown, high_school)
        scored.append((s, c))
    scored.sort(key=lambda x: x[0], reverse=True)

    # Try top K candidates; verify on swimmer page
    tried = 0
    for s, c in scored[:6]:
        tried += 1

        # Click candidate option
        prev_url = ""
        try:
            prev_url = page.url
        except Exception:
            prev_url = ""

        try:
            sel = c.get("sel", "[role='option']")
            idx = int(c.get("idx", "0"))
            page.locator(sel).nth(idx).click(timeout=2500)
        except Exception:
            # fallback: press Enter (sometimes selects first)
            try:
                page.keyboard.press("Enter")
            except Exception:
                continue

        # Wait for navigation to swimmer
        ok_nav = False
        t1 = time.time()
        while time.time() - t1 < 8.0:
            try:
                u = page.url
            except Exception:
                u = ""
            if "/swimmer/" in u:
                ok_nav = True
                break
            try:
                page.wait_for_timeout(250)
            except Exception:
                pass

        if not ok_nav:
            # go back to search and try next
            try:
                page.goto(SWIMCLOUD_SEARCH, wait_until="domcontentloaded", timeout=timeout_ms)
            except Exception:
                pass
            continue

        # Load HTML and verify identity/grad year
        try:
            page.wait_for_timeout(600)
        except Exception:
            pass

        html = ""
        try:
            html = page.content()
        except Exception:
            html = ""

        displayed, sc_hs, sc_grad = parse_swimmer_header_info(html)

        # Name must be compatible (prevents wrong-person selection)
        if displayed and not names_compatible(qname, displayed):
            # try next candidate
            try:
                page.goto(SWIMCLOUD_SEARCH, wait_until="domcontentloaded", timeout=timeout_ms)
            except Exception:
                pass
            continue

        # grad year sanity
        if sc_grad:
            try:
                gy = int(sc_grad)
                if gy < min_grad_year:
                    try:
                        page.goto(SWIMCLOUD_SEARCH, wait_until="domcontentloaded", timeout=timeout_ms)
                    except Exception:
                        pass
                    continue
            except Exception:
                pass

        # Navigate to times page
        swimmer_url = page.url
        times_url = _normalize_times_url(swimmer_url)
        try:
            page.goto(times_url, wait_until="domcontentloaded", timeout=timeout_ms)
            page.wait_for_timeout(800)
        except Exception:
            pass

        note = f"ok;score={s};cand={displayed or c.get('text','')};tried={tried}"
        return times_url, note, sc_hs, sc_grad

    return "", f"no_verified_candidate;tried={min(len(scored),6)}", "", ""


def fetch_times_html(page, times_url: str, timeout_ms: int, debug_dir: Optional[Path], debug_name: str) -> str:
    """
    Navigate to times_url and return rendered HTML.
    Avoid networkidle waits.
    """
    try:
        page.goto(times_url, wait_until="domcontentloaded", timeout=timeout_ms)
    except Exception:
        pass

    # Wait for some hint of times content (best effort)
    for sel in ["text=50", "text=Freestyle", "table", ".c-table", "main"]:
        try:
            page.locator(sel).first.wait_for(timeout=2500)
            break
        except Exception:
            continue

    try:
        page.wait_for_timeout(800)
    except Exception:
        pass

    html = ""
    try:
        html = page.content()
    except Exception:
        html = ""

    if debug_dir and html:
        safe = re.sub(r"[^A-Za-z0-9_\-]+", "_", debug_name)[:80]
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
            (debug_dir / f"{safe}__times.html").write_text(html, encoding="utf-8", errors="ignore")
        except Exception:
            pass

    return html


# ---------- main enrichment ----------

def enrich_row_playwright(
    row: Dict[str, str],
    page,
    *,
    timeout_s: float,
    debug_dir: Optional[Path],
    zip_code: str,
    min_grad_year: int,
) -> EnrichResult:
    name = clean_name(row.get("name", "") or row.get("player_name", "") or "")
    hometown = norm_space(row.get("hometown", "") or row.get("home_town", "") or "")
    high_school = clean_high_school(
        row.get("high_school", "") or row.get("hs", "") or row.get("previous_school", "") or row.get("school", "") or ""
    )

    if not name or len(name.split()) < 2:
        return EnrichResult(error="missing_or_bad_name")

    timeout_ms = int(float(timeout_s) * 1000)

    times_url, note, sc_hs, sc_grad = dropdown_search_best_swimmer_url(
        page,
        name=name,
        hometown=hometown,
        high_school=high_school,
        debug_dir=debug_dir,
        zip_code=zip_code,
        timeout_ms=timeout_ms,
        min_grad_year=min_grad_year,
    )
    if not times_url:
        return EnrichResult(error="no_swimmer_url", note=note)

    html = fetch_times_html(page, times_url, timeout_ms, debug_dir, debug_name=name)

    if not html:
        return EnrichResult(url=times_url, error="empty_html", note=note, swimcloud_high_school=sc_hs, swimcloud_grad_year=sc_grad)

    best, course, parse_note = extract_50free_times(html)
    if not best:
        return EnrichResult(url=times_url, error=parse_note, note=f"{note};parse={parse_note}", swimcloud_high_school=sc_hs, swimcloud_grad_year=sc_grad)

    return EnrichResult(
        url=times_url,
        time=best,
        course=course,
        note=f"{note};parse={parse_note}",
        swimcloud_high_school=sc_hs,
        swimcloud_grad_year=sc_grad,
    )


def safe_checkpoint_write(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.6)
    ap.add_argument("--timeout", type=float, default=25.0)

    ap.add_argument("--playwright", action="store_true", help="Use Playwright (required for dropdown search).")
    ap.add_argument("--pw-headful", action="store_true", help="Run Playwright headed (debug).")
    ap.add_argument(
        "--pw-user-data",
        dest="pw_user_data",
        default=str(Path.home() / ".pw_profile_swimcloud"),
        help="Persistent profile directory (cookies/session).",
    )
    ap.add_argument("--debug-html", dest="debug_html", default="")
    ap.add_argument("--zip", dest="zip_code", default="77005")
    ap.add_argument("--min-grad-year", dest="min_grad_year", type=int, default=2019)

    ap.add_argument("--checkpoint-every", dest="checkpoint_every", type=int, default=50)

    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    debug_dir = Path(args.debug_html) if args.debug_html else None

    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    df = pd.read_csv(in_path)

    if "name" not in df.columns and "player_name" in df.columns:
        df["name"] = df["player_name"]

    # Ensure output columns
    for c in [
        "swimcloud_url",
        "swimcloud_50free_time",
        "swimcloud_50free_course",
        "swimcloud_50free_note",
        "swimcloud_high_school",
        "swimcloud_grad_year",
    ]:
        if c not in df.columns:
            df[c] = ""

    # Work subset
    if args.limit and args.limit > 0:
        work_idx = df.index[: args.limit]
    else:
        work_idx = df.index

    errors: List[Dict[str, str]] = []

    if not args.playwright:
        raise SystemExit("This version requires --playwright because SwimCloud uses a dropdown autocomplete UI.")

    if not PW_AVAILABLE:
        raise SystemExit(
            "Playwright is not available. Install with:\n"
            "  pip install playwright\n"
            "  python -m playwright install chromium\n"
        )

    pw_ctx = None
    pw = None
    browser = None
    page = None

    try:
        pw_ctx = sync_playwright()
        pw = pw_ctx.start()
        browser = pw.chromium.launch_persistent_context(
            args.pw_user_data,
            headless=not args.pw_headful,
            user_agent=UA,
            viewport={"width": 1280, "height": 720},
            locale="en-US",
        )
        page = browser.pages[0] if browser.pages else browser.new_page()

        n_done = 0
        for idx in work_idx:
            row = df.loc[idx].to_dict()
            rdict = {k: ("" if pd.isna(v) else str(v)) for k, v in row.items()}

            res = enrich_row_playwright(
                rdict,
                page,
                timeout_s=float(args.timeout),
                debug_dir=debug_dir,
                zip_code=args.zip_code,
                min_grad_year=int(args.min_grad_year),
            )

            if res.url:
                df.at[idx, "swimcloud_url"] = res.url
            if res.time:
                df.at[idx, "swimcloud_50free_time"] = res.time
            if res.course:
                df.at[idx, "swimcloud_50free_course"] = res.course
            if res.note:
                df.at[idx, "swimcloud_50free_note"] = res.note
            if res.swimcloud_high_school:
                df.at[idx, "swimcloud_high_school"] = res.swimcloud_high_school
                # If the roster HS field is empty, also backfill it
                if not norm_space(str(df.at[idx, "high_school"] if "high_school" in df.columns else "")):
                    if "high_school" in df.columns:
                        df.at[idx, "high_school"] = res.swimcloud_high_school
            if res.swimcloud_grad_year:
                df.at[idx, "swimcloud_grad_year"] = res.swimcloud_grad_year

            if res.error:
                errors.append(
                    {
                        "row_index": str(idx),
                        "name": rdict.get("name", ""),
                        "hometown": rdict.get("hometown", ""),
                        "high_school": rdict.get("high_school", ""),
                        "school": rdict.get("school", "") or rdict.get("college", ""),
                        "team": rdict.get("team", ""),
                        "error": res.error,
                        "note": res.note,
                        "url": res.url,
                    }
                )

            n_done += 1

            # checkpoint write
            if args.checkpoint_every and (n_done % int(args.checkpoint_every) == 0):
                safe_checkpoint_write(df, out_path)

            time.sleep(max(0.0, float(args.sleep)))

    finally:
        # Final write
        safe_checkpoint_write(df, out_path)

        if errors:
            err_path = out_path.with_name(out_path.stem + "_errors.csv")
            with open(err_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(errors[0].keys()))
                w.writeheader()
                w.writerows(errors)
            print(f"[WARN] {len(errors)} rows missing 50 Free. See: {err_path}", file=sys.stderr)
        else:
            print("[OK] All rows enriched.", file=sys.stderr)

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

    print(f"[DONE] Wrote: {out_path}")


if __name__ == "__main__":
    main()
