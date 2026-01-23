#!/usr/bin/env python3
"""enrich_swimcloud_50free.py

Enrich a roster CSV with a SwimCloud *best* 50 Freestyle time.

Why this version:
- SwimCloud's own search is frequently JS-heavy and can fail in headless scraping.
- This script discovers SwimCloud swimmer profile URLs via DuckDuckGo HTML results, then
  scrapes the swimmer page for a 50 Free best time with tolerant parsing.

Usage:
  python enrich_swimcloud_50free.py --in waterpolo_rosters.csv --out waterpolo_rosters_with_50free.csv
  python enrich_swimcloud_50free.py --in waterpolo_rosters.csv --out out.csv --sleep 1.2 --limit 200 --debug-html debug_html/

Outputs:
  - <out>.csv enriched with:
      swimcloud_url
      swimcloud_50free_time
      swimcloud_50free_course
      swimcloud_50free_note
  - <out>_errors.csv if any rows fail

Notes:
- Keep --sleep >= 1.0 for large runs.
- This is heuristic; spot-check a sample.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import html as html_lib
import re
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None  # type: ignore

# Optional: Playwright (more reliable when SwimCloud search is JS-heavy or blocks requests).
PW_AVAILABLE = False
try:
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError  # type: ignore

    PW_AVAILABLE = True
except Exception:
    PW_AVAILABLE = False

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
)
DDG_HTML = "https://duckduckgo.com/html/"

# SwimCloud search entry points (site sometimes uses different routes/params).
# We try these before falling back to an external search engine.
SWIMCLOUD_SEARCH_CANDIDATES = [
    "https://www.swimcloud.com/search/?q={q}",
    "https://www.swimcloud.com/search/?q={q}&type=swimmers",
    "https://www.swimcloud.com/search/?q={q}&c=swimmers",
    "https://www.swimcloud.com/search/?q={q}&category=swimmers",
]


def http_get(
    session: requests.Session,
    url: str,
    *,
    timeout: int = 20,
    verify: bool = True,
    params: Optional[Dict[str, str]] = None,
    headers: Optional[Dict[str, str]] = None,
    allow_redirects: bool = True,
) -> requests.Response:
    """GET with consistent headers and robust SSL handling.

    In some school/enterprise networks, TLS interception causes certificate/hostname
    mismatches. We default to verify=True, but auto-retry once with verify=False
    if we hit an SSL error. You can force behavior with CLI flags.
    """
    base_headers = {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"}
    if headers:
        base_headers.update(headers)
    try:
        return session.get(
            url,
            headers=base_headers,
            params=params,
            timeout=timeout,
            verify=verify,
            allow_redirects=allow_redirects,
        )
    except requests.exceptions.SSLError:
        # Retry once without verification; downstream code will log any failure.
        return session.get(
            url,
            headers=base_headers,
            params=params,
            timeout=timeout,
            verify=False,
            allow_redirects=allow_redirects,
        )


def _extract_swimmer_links(html: str) -> List[str]:
    """Extract SwimCloud swimmer profile links from an HTML page."""
    links = []
    # Common pattern: /swimmer/<id>/
    for m in re.finditer(r"https?://www\.swimcloud\.com/swimmer/\d+/?", html):
        links.append(m.group(0).rstrip("/") + "/")
    # Relative URLs
    for m in re.finditer(r"href=\"(/swimmer/\d+/? )\"", html):
        rel = m.group(1).strip()
        links.append("https://www.swimcloud.com" + rel.rstrip("/") + "/")
    # Deduplicate while preserving order
    seen = set()
    out = []
    for u in links:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def swimcloud_search(session: requests.Session, query: str) -> Optional[str]:
    """Try to resolve a SwimCloud swimmer profile URL using SwimCloud's own search."""
    q = requests.utils.quote(query)
    for tmpl in SWIMCLOUD_SEARCH_CANDIDATES:
        url = tmpl.format(q=q)
        r = http_get(session, url)
        if r.status_code != 200:
            continue
        links = _extract_swimmer_links(r.text)
        if links:
            return links[0]
    return None

TIME_PAT = re.compile(r"\b(\d{1,2}:\d{2}\.\d{2}|\d{1,2}\.\d{2})\b")
COURSE_PAT = re.compile(r"\b(SCY|SCM|LCM)\b", re.IGNORECASE)

EVENT_KEYS = ["50 Free", "50 Freestyle", "50 Yard Freestyle", "50 Y Free", "50 FR"]


@dataclasses.dataclass
class EnrichResult:
    url: str = ""
    time: str = ""
    course: str = ""
    note: str = ""
    error: str = ""


def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def norm_name(name: str) -> str:
    name = norm_space(name)
    name = re.sub(r"\b(JR\.?|SR\.?|II|III|IV)\b", "", name, flags=re.IGNORECASE)
    name = re.sub(r"[^A-Za-z\-\s']", " ", name)
    return norm_space(name)


def ddg_search_swimcloud(name: str, hometown_hint: str, session: requests.Session, timeout: float) -> Tuple[str, str]:
    name = norm_name(name)
    hometown_hint = norm_space(hometown_hint)
    if hometown_hint:
        q = f'site:swimcloud.com/swimmer "{name}" "{hometown_hint}"'
    else:
        q = f'site:swimcloud.com/swimmer "{name}"'

    r = http_get(
        session,
        DDG_HTML,
        params={"q": q},
        headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"},
        timeout=timeout,
    )
    if r is None:
        return "", "ddg_no_response"

    if r.status_code != 200:
        return "", f"ddg_status_{r.status_code}"

    txt = r.text
    urls: List[str] = []

    if BeautifulSoup is not None:
        soup = BeautifulSoup(txt, "html.parser")
        for a in soup.select("a.result__a[href]"):
            urls.append(a.get("href") or "")
    else:
        urls = re.findall(r'href="([^"]+)"', txt)

    def decode_ddg(href: str) -> str:
        href = html_lib.unescape(href)
        if "duckduckgo.com/l/?" in href and "uddg=" in href:
            parsed = urllib.parse.urlparse(href)
            qs = urllib.parse.parse_qs(parsed.query)
            if qs.get("uddg"):
                return urllib.parse.unquote(qs["uddg"][0])
        return href

    cleaned: List[str] = []
    for u in urls:
        u = decode_ddg(u)
        if "swimcloud.com" not in u or "/swimmer/" not in u:
            continue
        u = u.split("#")[0]
        if re.search(r"swimcloud\.com/swimmer/\d+/?", u) or u.startswith("https://www.swimcloud.com/swimmer/"):
            cleaned.append(u)

    # de-dupe
    out: List[str] = []
    seen = set()
    for u in cleaned:
        if u in seen:
            continue
        seen.add(u)
        out.append(u)

    if out:
        return out[0], f"ddg:{q}"

    # fallback without hometown
    if hometown_hint:
        return ddg_search_swimcloud(name, "", session, timeout)

    return "", "no_swimcloud_result"


def fetch_page(url: str, session: requests.Session, timeout: float) -> Tuple[str, str]:
    r = http_get(
        session,
        url,
        headers={
            "User-Agent": UA,
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Referer": "https://www.google.com/",
            "DNT": "1",
        },
        timeout=timeout,
    )
    if r is None:
        return "", "fetch_error:no_response"
    return r.text or "", f"status:{r.status_code}"


def find_swimcloud_profile(name: str, hometown: str, high_school: str, session: requests.Session, timeout: float) -> Tuple[str, str]:
    """Return (swimmer_url, note). Tries SwimCloud's own search first, then DDG fallback."""
    # Try SwimCloud native search with progressively broader queries.
    queries: List[str] = []
    if high_school:
        queries.append(f"{name} {high_school}")
    if hometown:
        queries.append(f"{name} {hometown}")
    queries.append(name)

    for q in queries:
        try:
            url = swimcloud_search(q, session, timeout)
            if url:
                return url, "ok_swimcloud_search"
        except Exception:
            # fall through to next query
            pass

    # Fallback: general web search (often blocked/rate-limited)
    try:
        url, note = ddg_search_swimcloud(name, hometown, session, timeout)
        if url:
            return url, note
        return "", note
    except Exception as e:
        return "", f"search_failed:{type(e).__name__}"


def _score_search_hit(text: str, hometown: str, high_school: str) -> int:
    """Heuristic to choose best swimmer result."""
    t = (text or "").lower()
    score = 0
    if hometown and hometown.lower() in t:
        score += 3
    # Token-based partial matches for "City, ST" etc.
    for tok in re.split(r"[,\s]+", (hometown or "").lower()):
        tok = tok.strip()
        if tok and len(tok) >= 3 and tok in t:
            score += 1
    if high_school and high_school.lower() in t:
        score += 5
    return score


def find_swimcloud_profile_playwright(
    name: str,
    hometown: str,
    high_school: str,
    page,
    timeout_s: int = 25,
) -> Tuple[str, str]:
    """Playwright-based SwimCloud search (more reliable than request/DDG).

    Uses https://www.swimcloud.com/search/?q=... and picks a likely swimmer profile.
    """
    if not BeautifulSoup:
        return "", "bs4_missing"

    # Build progressively broader queries.
    queries: List[str] = []
    if high_school:
        queries.append(f"{name} {high_school}")
    if hometown:
        queries.append(f"{name} {hometown}")
    queries.append(name)

    base = "https://www.swimcloud.com"
    for q in queries:
        try:
            url = f"{base}/search/?q={requests.utils.quote(q)}"
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_s * 1000)
            # Give it a moment to render client-side results.
            page.wait_for_timeout(800)
            html = page.content()
            soup = BeautifulSoup(html, "html.parser")

            # SwimCloud search results typically include /swimmer/<id>/ links.
            hits = []
            for a in soup.select('a[href^="/swimmer/"]'):
                href = a.get("href") or ""
                if not re.search(r"^/swimmer/\d+/?", href):
                    continue
                card_text = " ".join(a.stripped_strings)
                # Expand: include parent text if present.
                parent = a.parent
                if parent:
                    card_text = (parent.get_text(" ", strip=True) or "")
                hits.append((href, card_text))

            if not hits:
                # Some layouts hide links in nested cards.
                for a in soup.select('a[href*="/swimmer/"]'):
                    href = a.get("href") or ""
                    m = re.search(r"(/swimmer/\d+/?)(?:$|\?)", href)
                    if not m:
                        continue
                    href = m.group(1)
                    parent = a.parent
                    card_text = (parent.get_text(" ", strip=True) if parent else "") or ""
                    hits.append((href, card_text))

            if not hits:
                continue

            # Choose best match by heuristic.
            scored = []
            for href, txt in hits:
                scored.append((_score_search_hit(txt, hometown, high_school), href, txt))
            scored.sort(key=lambda x: x[0], reverse=True)
            best_score, best_href, best_txt = scored[0]
            full = best_href if best_href.startswith("http") else base + best_href
            return full, f"ok_playwright_search:score={best_score}"
        except PWTimeoutError:
            continue
        except Exception:
            continue

    return "", "no_swimcloud_result"


def fetch_page_playwright(url: str, page, timeout_s: int = 25) -> Tuple[str, str]:
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_s * 1000)
        page.wait_for_timeout(800)
        return page.content(), "ok_playwright_fetch"
    except PWTimeoutError:
        return "", "fetch_timeout"
    except Exception as e:
        return "", f"fetch_failed:{type(e).__name__}"

def extract_50free(html_txt: str) -> Tuple[str, str, str]:
    low = html_txt.lower()
    if "captcha" in low or "access denied" in low:
        return "", "", "blocked_or_captcha"

    # Try structured parse
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

    # Fallback: regex window scan
    for key in EVENT_KEYS:
        k = key.lower()
        start = 0
        while True:
            idx = low.find(k, start)
            if idx == -1:
                break
            window = html_txt[max(0, idx - 400) : idx + 1200]
            m = TIME_PAT.search(window)
            if m:
                t = m.group(1)
                cm = COURSE_PAT.search(window)
                course = cm.group(1).upper() if cm else ""
                return t, course, "regex_window_match"
            start = idx + len(k)

    return "", "", "not_found"


def enrich_one(
    row: Dict[str, str],
    session: requests.Session,
    timeout: float,
    debug_dir: Optional[Path],
    *,
    use_playwright: bool = False,
    pw_page=None,
) -> EnrichResult:
    name = norm_name(row.get("name", "") or row.get("player_name", "") or "")
    hometown = norm_space(row.get("hometown", "") or row.get("home_town", "") or "")
    high_school = norm_space(row.get("high_school", "") or row.get("hs", "") or row.get("previous_school", "") or row.get("school", "") or "")
    url = norm_space(row.get("swimcloud_url", "") or "")

    if not name or len(name.split()) < 2:
        return EnrichResult(error="missing_or_bad_name")

    if not url:
        if use_playwright:
            if not PW_AVAILABLE:
                return EnrichResult(error="playwright_not_installed")
            if pw_page is None:
                return EnrichResult(error="playwright_page_missing")
            url, note = find_swimcloud_profile_playwright(name, hometown, high_school, pw_page, timeout_s=int(timeout))
        else:
            url, note = find_swimcloud_profile(name, hometown, high_school, session, timeout)
        if not url:
            return EnrichResult(error=note)
    else:
        note = "provided_url"

    if use_playwright:
        html_txt, fetch_note = fetch_page_playwright(url, pw_page, timeout=timeout)
    else:
        html_txt, fetch_note = fetch_page(url, session, timeout)
    if not html_txt:
        return EnrichResult(url=url, note=f"{note};{fetch_note}", error="empty_html")

    t, course, parse_note = extract_50free(html_txt)

    if debug_dir and not t:
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
            safe = re.sub(r"[^A-Za-z0-9_\-]+", "_", name)[:80]
            (debug_dir / f"{safe}.html").write_text(html_txt, encoding="utf-8", errors="ignore")
        except Exception:
            pass

    if not t:
        return EnrichResult(url=url, note=f"{note};{fetch_note};{parse_note}", error=parse_note)

    return EnrichResult(url=url, time=t, course=course, note=f"{note};{fetch_note};{parse_note}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=1.2)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument(
        "--playwright",
        action="store_true",
        help="Use Playwright for SwimCloud search/page fetch (recommended if HTTP search finds no results)",
    )
    ap.add_argument(
        "--pw-headful",
        action="store_true",
        help="Run Playwright in headed mode (useful for debugging / login / captcha checks)",
    )
    ap.add_argument("--debug-html", dest="debug_html", default="")
    ap.add_argument("--xlsx", dest="xlsx_path", default="")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    debug_dir = Path(args.debug_html) if args.debug_html else None
    xlsx_path = Path(args.xlsx_path) if args.xlsx_path else None

    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    df = pd.read_csv(in_path)
    if "name" not in df.columns and "player_name" in df.columns:
        df["name"] = df["player_name"]

    for c in ["swimcloud_url", "swimcloud_50free_time", "swimcloud_50free_course", "swimcloud_50free_note"]:
        if c not in df.columns:
            df[c] = ""

    if args.limit and args.limit > 0:
        df_work = df.head(args.limit).copy()
        df_rest = df.iloc[args.limit:].copy()
    else:
        df_work = df.copy()
        df_rest = None

    errors: List[Dict[str, str]] = []

    with requests.Session() as session:
        # Playwright is much more reliable for SwimCloud search/profile pages than plain HTTP.
        pw_ctx = None
        pw = None
        browser = None
        page = None

        try:
            if args.playwright:
                if not PW_AVAILABLE:
                    raise RuntimeError(
                        "Playwright is not available in this environment. Install with: pip install playwright\n"
                        "Then install browsers with: python -m playwright install chromium"
                    )
                pw_ctx = sync_playwright()
                pw = pw_ctx.start()
                browser = pw.chromium.launch(headless=not args.pw_headful)
                page = browser.new_page()

            for i, row in df_work.iterrows():
                rdict = {k: ("" if pd.isna(v) else str(v)) for k, v in row.to_dict().items()}
                res = enrich_one(
                    rdict,
                    session,
                    args.timeout,
                    debug_dir,
                    use_playwright=bool(args.playwright),
                    pw_page=page,
                )

                if res.url:
                    df_work.at[i, "swimcloud_url"] = res.url
                if res.time:
                    df_work.at[i, "swimcloud_50free_time"] = res.time
                if res.course:
                    df_work.at[i, "swimcloud_50free_course"] = res.course
                if res.note:
                    df_work.at[i, "swimcloud_50free_note"] = res.note

                if res.error:
                    errors.append(
                        {
                            "row_index": str(i),
                            "name": rdict.get("name", ""),
                            "hometown": rdict.get("hometown", ""),
                            "school": rdict.get("school", "") or rdict.get("college", ""),
                            "team": rdict.get("team", ""),
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