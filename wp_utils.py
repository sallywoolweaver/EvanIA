from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def norm_name(s: str) -> str:
    s = norm_space(s)
    s = re.sub(r"\s*\(.*?\)\s*", " ", s)
    s = re.sub(r"[^A-Za-z\s\.\-']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clean_height(s: str) -> str:
    s = norm_space(s)
    m = re.search(r"(\d)\s*['’\-]\s*(\d{1,2})", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    m = re.search(r"(\d)\s*-\s*(\d{1,2})", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return s

def is_http_url(u: str) -> bool:
    try:
        return urlparse(u).scheme in ("http", "https")
    except Exception:
        return False

def fetch(url: str, *, timeout: int = 30, ssl_verify: bool = True, sleep_s: float = 0.0) -> str:
    if sleep_s > 0:
        time.sleep(sleep_s)
    r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout, verify=ssl_verify)
    r.raise_for_status()
    return r.text

def absolute(base_url: str, href: str) -> str:
    return urljoin(base_url, href)

def guess_sidearm_profile_links(html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    links = set()
    for a in soup.select('a[href]'):
        href = a.get("href") or ""
        if "/roster/" in href and "?" not in href:
            u = absolute(base_url, href)
            if is_http_url(u):
                links.add(u)
    return sorted(links)

def extract_kv_from_dl(soup: BeautifulSoup) -> Dict[str, str]:
    kv: Dict[str, str] = {}
    for dl in soup.select("dl"):
        dts = dl.select("dt")
        dds = dl.select("dd")
        if not dts or not dds or len(dts) != len(dds):
            continue
        for dt, dd in zip(dts, dds):
            k = norm_space(dt.get_text(" ", strip=True)).rstrip(":").lower()
            v = norm_space(dd.get_text(" ", strip=True))
            if k and v and k not in kv:
                kv[k] = v
    return kv

def extract_kv_from_sidearm_bio(soup: BeautifulSoup) -> Dict[str, str]:
    kv: Dict[str, str] = {}
    for li in soup.select(".sidearm-roster-player-details li"):
        t = norm_space(li.get_text(" ", strip=True))
        m = re.match(r"^(.*?):\s*(.+)$", t)
        if m:
            k = norm_space(m.group(1)).lower()
            v = norm_space(m.group(2))
            if k and v and k not in kv:
                kv[k] = v
    return kv

def find_stats_tables(html: str) -> List[Tuple[str, List[str], List[List[str]]]]:
    soup = BeautifulSoup(html, "lxml")
    tables = []
    for t in soup.select("table"):
        headers = [norm_space(th.get_text(" ", strip=True)) for th in t.select("thead th")]
        if not headers:
            headers = [norm_space(th.get_text(" ", strip=True)) for th in t.select("tr th")]
        rows: List[List[str]] = []
        for tr in t.select("tbody tr"):
            cells = [norm_space(td.get_text(" ", strip=True)) for td in tr.select("td")]
            if cells:
                rows.append(cells)
        if headers and rows:
            caption_el = t.select_one("caption")
            caption = norm_space(caption_el.get_text(" ", strip=True)) if caption_el else ""
            tables.append((caption, headers, rows))
    return tables

def looks_like_player_stats(headers: List[str]) -> bool:
    h = [x.lower() for x in headers]
    return ("gp" in h) and ("g" in h)

def looks_like_goalie_stats(headers: List[str]) -> bool:
    h = [x.lower() for x in headers]
    return any(k in h for k in ["ga", "saves", "sv", "gmin", "mins", "goals against"])
