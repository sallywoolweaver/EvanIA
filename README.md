# NCAA Men's Water Polo roster + record + stats + SwimCloud 50 Free (HS) enrichment

This project does four things:

1) Scrape **college roster** pages (names + height + class year + hometown + position + profile URL).
2) Scrape **team 2025 record** (computed from the schedule page when possible).
3) Scrape **2025 player stats** (goals/steals/etc) from team stats pages (Sidearm-style sites).
4) Enrich players with **SwimCloud** best time for **50 Freestyle (SCY)** (typically HS history).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium
```

## Run: roster + record + stats

```bash
python scrape_rosters.py --teams teams.yaml --season 2025 --out waterpolo_rosters_2025.csv --xlsx waterpolo_rosters_2025.xlsx
```

Optional:
- `--use-playwright` forces Playwright for roster pages (slower, but helps JS-heavy sites)
- `--max-players 0` means no limit; otherwise set a cap for debugging

## Run: SwimCloud 50 Free enrichment

```bash
python enrich_swimcloud_50free.py --in waterpolo_rosters_2025.csv --out waterpolo_rosters_2025_with_50free.csv --xlsx waterpolo_rosters_2025_with_50free.xlsx
```

Notes:
- SwimCloud pages can be rate-limited. Use `--sleep 1.5` (or higher) and/or `--limit 200` while testing.
- Matching is heuristic (name + hometown). If a match is wrong, you can override by adding a `swimcloud_url` column in your CSV.
