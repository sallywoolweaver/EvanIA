# NCAA Men's Water Polo Roster Scraper (Heights / Hometowns / Class→Grad Year / Team Record)

This project scrapes **official roster pages** for NCAA men’s water polo programs and outputs a single CSV with:
- `school`
- `record` (team record pulled from the NCAA RPI list)
- `player`
- `height_raw` (as shown on roster)
- `height_in` (numeric inches when parseable)
- `hometown`
- `class_raw` (Fr/So/Jr/Sr/Gr etc, as shown)
- `grad_year_est` (estimated graduation year)

## Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scrape_rosters.py --out ncaa_mwp_rosters.csv --grad_base_year 2026
```

### Notes on “graduation year”
Most rosters list **class / academic year** (Fr/So/Jr/Sr/Gr), not an explicit graduation year.
This scraper estimates graduation year from `--grad_base_year` (default: current year).
Typical mapping:
- Fr → base + 3
- So → base + 2
- Jr → base + 1
- Sr → base
- Gr / 5th → base (best-effort)

If you want “class year” only, keep `class_raw` and ignore `grad_year_est`.

## Updating / adding teams
Edit `teams.yaml`:
- `school`: display name in output
- `roster_url`: official athletics roster page URL
- `ncaa_name`: name as it appears on NCAA RPI page (for record matching)

Then rerun the script.

## Troubleshooting
- Some athletics sites block aggressive scraping. Use `--sleep 1.0` (or higher).
- If a roster is “card view” only, this script still usually works (it tries multiple parsers), but you may need to provide an alternate roster URL (table layout) in `teams.yaml`.
