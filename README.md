# NCAA Men's Water Polo Roster Scraper (curated D1/varsity list)

This project scrapes roster pages for a curated set of NCAA men's water polo programs
and outputs a CSV (and optionally an Excel .xlsx) containing:

- school
- player_name
- position
- height
- hometown
- class_year
- record (team W-L from NCAA RPI page, best-effort)
- source_url

## Quick start

```bash
cd waterpolo_d1_scraper
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scrape_rosters.py --out waterpolo_rosters.csv --xls waterpolo_rosters.xlsx
```

## If you see “nothing outputted”

This script **always** writes the output CSV you specify (even if empty) and will also
write an `_errors.csv` file next to it when sites block scraping or have layouts the
parser does not recognize.

Example:

```bash
python scrape_rosters.py --out waterpolo_rosters.csv
# outputs: waterpolo_rosters.csv and maybe waterpolo_rosters_errors.csv
```

## Notes

- Some athletics websites block automated requests. If that happens, the error will be
  captured in `*_errors.csv`.
- NCAA RPI pages occasionally change URLs; the default is currently:
  https://www.ncaa.com/rankings/waterpolo-men/nc/rpi
