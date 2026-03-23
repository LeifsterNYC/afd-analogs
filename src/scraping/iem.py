import sys
import os
import time
import requests
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import config

BASE_URL = "https://mesonet.agron.iastate.edu/api/1/nws/afos/list.json"
TEXT_URL = "https://mesonet.agron.iastate.edu/api/1/nwstext"
WORKERS = 6


def fetch_afds_for_date(d):
    """Grab all AFD entries for a given date from IEM"""
    pil = f"AFD{config.WFO}"
    resp = requests.get(BASE_URL, params={"pil": pil, "date": d.isoformat()})
    resp.raise_for_status()
    return resp.json()["data"]


def fetch_and_save(entry, d):
    ts = entry["entered"]
    hhmm = ts[11:13] + ts[14:16]
    filepath = config.RAW_AFD_DIR / f"{d.isoformat()}_{hhmm}.txt"

    if filepath.exists():
        return 0

    resp = requests.get(f"{TEXT_URL}/{entry['product_id']}")
    resp.raise_for_status()
    filepath.write_text(resp.text)
    return 1


def scrape():
    start_date = date(config.START_YEAR, 1, 1)
    end_date = date(config.END_YEAR, 12, 31)

    config.RAW_AFD_DIR.mkdir(parents=True, exist_ok=True)

    done_dates = {
        date.fromisoformat(f.name[:10])
        for f in config.RAW_AFD_DIR.iterdir()
        if f.suffix == ".txt"
    }

    d = start_date
    total_saved = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        while d <= end_date:
            if d in done_dates:
                d += timedelta(days=1)
                continue

            try:
                entries = fetch_afds_for_date(d)
            except requests.RequestException as e:
                print(f"  error on {d}: {e}")
                d += timedelta(days=1)
                time.sleep(2)
                continue

            futures = {pool.submit(fetch_and_save, e, d): e for e in entries}
            saved = 0
            for fut in as_completed(futures):
                try:
                    saved += fut.result()
                except requests.RequestException as e:
                    print(f"  error fetching {futures[fut]['product_id']}: {e}")

            total_saved += saved
            print(f"{d} — {len(entries)} AFDs, {saved} new ({total_saved} total)")
            d += timedelta(days=1)
            time.sleep(0.2)

    print(f"Done. {total_saved} new AFDs saved to {config.RAW_AFD_DIR}")


if __name__ == "__main__":
    scrape()
