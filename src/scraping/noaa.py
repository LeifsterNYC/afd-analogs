import sys
import os
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import config

BASE_URL = "https://www.ncei.noaa.gov/data/local-climatological-data/access"
STATION_ID = "72515004725"  # KBGM


def fetch_year(year):
    filepath = config.RAW_OBS_DIR / f"{year}.csv"
    if filepath.exists():
        print(f"{year} — already downloaded, skipping")
        return

    url = f"{BASE_URL}/{year}/{STATION_ID}.csv"
    resp = requests.get(url)
    resp.raise_for_status()
    filepath.write_bytes(resp.content)
    print(f"{year} — saved to {filepath}")


def fetch_all():
    config.RAW_OBS_DIR.mkdir(parents=True, exist_ok=True)

    for year in range(config.START_YEAR, config.END_YEAR + 1):
        try:
            fetch_year(year)
        except requests.RequestException as e:
            print(f"{year} — error: {e}")


if __name__ == "__main__":
    fetch_all()
