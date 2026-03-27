import sys
import os
import time
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import config

BASE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

# fields we need for the similarity metric
FIELDS = ["tmpf", "sknt", "gust", "p01i", "wxcodes"]


def fetch_station_year(station, year):
    """Download one year of hourly METAR obs for a station. Returns CSV text."""
    iem_id = station.lstrip("K")

    params = {
        "station": iem_id,
        "year1": year, "month1": 1, "day1": 1,
        "year2": year, "month2": 12, "day2": 31,
        "tz": "UTC",
        "format": "comma",
        "latlon": "no",
        "direct": "no",
        "report_type": 3,  # routine METARs + SPECIs
    }
    data_params = "&".join(f"data={f}" for f in FIELDS)
    base_params = "&".join(f"{k}={v}" for k, v in params.items() if k != "data")
    url = f"{BASE_URL}?{base_params}&{data_params}"

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.text


def scrape():
    config.RAW_METAR_DIR.mkdir(parents=True, exist_ok=True)

    for station in config.METAR_STATIONS:
        for year in range(config.START_YEAR, config.END_YEAR + 1):
            filepath = config.RAW_METAR_DIR / f"{station}_{year}.csv"
            if filepath.exists():
                print(f"{station} {year} — already downloaded, skipping")
                continue

            try:
                text = fetch_station_year(station, year)
                filepath.write_text(text)
                lines = text.count("\n") - 5  # subtract header lines
                print(f"{station} {year} — {lines} observations saved")
            except requests.RequestException as e:
                print(f"{station} {year} — error: {e}")

            time.sleep(0.5)


if __name__ == "__main__":
    scrape()
