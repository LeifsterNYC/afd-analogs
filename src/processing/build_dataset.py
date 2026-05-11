import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import config
from src.processing.parse_afds import load_afds
from src.processing.parse_obs import load_obs
from src.processing.parse_metar import load_metar_obs


def build(out_path=None):
    if out_path is None:
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = config.PROCESSED_DIR / "paired_dataset.json"

    print("Loading AFDs...")
    afds = load_afds()
    print(f"  {len(afds)} AFDs loaded")

    print("Loading LCD observations...")
    lcd = load_obs()
    print(f"  {len(lcd)} days of LCD obs loaded")

    metar = None
    if config.RAW_METAR_DIR.exists() and any(config.RAW_METAR_DIR.glob("*.csv")):
        print("Loading METAR observations...")
        metar = load_metar_obs()
        print(f"  {len(metar)} days of METAR obs loaded")
    else:
        print("No METAR data found, using LCD only")

    # KRME station and LCD snow data only exist from 2007 onward. Earlier records would have
    # structurally incomplete feature vectors, biasing the triplet miner toward false matches.
    MIN_DATE = "2007-01-01"

    records = []
    for date, text in afds.items():
        if date < MIN_DATE:
            continue
        if date not in lcd.index:
            continue

        obs = {}

        # snow amounts come from LCD
        lcd_row = lcd.loc[date]
        for col in ("snow", "snow_depth"):
            val = lcd_row.get(col)
            if val is not None and val == val:  # NaN != NaN
                obs[col] = val

        # per-station METAR obs
        if metar is not None and date in metar.index:
            obs.update(metar.loc[date].dropna().to_dict())
        else:
            # no METAR available, fall back to LCD for the basic fields
            lcd_fallback = {"tmax", "tmin", "tmean", "precip", "wind", "thunderstorm"}
            obs.update({k: v for k, v in lcd_row.dropna().items() if k in lcd_fallback})

        records.append({
            "date": date,
            "text": text,
            "obs": obs,
        })

    print(f"  {len(records)} paired records after join")

    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Saved to {out_path}")
    return records


if __name__ == "__main__":
    build()
