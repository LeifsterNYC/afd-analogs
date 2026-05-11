import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import config

KT_TO_MPH = 1.15078


def _load_station(station, metar_dir):
    frames = []
    for f in sorted(metar_dir.glob(f"{station}_*.csv")):
        try:
            df = pd.read_csv(f, comment="#", low_memory=False)
            frames.append(df)
        except Exception as e:
            print(f"  skipping {f.name}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _to_numeric(series):
    return pd.to_numeric(series.replace("M", np.nan).replace("0.0001", 0.0), errors="coerce")


def _daily_for_station(df):
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["date"] = pd.to_datetime(df["valid"], utc=True).dt.date.astype(str)
    df["tmpf"]    = _to_numeric(df["tmpf"])
    df["sknt"]    = _to_numeric(df["sknt"])
    df["gust"]    = _to_numeric(df["gust"])
    df["p01i"]    = _to_numeric(df["p01i"])
    df["wxcodes"] = df["wxcodes"].fillna("").replace("M", "")

    daily = df.groupby("date").agg(
        tmax         = ("tmpf",    "max"),
        tmin         = ("tmpf",    "min"),
        wind_kts     = ("sknt",    "mean"),
        max_gust_kts = ("gust",    "max"),
        precip       = ("p01i",    "sum"),
        wx_combined  = ("wxcodes", lambda x: " ".join(x.dropna())),
    )

    daily["wind"]        = daily["wind_kts"] * KT_TO_MPH
    daily["max_gust"]    = daily["max_gust_kts"] * KT_TO_MPH
    # KAVP 2003-2007 has dozens of 195-200 kt gust readings (impossible). Drop anything above
    # a realistic regional ceiling so the similarity vector doesn't get poisoned by sensor errors.
    daily.loc[daily["max_gust"] > 100, "max_gust"] = np.nan
    # TS appears in TS, +TS, -TS, TSRA, TSGR, TSSN, VCTS, etc. — any substring "TS" is a thunderstorm signal.
    daily["thunderstorm"] = daily["wx_combined"].str.contains("TS").astype(float)
    daily["fzra"]        = daily["wx_combined"].str.contains("FZRA").astype(float)

    return daily.drop(columns=["wind_kts", "max_gust_kts", "wx_combined"])


def load_metar_obs(metar_dir=None):
    """
    Load METAR data for all configured stations, returning a DataFrame indexed
    by date with per-station columns
    """
    if metar_dir is None:
        metar_dir = config.RAW_METAR_DIR
    metar_dir = Path(metar_dir)

    result = None
    for station in config.METAR_STATIONS:
        raw = _load_station(station, metar_dir)
        if raw.empty:
            print(f"  no data for {station}")
            continue

        daily = _daily_for_station(raw)
        daily.columns = [f"{station}_{col}" for col in daily.columns]
        print(f"  {station}: {len(daily)} days")

        result = daily if result is None else result.join(daily, how="outer")

    return result if result is not None else pd.DataFrame()
