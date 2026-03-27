import pandas as pd
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import config

# LCD column names -> short names
RENAME = {
    "DailyMaximumDryBulbTemperature": "tmax",
    "DailyMinimumDryBulbTemperature": "tmin",
    "DailyMeanDryBulbTemperature": "tmean",
    "DailyPrecipitation": "precip",
    "DailySnowfall": "snow",
    "DailySnowDepth": "snow_depth",
    "DailyAverageWindSpeed": "wind",
}


def load_obs(obs_dir=None):
    """
    Read all obs and return a DataFrame indexed by date.
    """
    if obs_dir is None:
        obs_dir = config.RAW_OBS_DIR
    obs_dir = Path(obs_dir)

    frames = []
    for csv_file in sorted(obs_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            daily = df[df["REPORT_TYPE"].str.strip() == "SOD"].copy()
            if not daily.empty:
                frames.append(daily)
        except Exception as e:
            print(f"  skipping {csv_file.name}: {e}")

    if not frames:
        return pd.DataFrame()

    obs = pd.concat(frames, ignore_index=True)
    obs["date"] = pd.to_datetime(obs["DATE"]).dt.date.astype(str)

    available = {k: v for k, v in RENAME.items() if k in obs.columns}
    extra = ["DailyWeather"] if "DailyWeather" in obs.columns else []
    obs = obs[["date"] + list(available.keys()) + extra].rename(columns=available)

    # treat T as 0
    for col in available.values():
        obs[col] = obs[col].replace("T", 0.0)
        obs[col] = pd.to_numeric(obs[col], errors="coerce")

    # thunderstorm flag
    if "DailyWeather" in obs.columns:
        obs["thunderstorm"] = obs["DailyWeather"].fillna("").str.contains(r"\bTS\b").astype(float)
        obs = obs.drop(columns=["DailyWeather"])

    obs = obs.groupby("date").first()
    return obs
