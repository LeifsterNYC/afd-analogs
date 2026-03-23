from pathlib import Path

# target forecast office
WFO = "BGM"

# NOAA LCD station IDs
STATIONS = ["KBGM"]

# date range
START_YEAR = 2003
END_YEAR = 2025

# paths
DATA_DIR = Path("data")
RAW_AFD_DIR = DATA_DIR / "raw" / "afds"
RAW_OBS_DIR = DATA_DIR / "raw" / "obs"
PROCESSED_DIR = DATA_DIR / "processed"
