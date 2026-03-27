import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import config

# climatological ranges for the BGM CWA stations (F / in / mph)
_CONT = {
    "tmax":     ((-15, 98),  False), 
    "tmin":     ((-25, 80),  False),
    "wind":     ((0,  30.0), True),
    "max_gust": ((0,  60.0), True),
    "precip":   ((0,   3.5), True),
}
_BIN = {
    "thunderstorm": 0.5,
    "fzra":         0.4,
}
# CWA-wide features from LCD
_CWA_CONT = {
    "snow":       ((0, 20.0), True),
    "snow_depth": ((0, 30.0), True),
}

# build ordered feature lists at import time
# order: all per-station continuous, then per-station binary, then CWA-wide
_CONT_FEATURES = []  # list of (key, lo, hi, zero_impute)
_BIN_FEATURES  = []  # list of (key, weight)

for _station in config.METAR_STATIONS:
    for _feat, (_rng, _zero) in _CONT.items():
        _CONT_FEATURES.append((f"{_station}_{_feat}", _rng[0], _rng[1], _zero))
    for _feat, _w in _BIN.items():
        _BIN_FEATURES.append((f"{_station}_{_feat}", _w))

for _feat, (_rng, _zero) in _CWA_CONT.items():
    _CONT_FEATURES.append((_feat, _rng[0], _rng[1], _zero))

# precompute max possible distance for normalization
MAX_DIST = float(np.sqrt(
    len(_CONT_FEATURES) + sum(w ** 2 for _, w in _BIN_FEATURES)
))


def obs_to_vec(obs_dict):
    vec = []

    for key, lo, hi, zero_impute in _CONT_FEATURES:
        val = obs_dict.get(key)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            vec.append(0.0 if zero_impute else 0.5)
        else:
            vec.append(float(np.clip((val - lo) / (hi - lo), 0.0, 1.0)))

    for key, weight in _BIN_FEATURES:
        val = obs_dict.get(key)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = 0.0
        vec.append(float(val) * weight)

    return np.array(vec, dtype=np.float32)


def weather_similarity(obs_a, obs_b):
    """Normalized similarity in [0, 1] where 1 = identical conditions."""
    dist = float(np.linalg.norm(obs_to_vec(obs_a) - obs_to_vec(obs_b)))
    return 1.0 - dist / MAX_DIST
