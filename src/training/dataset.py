import json
import random
import sys
import os

import numpy as np
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import config
from src.training.similarity import obs_to_vec, MAX_DIST


def load_records(path=None):
    if path is None:
        path = config.PROCESSED_DIR / "paired_dataset.json"
    with open(path) as f:
        return json.load(f)


def build_triplets(records, seed=42):
    # for each anchor pick the most-similar and least-similar AFD by weather vector
    rng = random.Random(seed)
    n = len(records)

    vecs = np.stack([obs_to_vec(r["obs"]) for r in records])
    sim = 1.0 - cdist(vecs, vecs) / MAX_DIST

    sim_pos = sim.copy()
    sim_neg = sim.copy()
    np.fill_diagonal(sim_pos, -np.inf)
    np.fill_diagonal(sim_neg, +np.inf)

    pos_idxs = sim_pos.argmax(axis=1)
    neg_idxs = sim_neg.argmin(axis=1)

    triplets = [
        (records[i]["text"], records[pos_idxs[i]]["text"], records[neg_idxs[i]]["text"])
        for i in range(n)
    ]

    rng.shuffle(triplets)
    return triplets
