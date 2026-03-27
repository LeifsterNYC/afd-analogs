import json
import random
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import config
from src.training.similarity import obs_to_vec, MAX_DIST


def load_records(path=None):
    if path is None:
        path = config.PROCESSED_DIR / "paired_dataset.json"
    with open(path) as f:
        return json.load(f)


def build_triplets(records, seed=42):
    """
    Build (anchor, positive, negative) text triplets using full pairwise
    weather similarity.
    """
    rng = random.Random(seed)
    n = len(records)

    # vectorized pairwise similarity over all obs at once
    vecs = np.stack([obs_to_vec(r["obs"]) for r in records])
    diff = vecs[:, None, :] - vecs[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    sim = 1.0 - dist / MAX_DIST

    # mask diagonal so a record isn't its own positive
    np.fill_diagonal(sim, -1.0)

    triplets = []
    for i in range(n):
        pos_idx = int(np.argmax(sim[i]))
        neg_idx = int(np.argmin(sim[i]))
        triplets.append(
            (
                records[i]["text"],
                records[pos_idx]["text"],
                records[neg_idx]["text"],
            )
        )

    rng.shuffle(triplets)
    return triplets
