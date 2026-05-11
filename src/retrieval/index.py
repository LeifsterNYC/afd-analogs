import sys
import os
from pathlib import Path

import numpy as np
import faiss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import config


def embed(model, texts, batch_size=8):
    """Encode texts and L2-normalize for cosine-via-inner-product."""
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embs.astype(np.float32)


def build_index(embs):
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    return idx


def retrieve(idx, query_embs, k):
    sims, ids = idx.search(query_embs, k)
    return sims, ids


def save_index(idx, embs, records, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(idx, str(out_dir / "index.faiss"))
    np.save(out_dir / "embs.npy", embs)
    # keep dates so we can map FAISS ids back to records
    with open(out_dir / "dates.txt", "w") as f:
        f.write("\n".join(r["date"] for r in records))
