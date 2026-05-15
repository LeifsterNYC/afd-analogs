import argparse
import json
import random
import sys
import os
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import config
from src.retrieval.index import embed, build_index, retrieve
from src.retrieval.evaluate import load_splits

from sentence_transformers import SentenceTransformer


def fmt_obs(obs):
    keys = sorted(obs.keys())
    return ", ".join(f"{k}={obs[k]}" for k in keys if obs[k] is not None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="jinaai/jina-embeddings-v2-base-en")
    parser.add_argument("--n", type=int, default=20, help="number of queries to sample")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    corpus, test_records = load_splits()
    random.seed(args.seed)
    sample_queries = random.sample(test_records, min(args.n, len(test_records)))

    print(f"loading model: {args.model}")
    model = SentenceTransformer(args.model, trust_remote_code=True)
    model.max_seq_length = 2048

    print("embedding corpus...")
    corpus_embs = embed(model, [r["text"] for r in corpus], batch_size=args.batch_size)
    idx = build_index(corpus_embs)

    print("embedding queries...")
    query_embs = embed(model, [r["text"] for r in sample_queries], batch_size=args.batch_size)
    _, ids = retrieve(idx, query_embs, args.k)

    out = args.out or (config.DATA_DIR / "qualitative_samples.txt")
    with open(out, "w", encoding="utf-8") as f:
        for i, q in enumerate(sample_queries):
            f.write(f"{'='*80}\nQUERY {i+1} - {q['date']}\n")
            f.write(f"actual weather: {fmt_obs(q['obs'])}\n\n")
            f.write(q["text"][:2000])
            f.write("\n\n--- TOP MATCHES ---\n")
            for rank, j in enumerate(ids[i]):
                m = corpus[int(j)]
                f.write(f"\n[match {rank+1}] {m['date']}\n")
                f.write(f"actual weather: {fmt_obs(m['obs'])}\n")
                f.write(m["text"][:1500])
                f.write("\n")
            f.write("\n\nRater questions:\n")
            f.write("  1. Are these retrieved AFDs reasonable meteorological analogs? (1-5)\n")
            f.write("  2. Do the actual weather outcomes look similar?\n\n")

    print(f"wrote {len(sample_queries)} samples to {out}")


if __name__ == "__main__":
    main()
