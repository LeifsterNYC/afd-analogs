import argparse
import json
import sys
import os
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import config
from src.training.similarity import weather_similarity
from src.retrieval.index import embed, build_index, retrieve

from sentence_transformers import SentenceTransformer


def load_splits():
    with open(config.PROCESSED_DIR / "paired_dataset.json") as f:
        all_records = json.load(f)
    with open(config.PROCESSED_DIR / "test_split.json") as f:
        test_records = json.load(f)

    test_dates = {r["date"] for r in test_records}
    corpus = [r for r in all_records if r["date"] not in test_dates]
    return corpus, test_records


def score_retrievals(test_records, corpus, sims, ids):
    """Mean weather similarity between each query's own obs and its top-K matches."""
    per_query = []
    for i, query in enumerate(test_records):
        match_sims = [
            weather_similarity(query["obs"], corpus[int(j)]["obs"])
            for j in ids[i]
            if int(j) >= 0
        ]
        if match_sims:
            per_query.append(float(np.mean(match_sims)))
    return float(np.mean(per_query)), per_query


def random_baseline(test_records, corpus, k, seed=0):
    rng = np.random.default_rng(seed)
    n = len(corpus)
    per_query = []
    for query in test_records:
        picks = rng.choice(n, size=k, replace=False)
        match_sims = [weather_similarity(query["obs"], corpus[int(j)]["obs"]) for j in picks]
        per_query.append(float(np.mean(match_sims)))
    return float(np.mean(per_query)), per_query


def evaluate_model(model, corpus, test_records, k, batch_size, label):
    print(f"[{label}] embedding corpus ({len(corpus)} records)...")
    corpus_embs = embed(model, [r["text"] for r in corpus], batch_size=batch_size)
    idx = build_index(corpus_embs)

    print(f"[{label}] embedding test queries ({len(test_records)} records)...")
    query_embs = embed(model, [r["text"] for r in test_records], batch_size=batch_size)

    sims, ids = retrieve(idx, query_embs, k)
    mean, per_query = score_retrievals(test_records, corpus, sims, ids)
    print(f"[{label}] mean top-{k} weather similarity: {mean:.4f}")
    return mean, per_query, ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--zeroshot-model", default="jinaai/jina-embeddings-v2-base-en")
    parser.add_argument("--finetuned", default=None,
                        help="path to a fine-tuned SentenceTransformer checkpoint")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    corpus, test_records = load_splits()
    print(f"corpus: {len(corpus)}  test: {len(test_records)}")

    results = {}

    rand_mean, _ = random_baseline(test_records, corpus, args.k)
    results["random"] = rand_mean
    print(f"[random] mean top-{args.k} weather similarity: {rand_mean:.4f}")

    print(f"loading zero-shot model: {args.zeroshot_model}")
    zs_model = SentenceTransformer(args.zeroshot_model, trust_remote_code=True)
    zs_model.max_seq_length = 2048
    zs_mean, _, _ = evaluate_model(zs_model, corpus, test_records, args.k, args.batch_size, "zero-shot")
    results["zero_shot"] = zs_mean
    del zs_model

    if args.finetuned:
        print(f"loading fine-tuned model: {args.finetuned}")
        ft_model = SentenceTransformer(args.finetuned, trust_remote_code=True)
        ft_model.max_seq_length = 2048
        ft_mean, _, _ = evaluate_model(ft_model, corpus, test_records, args.k, args.batch_size, "fine-tuned")
        results["fine_tuned"] = ft_mean
        results["lift_over_zero_shot"] = ft_mean - zs_mean

    print("\n=== summary ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    out = args.out or (config.DATA_DIR / "eval_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
