import argparse
import json
import hashlib
import sys
import os
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import config
from src.training.similarity import weather_similarity
from src.retrieval.index import embed, build_index, retrieve

from sentence_transformers import SentenceTransformer


KS = [1, 3, 5, 10]
RANDOM_SEEDS = 20


def load_splits():
    with open(config.PROCESSED_DIR / "paired_dataset.json") as f:
        all_records = json.load(f)
    with open(config.PROCESSED_DIR / "test_split.json") as f:
        test_records = json.load(f)

    test_dates = {r["date"] for r in test_records}
    corpus = [r for r in all_records if r["date"] not in test_dates]
    return corpus, test_records


def per_query_topk_sim(test_records, corpus, ids, k):
    per_query = []
    for i, query in enumerate(test_records):
        match_sims = [
            weather_similarity(query["obs"], corpus[int(j)]["obs"])
            for j in ids[i][:k]
            if int(j) >= 0
        ]
        per_query.append(float(np.mean(match_sims)) if match_sims else 0.0)
    return np.array(per_query)


def random_topk_sim(test_records, corpus, k, seed):
    rng = np.random.default_rng(seed)
    n = len(corpus)
    per_query = []
    for query in test_records:
        picks = rng.choice(n, size=k, replace=False)
        sims = [weather_similarity(query["obs"], corpus[int(j)]["obs"]) for j in picks]
        per_query.append(float(np.mean(sims)))
    return np.array(per_query)


def summarize(per_query):
    return {
        "mean": float(per_query.mean()),
        "std": float(per_query.std(ddof=1)),
        "median": float(np.median(per_query)),
        "n": int(per_query.size),
        "ci95_half": float(1.96 * per_query.std(ddof=1) / np.sqrt(per_query.size)),
    }


def cache_path(model_name, role, n_records):
    h = hashlib.md5(f"{model_name}|{role}|{n_records}".encode()).hexdigest()[:10]
    return config.PROCESSED_DIR / "emb_cache" / f"{role}_{h}.npy"


def cached_embed(model, texts, role, model_name, batch_size):
    p = cache_path(model_name, role, len(texts))
    if p.exists():
        print(f"  cache hit: {p.name}")
        return np.load(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    embs = embed(model, texts, batch_size=batch_size)
    np.save(p, embs)
    return embs


def evaluate_model(model, model_name, corpus, test_records, batch_size, label):
    print(f"[{label}] embedding corpus ({len(corpus)} records)...")
    corpus_embs = cached_embed(model, [r["text"] for r in corpus], f"{label}_corpus", model_name, batch_size)
    idx = build_index(corpus_embs)

    print(f"[{label}] embedding queries ({len(test_records)} records)...")
    query_embs = cached_embed(model, [r["text"] for r in test_records], f"{label}_test", model_name, batch_size)

    _, ids = retrieve(idx, query_embs, max(KS))

    per_k = {}
    for k in KS:
        per_query = per_query_topk_sim(test_records, corpus, ids, k)
        per_k[k] = {"per_query": per_query.tolist(), **summarize(per_query)}
        s = per_k[k]
        print(f"[{label}] k={k}: mean={s['mean']:.4f} std={s['std']:.4f} ci95={s['ci95_half']:.4f}")
    return per_k, ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--zeroshot-model", default="jinaai/jina-embeddings-v2-base-en")
    parser.add_argument("--finetuned", default=None,
                        help="path to a fine-tuned SentenceTransformer checkpoint")
    parser.add_argument("--out", default=None)
    parser.add_argument("--skip-random", action="store_true")
    parser.add_argument("--skip-zeroshot", action="store_true")
    args = parser.parse_args()

    corpus, test_records = load_splits()
    print(f"corpus: {len(corpus)}  test: {len(test_records)}")

    results = {"meta": {"corpus_size": len(corpus), "test_size": len(test_records), "ks": KS}}

    if not args.skip_random:
        print("\nrandom baseline (averaging over %d seeds)..." % RANDOM_SEEDS)
        results["random"] = {}
        for k in KS:
            seed_means = np.array([
                random_topk_sim(test_records, corpus, k, seed=s).mean()
                for s in range(RANDOM_SEEDS)
            ])
            per_query = random_topk_sim(test_records, corpus, k, seed=0)
            results["random"][k] = {
                "per_query": per_query.tolist(),
                **summarize(per_query),
                "seed_mean_std": float(seed_means.std(ddof=1)),
            }
            s = results["random"][k]
            print(f"[random] k={k}: mean={s['mean']:.4f} std={s['std']:.4f} seed_std={s['seed_mean_std']:.4f}")

    if not args.skip_zeroshot:
        print(f"\nloading zero-shot model: {args.zeroshot_model}")
        zs_model = SentenceTransformer(args.zeroshot_model, trust_remote_code=True)
        zs_model.max_seq_length = 2048
        zs_results, _ = evaluate_model(zs_model, args.zeroshot_model, corpus, test_records, args.batch_size, "zeroshot")
        results["zero_shot"] = zs_results
        del zs_model

    if args.finetuned:
        print(f"\nloading fine-tuned model: {args.finetuned}")
        ft_model = SentenceTransformer(args.finetuned, trust_remote_code=True)
        ft_model.max_seq_length = 2048
        ft_results, _ = evaluate_model(ft_model, args.finetuned, corpus, test_records, args.batch_size, "finetuned")
        results["fine_tuned"] = ft_results

        if "zero_shot" in results:
            for k in KS:
                ft_pq = np.array(ft_results[k]["per_query"])
                zs_pq = np.array(results["zero_shot"][k]["per_query"])
                diffs = ft_pq - zs_pq
                t = float(diffs.mean() / (diffs.std(ddof=1) / np.sqrt(diffs.size) + 1e-12))
                print(f"[paired ft vs zs] k={k}: mean_diff={diffs.mean():.4f} t={t:.2f} win_rate={(diffs>0).mean():.2%}")
                results.setdefault("ft_vs_zs", {})[k] = {
                    "mean_diff": float(diffs.mean()),
                    "t": t,
                    "win_rate": float((diffs > 0).mean()),
                }

    print("\n=== summary ===")
    for method in ("random", "zero_shot", "fine_tuned"):
        if method not in results:
            continue
        for k in KS:
            s = results[method][k]
            print(f"  {method:>12} k={k:2d}: {s['mean']:.4f} +/- {s['ci95_half']:.4f}")

    out = args.out or (config.DATA_DIR / "eval_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
