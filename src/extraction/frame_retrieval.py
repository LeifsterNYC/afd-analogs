import argparse
import json
import sys
import os
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import config
from src.training.similarity import weather_similarity
from src.retrieval.evaluate import load_splits, per_query_topk_sim, summarize, KS


def load_frames(date):
    p = config.PROCESSED_DIR / "frames" / f"{date}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def frame_signature(frames):
    if not frames:
        return set()
    sig = set()
    for f in frames:
        et = (f.get("event_type") or "").lower()
        it = (f.get("intensity") or "").lower()
        mc = (f.get("mechanism") or "").lower()
        if et:
            sig.add(("event", et))
        if et and it:
            sig.add(("event_int", et, it))
        if mc:
            # mechanism is free text; first word only
            sig.add(("mech", mc.split()[0]))
    return sig


def jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def build_signatures(records):
    sigs = []
    missing = 0
    for r in records:
        fr = load_frames(r["date"])
        if fr is None:
            missing += 1
            sigs.append(set())
        else:
            sigs.append(frame_signature(fr))
    print(f"  loaded {len(records) - missing}/{len(records)} frame files")
    return sigs


def retrieve_topk(query_sigs, corpus_sigs, k):
    ids = np.zeros((len(query_sigs), k), dtype=np.int64)
    for i, qs in enumerate(query_sigs):
        scores = np.array([jaccard(qs, cs) for cs in corpus_sigs])
        ids[i] = np.argsort(-scores)[:k]
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    corpus, test_records = load_splits()
    print(f"corpus: {len(corpus)}  test: {len(test_records)}")

    print("loading corpus frame signatures...")
    corpus_sigs = build_signatures(corpus)
    print("loading test frame signatures...")
    test_sigs = build_signatures(test_records)

    print("retrieving top-K by frame jaccard...")
    ids = retrieve_topk(test_sigs, corpus_sigs, max(KS))

    results = {"meta": {"corpus_size": len(corpus), "test_size": len(test_records), "ks": KS}}
    results["extraction"] = {}
    for k in KS:
        per_query = per_query_topk_sim(test_records, corpus, ids, k)
        results["extraction"][k] = {"per_query": per_query.tolist(), **summarize(per_query)}
        s = results["extraction"][k]
        print(f"[extraction] k={k}: mean={s['mean']:.4f} std={s['std']:.4f} ci95+-{s['ci95_half']:.4f}")

    out = args.out or (config.DATA_DIR / "extraction_eval.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
