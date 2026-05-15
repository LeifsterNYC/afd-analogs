import argparse
import json
import random
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import config
from src.training.dataset import load_records, build_triplets

from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.evaluation import TripletEvaluator
from torch.utils.data import DataLoader


def train(
    data_path=None,
    model_name="jinaai/jina-embeddings-v2-base-en",
    output_dir=None,
    epochs=3,
    batch_size=64,
    val_split=0.1,
    test_split=0.1,
    max_seq_length=4096,
    use_amp=False,
    seed=42,
):
    if data_path is None:
        data_path = config.PROCESSED_DIR / "paired_dataset.json"
    if output_dir is None:
        output_dir = config.DATA_DIR / "models" / "contrastive"

    random.seed(seed)

    print(f"Loading dataset from {data_path}...")
    records = load_records(data_path)
    random.shuffle(records)

    n = len(records)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    test_records = records[:n_test]
    val_records = records[n_test:n_test + n_val]
    train_records = records[n_test + n_val:]
    print(f"  {len(train_records)} train / {len(val_records)} val / {len(test_records)} test")

    test_path = config.PROCESSED_DIR / "test_split.json"
    with open(test_path, "w") as f:
        json.dump(test_records, f)

    print("Building triplets...")
    train_triplets = build_triplets(train_records)
    val_triplets = build_triplets(val_records)

    def to_examples(triplets):
        return [InputExample(texts=[a, p, n]) for a, p, n in triplets]

    train_examples = to_examples(train_triplets)
    val_examples = to_examples(val_triplets)

    print(f"Loading base model: {model_name}")
    model = SentenceTransformer(model_name, trust_remote_code=True)
    model.max_seq_length = max_seq_length

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    loss_fn = losses.TripletLoss(model=model)

    evaluator = TripletEvaluator(
        anchors=[e.texts[0] for e in val_examples],
        positives=[e.texts[1] for e in val_examples],
        negatives=[e.texts[2] for e in val_examples],
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    warmup_steps = int(len(train_dataloader) * epochs * 0.1)
    print(f"Training {epochs} epochs, warmup={warmup_steps} steps...")

    model.fit(
        train_objectives=[(train_dataloader, loss_fn)],
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=str(output_dir),
        save_best_model=True,
        show_progress_bar=True,
        use_amp=use_amp,
    )

    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="jinaai/jina-embeddings-v2-base-en")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq", type=int, default=4096)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--data", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    train(
        data_path=args.data,
        model_name=args.model,
        output_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq,
        use_amp=args.amp,
    )
