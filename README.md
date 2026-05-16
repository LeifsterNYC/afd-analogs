# afd-analogs

Historical weather pattern matching by NLP analysis of NWS Area Forecast Discussions (AFDs). Given a current AFD, retrieve the most similar historical AFDs from an archive and surface the weather that actually occurred on those dates.

The project compares four retrieval methods on a held-out test set:

1. Random baseline
2. Zero-shot retrieval using a pretrained Jina v2 sentence embedding
3. A contrastive fine-tune of Jina v2 using triplet loss with hardest-pair mining over a 37-dimensional weather vector
4. Structured-extraction retrieval that parses each AFD into JSON event frames via Claude and matches by frame-Jaccard

## Team

- Leif Rogers (lfr42)
- Jingyu Xu (jx62)
- Eric Gao (xg255)

## Setup

```bash
python -m venv venv
source venv/bin/activate          # or: venv\Scripts\activate on Windows
pip install -r requirements.txt
```

The structured-extraction pipeline shells out to `claude -p` (the Claude Code CLI). If you want to run that part, install Claude Code from <https://claude.com/claude-code> and make sure `claude --version` works in your shell.

## Data

The repo does not ship the raw data. Two ways to get the paired dataset that the rest of the pipeline needs:

### Option A: scrape everything from scratch

```bash
python -m src.scraping.iem            # NWS AFDs for the BGM forecast office
python -m src.scraping.iem_asos       # METAR for KBGM, KRME, KAVP, KELM, KITH
python -m src.scraping.noaa           # NOAA LCD daily snow data
python -m src.processing.build_dataset
```

This takes a few hours (mostly the AFD scrape). All scrapers are resume-safe.

### Option B: use a pre-built `paired_dataset.json`

Drop it at `data/processed/paired_dataset.json` and skip straight to training or eval.

## Train

```bash
python -m src.training.train --epochs 3 --batch-size 64
```

Defaults match the spec from the status report (jina-embeddings-v2-base-en, seq=4096, batch=64, 3 epochs). Needs roughly 40-80 GB of GPU VRAM at the spec; on smaller GPUs drop `--max-seq 2048 --batch-size 16 --amp`. The trained model is written to `data/models/contrastive/`.

## Evaluate

```bash
python -m src.retrieval.evaluate --finetuned data/models/contrastive
```

Runs random + zero-shot + fine-tuned retrieval, computes mean weather similarity at K=1,3,5,10 with 95% CIs and paired t-statistics, writes `data/eval_results.json`.

For the structured-extraction comparison:

```bash
python -m src.extraction.extract --workers 4
python -m src.extraction.frame_retrieval
```

The first command writes one JSON file per AFD under `data/processed/frames/`. The second consumes those and writes `data/extraction_eval.json`.

## Layout

```
config.py                       paths, station list, date range
requirements.txt                python deps
src/
  scraping/                     IEM AFD, IEM ASOS, NOAA LCD scrapers
  processing/                   parse + align AFDs with observations
  training/                     contrastive fine-tune (similarity, dataset, train)
  retrieval/                    FAISS index, evaluate, sample
  extraction/                   claude -p extraction, frame-Jaccard retrieval
report/                         LaTeX source + PDF of the final report
```

## Stations

The dataset uses METAR from five stations covering the NWS Binghamton forecast area: KBGM (Binghamton), KRME (Griffiss/Rome), KAVP (Wilkes-Barre/Scranton), KELM (Elmira), and KITH (Ithaca). CWA-wide snow data comes from NOAA LCD at KBGM.

## Reproducing the reported numbers

The headline numbers in the report (random 0.7904, zero-shot 0.8469, fine-tuned 0.8381, extraction 0.8495 at K=1) come from:

- `data/processed/paired_dataset.json` after the 2007 date floor and max_gust outlier filter
- 80/10/10 random split with seed 42 (`src/training/train.py`)
- 681-record test split (`data/processed/test_split.json`) held out before training
- Evaluation against the remaining 6,129-record corpus
