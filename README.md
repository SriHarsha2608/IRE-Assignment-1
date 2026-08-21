# IRE Assignment 1 - Lexical & Semantic Retrieval on EB-NeRD and MIND

CS4.406 Information Retrieval & Extraction - Assignment 1

## Overview

Recommendation/retrieval foundation pipeline on two news datasets:
- **MIND** (English, Microsoft News)
- **EB-NeRD** (Danish, Ekstra Bladet / RecSys 2024)

Pipeline: reproducible data pipeline -> lexical (BM25) and semantic (embeddings + ANN)
candidate generation -> offline evaluation harness (AUC, MRR, nDCG@5/10, diversity,
novelty, coverage, sliced + bootstrap CI).

## Structure

```
IRE-Assignment-1/
├── Assignment1_v1.pdf        # assignment spec
├── Makefile                  # one-command entry points
├── requirements.txt          # python deps
├── .gitignore                # excludes large files / data (per Q8 policy)
├── configs/                  # dataset / run configurations
├── data/
│   └── raw/                  # raw downloads (gitignored)
│       ├── MIND/             # MINDsmall_train/dev.zip
│       └── EB-NeRD/          # ebnerd_demo/small.zip + embeddings
├── libs/
│   └── ebnerd-benchmark/     # EB-NeRD starter code (ebrec package)
├── src/
│   └── ire_rec/              # our pipeline code (Q1-Q4)
├── tests/                    # incl. no-future-click-leakage test (Q9)
├── notebooks/                # exploratory analysis / experiments
├── predictions/              # Codabench submission files (Q5)
└── report/                   # design note (Q6)
```

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate   # one-time
pip install -r requirements.txt
pip install -e .                                      # installs the ire_rec package (src/)
make data          # Q1: download -> parse -> temporal split -> feature store (--rebuild)
make pipeline      # re-run quickly; skips everything already up to date
make test          # unit + no-future-click-leakage tests
```

Subset selection and safe flags:

```bash
python -m ire_rec.build_pipeline --datasets MIND,EB-NeRD-demo
python -m ire_rec.build_pipeline --rebuild --skip-embeddings   # rebuild without the heavy BERT consolidation
```

## Large datasets (Codabench submissions)

The default `make data` / `make pipeline` builds only the **small** subsets
(`MINDsmall`, `EB-NeRD-demo`, `EB-NeRD-small`) used for local development. The
Codabench competitions require the **Large** subsets:

- **MIND (Codabench 13967)** → `MIND-large` (`MINDlarge_train.zip`, `MINDlarge_dev.zip`)
- **EB-NeRD / RecSys 2024 (Codabench 2469)** → `EB-NeRD-large` (`ebnerd_large.zip`)

Support is wired but **opt-in** — these are never built by `make data` or by the
default `--datasets all`. Request them explicitly:

```bash
python -m ire_rec.build_pipeline --datasets MIND-large
python -m ire_rec.build_pipeline --datasets EB-NeRD-large
python -m ire_rec.make_predictions --datasets MIND-large,EB-NeRD-large --out predictions/large
```

Caveats:

- **MIND Large is GATED on Hugging Face** (`yjw1029/MIND`, `MIND-Large`). You must
  run it on a machine where you have authenticated access (`huggingface-cli login`
  with a MIND license). The pipeline fails with a clear "gated" error rather than
  bypassing auth or downloading an unrelated mirror. The `downloads.MIND_large` URL
  in `configs/default.yaml` is the official gated repo.
- **EB-NeRD Large is far larger than this 11 GB laptop environment.** It is
  supported (config + shared `EB-NeRD/embeddings` stage), but running it locally
  here is not feasible — use a capable machine.
- Each Large dataset has its **own** manifest entry and output directory
  (`data/processed/MIND-large`, `data/processed/EB-NeRD-large`); a small subset's
  cache never satisfies a Large run and vice versa.
- MIND embeddings stay `entity_mean` (Wikidata entity mean-pool); EB-NeRD Large
  uses the configured `retrieval.semantic.embedding` (default `word2vec`).

This environment only executed the small subsets end-to-end; Large runs are
structural-only (config + routing + tests) unless run on a suitably provisioned
machine.

## Deliverables map (Q1-Q9)

| Q | Item | Location |
|---|------|----------|
| Q1 | Reproducible data pipeline | `make data` -> `src/ire_rec/build_pipeline.py`, `datasets/`, `split.py`, `dataio.py` |
| Q2 | BM25 lexical retrieval | `src/ire_rec/retrieval/bm25.py` |
| Q3 | Embedding/semantic retrieval | `src/ire_rec/retrieval/semantic.py`, `run_semantic.py` (`make semantic` → `retrieval/semantic/<emb>/` candidates + recall/comparison; fair row-level BM25-vs-semantic recall on the embedding-covered GT population with coverage reported). MIND embeddings are mean-pooled Wikidata entity vectors (`entity_mean`, 100-d, ~87% coverage), **not** BERT/XLM-R. |
| Q4 | Evaluation harness | `src/ire_rec/evaluation/` |
| Q5 | Codabench submissions | `predictions/` |
| Q6 | Design note (<=4 pages) | `report/design_note.pdf` |
| Q7-Q9 | Policies | see spec; leakage test in `tests/` |

## Q1 feature store layout

```
data/processed/                      # gitignored (Q8)
├── manifest.json                    # pipeline version, raw fingerprints, split boundaries, counts
├── MIND/                            # articles / impressions(with split) / history / embeddings(entity_mean)
├── EB-NeRD-demo/  EB-NeRD-small/    # articles / impressions / history per bundle
└── EB-NeRD/embeddings/              # shared w2v.npy & bert.npy + *_ids.parquet (dense matrix + id order)
```