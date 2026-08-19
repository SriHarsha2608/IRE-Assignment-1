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

## Deliverables map (Q1-Q9)

| Q | Item | Location |
|---|------|----------|
| Q1 | Reproducible data pipeline | `make data` -> `src/ire_rec/build_pipeline.py`, `datasets/`, `split.py`, `dataio.py` |
| Q2 | BM25 lexical retrieval | `src/ire_rec/retrieval/bm25.py` |
| Q3 | Embedding/semantic retrieval | `src/ire_rec/retrieval/semantic.py`, `run_semantic.py` (`make semantic` → `retrieval/semantic/<emb>/` candidates + recall/comparison; fair BM25-vs-semantic recall on the embedding-covered GT population with coverage reported). MIND embeddings are mean-pooled Wikidata entity vectors (`entity_mean`, 100-d, ~87% coverage), **not** BERT/XLM-R. |
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