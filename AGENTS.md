# AGENTS.md

IRE Assignment 1: lexical (BM25) + semantic (embedding/ANN) retrieval pipeline on MIND (English) and EB-NeRD (Danish). Spec: `Assignment1_v1.pdf`. Q1 (data pipeline) is done; Q2–Q5 are pending.

## Environment (critical)

- **Always use `.venv/bin/python`** (Python 3.14). System `python` has no deps installed.
- The `ire_rec` package (`src/`) is installed editable into `.venv` via `pyproject.toml` (`pip install -e .`). `import ire_rec` must resolve — if it fails, run `make install`.
- `make` targets auto-detect the venv (`PY` var). Run tests with `make test`.

## Commands

```bash
make install      # dependencies (requirements.txt) + editable install
make data         # full Q1 rebuild: download -> parse -> temporal split -> feature store
make pipeline     # incremental; skips datasets whose raw fingerprints are unchanged
make test         # unit tests (synthetic data only, no raw files needed)
.venv/bin/python -m ire_rec.build_pipeline --datasets MIND,EB-NeRD-demo
.venv/bin/python -m ire_rec.build_pipeline --rebuild --skip-embeddings   # avoid the ~4.6GB BERT consolidation
```

## Layout & flow

- `src/ire_rec/build_pipeline.py` orchestrates; `datasets/mind.py` + `datasets/ebnerd.py` parse raw → unified schema; `split.py` does the temporal split; `dataio.py` owns unified column names + store I/O + manifest.
- Column-name constants live in `dataio.py` (`ARTICLE_ID`, `HISTORY`, `LABELS`, `SPLIT`, ...). Use them; never invent variant spellings.
- Output: `data/processed/` (gitignored) — one dir per dataset (`MIND`, `EB-NeRD-demo`, `EB-NeRD-small`), shared `EB-NeRD/embeddings/`, plus `manifest.json` (fingerprints, split boundaries, counts).
- IDs are **strings everywhere** in the store. MIND IDs look like `N123`/`U123`; EB-NeRD numeric IDs are cast to str.
- Zip archives nest their content under a same-named top-level dir (`MINDsmall_train/MINDsmall_train/news.tsv`). Use `utils.find_file`; never hardcode `dir / "news.tsv"`.
- **Do not commit** `data/` or any `*.zip`/`*.parquet`/`*.npy` (`.gitignore`, Q8 policy). Raw zips must already exist in `data/raw`; the pipeline only downloads what's missing.

## What will bite you

- **Memory (11 GB laptop)**: the EB-NeRD per-impression history is built per-user via `partition_by(USER_ID)` + numpy `searchsorted`, capped at `history_size` (default 50). A naive `clicks ⋈ behaviors` join on `user_id` OOMs on the `small` bundle — keep the partitioned implementation. BERT consolidation peaks ~4.6 GB; use `--skip-embeddings` for local bundle rebuilds.
- **polars 1.43 quirks**:
  - `map_elements` over a List-typed column passes a Series (truth-testing it raises). Use a `pl.struct([...]).map_elements` wrapper (dict in) or a vectorized approach (`list.eval`, `when/then`).
  - `list.is_in(other_list)` between two List columns is NOT elementwise (returns a scalar) — use explode/regroup.
  - `list.eval` cannot reference sibling row columns — only `pl.element()`.
  - `.explode()` on List columns emits an `empty_as_null` deprecation warning (polars 2.0 flips the default) — pass `empty_as_null=True` explicitly to preserve current semantics.
- **MIND `news.tsv`** contains unescaped `"` in abstracts → parse with `quote_char=None` in `pl.read_csv`. It has no `body`/`published_time` (nullable in unified schema).
- **MIND `entity_embedding.vec`** is space-separated (`id f0 f1 ...`, dim 100), keyed by WikidataId that matches the `WikidataId` field in `news.tsv` entity JSON. Article embedding = mean-pool of entity vectors; only articles with entities get one (~87% coverage). `entity_embedding.vec` is optional per archive in `build_mind` (missing file → warning, not a crash).
- **History schema is unified** across datasets: `[user_id, impression_id, impression_time, article_id, click_time, read_time, recency]` (same column order + dtypes). MIND's `click_time`/`read_time`/`recency` are null (unavailable in raw data) but present; EB-NeRD carries real values.
- **Cold-start users get `history=[]`, never `None`** — downstream Q2/Q3 loops can iterate safely.
- **EB-NeRD history**: raw `history.parquet` is per-user *aligned lists* (`impression_time_fixed`/`article_id_fixed`/`read_time_fixed`). Per-impression history must contain **strictly prior** clicks (`click_time < impression_time`) — this is the no-future-click-leakage guarantee (Q9).
- **Popularity is train-only**: `dataio.add_popularity` aggregates `n_inviews`/`n_clicks` from the **train** split only (Q9 — no test-set information in features). Articles never exposed in train get null.
- **Temp/extract dir is repo-local** (`data/tmp`, gitignored via `data/`) — not `/tmp/ire_rec`.
- `pyproject.toml` declares runtime deps (mirrors `requirements.txt`); `make install` is still the canonical bootstrap.

## Tests

- `tests/test_data_pipeline.py`: 11 tests, fully synthetic (no data/ needed), fast. Covers TSV parsing, labels, temporal split (day + fallback), entity pooling, history causality/cap, unified history schema, cold-start `[]`, and train-only popularity.
- `make lexical`/`semantic`/`eval`/`predict` reference modules that do not exist yet (`retrieval/`, `evaluation/` are empty stubs) — Q2–Q5 pending; don't run them.

## Git

- `origin` is **SSH** (`git@github.com:SriHarsha2608/IRE-Assignment-1.git`), not HTTPS. Commit with meaningful messages (Q8); no large files, no force-push.