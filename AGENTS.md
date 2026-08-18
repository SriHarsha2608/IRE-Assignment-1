# AGENTS.md

IRE Assignment 1: lexical (BM25) + semantic (embedding/ANN) retrieval pipeline on MIND (English) and EB-NeRD (Danish). Spec: `Assignment1_v1.pdf`. Q1 (data pipeline) and Q2 (BM25) are done; Q3–Q5 are pending.

## Environment (critical)

- **Always use `.venv/bin/python`** (Python 3.14). System `python` has no deps installed.
- The `ire_rec` package (`src/`) is installed editable into `.venv` via `pyproject.toml` (`pip install -e .`). `import ire_rec` must resolve — if it fails, run `make install`.
- **Runtime deps are NOT guaranteed in `.venv`**: the initial venv shipped with only numpy/pandas/polars/pytest (Q1 tests pass without `rank_bm25`, `scikit-learn`, `nltk`, `faiss-cpu`). Before touching retrieval/eval code, verify `import rank_bm25`/`faiss` succeed or run `make install`. Missing deps fail at import time, not at test time.
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
- Embeddings are stored as `{name}.npy` (row i = article i of `{name}_ids.parquet`): MIND `entity_mean` (100-d, ~87% of articles), EB-NeRD `word2vec` (300-d) + `bert` (768-d). Dims/coverage in the manifest.
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
- **Pipeline version guard**: `build_pipeline.py` has `VERSION` (currently `"2"`). If you change the store schema or output layout, **bump `VERSION`** — otherwise `make pipeline` sees a matching manifest and skips the rebuild.

## Retrieval (Q2 done, Q3 pending)

- `retrieval/bm25.py` implements its **own** inverted-index BM25 (numpy-vectorized postings `np.add.at`); do NOT switch to `rank_bm25.BM25Okapi` (`get_scores` scans the whole 65k-doc corpus per query → ~10 min/run vs ~6 min vectorized). Verified `rank-bm25`'s idf formula is replicated (ATIRE eps-floor + 1).
- `retrieval/run_bm25.py` (make `lexical`): builds index over ALL articles (title_abstract), query = titles of up to `history_query_cap` (default 20) most-recent history clicks, retrieves top-K ∈ {50,100,200}, writes `data/processed/<DS>/retrieval/bm25/candidates_{split}.parquet` (`impression_id, split, gt_clicked, candidates, scores, n_query_terms`) + aggregated `recall.json` per dataset. Default `--splits val`; pass `--splits val,test` for both. Candidates are written per-split and recall.json aggregates over whatever per-split files are on disk, so separate val/test runs merge cleanly — rerunning one split keeps the other's recall entries.
- **Q2 gold numbers (BM25 title_abstract, mean recall over impressions with ≥1 click)**: MIND val 0.008/0.016/0.028, test 0.007/0.015/0.025; EB-NeRD-demo val 0.008/0.012/0.022, test 0.008/0.018/0.034; EB-NeRD-small val 0.006/0.010/0.017, test 0.006/0.012/0.022 (K=50/100/200). Low-but-topical retrieval is expected for a candidate generator; the query is title-only and the candidate space is the full catalog.
- Full MIND run ≈ 6 min/split, EB-NeRD-small ≈ 2 min/split. Use `--limit N` for quick smoke tests. Retrieval cost is linear in impressions — running `val,test` in one invocation does not double the work beyond the extra rows.

## Tests

- `tests/test_data_pipeline.py`: 11 tests, fully synthetic (no data/ needed), fast. Covers TSV parsing, labels, temporal split (day + fallback), entity pooling, history causality/cap, unified history schema, cold-start `[]`, and train-only popularity.
- `tests/test_bm25.py`: 7 tests covering the BM25 index, query-from-history capping, recall@K, corpus field variants.
- `make lexical`/`semantic`/`eval`/`predict` reference modules — lexical now exists and works; `semantic`/`eval`/`predict` (Q3–Q5) still pending; don't run them.

## Git

- `origin` is **SSH** (`git@github.com:SriHarsha2608/IRE-Assignment-1.git`), not HTTPS. Commit with meaningful messages (Q8); no large files, no force-push.