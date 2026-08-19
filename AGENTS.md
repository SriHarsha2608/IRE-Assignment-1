# AGENTS.md

IRE Assignment 1: lexical (BM25) + semantic (embedding/ANN) retrieval pipeline on MIND (English) and EB-NeRD (Danish). Spec: `Assignment1_v1.pdf`. Q1 (data pipeline), Q2 (BM25), and Q3 (semantic/embedding) are done; Q4–Q5 are pending.

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

## Retrieval (Q2 + Q3 done, Q4 pending)

- `retrieval/bm25.py` implements its **own** inverted-index BM25 (numpy-vectorized postings `np.add.at`); do NOT switch to `rank_bm25.BM25Okapi` (`get_scores` scans the whole 65k-doc corpus per query → ~10 min/run vs ~6 min vectorized). IDF = `ln((N - df + 0.5)/(df + 0.5)) + 1`; negative raw IDF terms are floored to `epsilon * average_idf` (rank-bm25's ATIRE-style floor) so a query term never lowers a score. Real corpora have 0 negative-IDF terms, so the floor never fires in production.
- `retrieval/run_bm25.py` (make `lexical`): builds index over ALL articles (title_abstract), query = titles of up to `history_query_cap` (default 20) most-recent history clicks, retrieves top-K ∈ {50,100,200}, writes `data/processed/<DS>/retrieval/bm25/candidates_{split}.parquet` (`impression_id, split, gt_clicked, candidates, scores, n_query_terms`) + aggregated `recall.json` per dataset. Default `--splits val`; pass `--splits val,test` for both. Candidates are written per-split and recall.json aggregates over whatever per-split files are on disk, so separate val/test runs merge cleanly — rerunning one split keeps the other's recall entries.
- **Q2 gold numbers (BM25 title_abstract, mean recall over impressions with ≥1 click)**: MIND val 0.008/0.016/0.028, test 0.007/0.015/0.025; EB-NeRD-demo val 0.008/0.012/0.022, test 0.008/0.018/0.034; EB-NeRD-small val 0.006/0.010/0.017, test 0.006/0.012/0.022 (K=50/100/200). Low-but-topical retrieval is expected for a candidate generator; the query is title-only and the candidate space is the full catalog.
- `retrieval/semantic.py` (Q3): `load_embeddings` loads `{name}.npy` + `{name}_ids.parquet`, validates `len(ids) == mat.shape[0]` and unique ids (hard errors, no silent corruption), and restricts to the dataset's article catalog so the candidate space matches BM25. `build_ann_index` **never mutates the input matrix**: it normalizes a *copy* (L2 row-wise → FAISS `IndexFlatIP` cosine; exact, fine up to 65k×768; HNSW/IVF are the 10×-scale options) and the raw matrix stays for pooling. `mean_pool_user_vector` mean-pools the **raw** history-article embeddings, then unit-normalizes the pooled mean when `normalize` is on (config `retrieval.semantic.normalize`); empty history **or a zero-norm pooled mean** → `None` = no search (empty candidates, recall 0), so an undefined zero query is never sent to FAISS. `normalize=false` is internally consistent (raw inner product on both documents and query). `retrieval/run_semantic.py` (make `semantic`) mirrors run_bm25's CLI (`--datasets --splits --top-k --limit --embedding`); outputs go to `retrieval/semantic/<emb>/candidates_{split}.parquet` (`impression_id, split, gt_clicked, candidates, scores, n_history_used, n_history`) + `recall.json` + `comparison.json`.
- `comparison.json` = BM25 vs semantic recall table + cold-start/warm slice at recall@maxK + a **`fair` block**: because semantic can only retrieve embedding-covered articles, direct recall@K would be unfair (gt clicks on unembedded articles are unreachable for semantic). The `fair` block reports `coverage` (`n_covered/n_catalog`; MIND `entity_mean` covers 56777/65238 = 87.03%, EB-NeRD 100%) and recomputes recall@K for **both** methods only on impressions whose `gt_clicked` are all embedding-covered (equal recall ceiling), with a note that BM25 still searched the full catalog.
- **MIND embeddings are `entity_mean` = mean-pooled Wikidata entity vectors** from the provided `entity_embedding.vec` (100-d, `mind.build_entity_article_embeddings`); they are **NOT** a BERT/XLM-R article-text embedding. Only articles with entities get one (~87% coverage). MIND semantic recall is low partly for this reason — keep this caveat in any comparison.
- **Q3 gold numbers (semantic, mean recall over impressions with ≥1 click)**: MIND `entity_mean` val 0.003/0.004/0.007, test 0.002/0.003/0.006; EB-NeRD-demo `word2vec` val 0.007/0.013/0.027, test 0.006/0.011/0.019; EB-NeRD-small `word2vec` val 0.005/0.010/0.019, test 0.004/0.007/0.012 (K=50/100/200). Default EB-NeRD embedding is `word2vec` (config `retrieval.semantic.embedding`); pass `--embedding bert` to switch — bert demo val recall@200 0.030 (beats BM25 0.022) but is similar/slightly weaker on small. **Fair-population recall@200 (MIND, impressions with all gt covered, n_val=49 980 / n_test=82 310)**: val semantic 0.009 vs BM25 0.028; test semantic 0.007 vs BM25 0.026. **Takeaway: BM25 ≫ entity_mean semantic on MIND (entity pooling is a weak signal, 13% of articles are unembeddable) even on the embedding-covered population; semantic ≈/slightly < BM25 on EB-NeRD (word2vec/bert capture Danish topical similarity well).**
- Cold-start users (empty history) get an empty query in both BM25 and semantic → empty candidates → 0 recall (MIND val has 1507 cold impressions, EB-NeRD demo/small have none in val/test).
- **MIND dev quirk**: some `impression_id` values are reused for genuinely distinct impressions (13 921 dup ids, test split only). run_bm25/run_semantic process one row per impression row (recall.json is per-row, matching Q2 gold numbers); the Q3 comparison's cold/warm slice dedupes on `impression_id` (keep-first, per-split) so both methods are sliced on identical populations.
- Full MIND semantic run ≈ 4.5 min (val+test), EB-NeRD-small word2vec ≈ 7 min, bert ≈ 14 min. Use `--limit N` for quick smoke tests.

## Tests

- `tests/test_data_pipeline.py`: 11 tests, fully synthetic (no data/ needed), fast. Covers TSV parsing, labels, temporal split (day + fallback), entity pooling, history causality/cap, unified history schema, cold-start `[]`, and train-only popularity.
- `tests/test_bm25.py`: 7 tests covering the BM25 index, query-from-history capping, recall@K, corpus field variants.
- `tests/test_semantic.py`: 11 tests covering embedding load + validation (dim match, unique ids, catalog restriction), L2-normalize of a *copy* + FAISS cosine index, mean-pool of raw vectors (normalize on/off, zero-norm → `None`), cold-start `None`, result padding, and an end-to-end run_semantic on a synthetic store (embeddings → index → history → user vec → FAISS → recall@K).
- `make lexical` and `make semantic` (Q2/Q3) both work; `eval`/`predict` (Q4/Q5) still pending; don't run them.

## Git

- `origin` is **SSH** (`git@github.com:SriHarsha2608/IRE-Assignment-1.git`), not HTTPS. Commit with meaningful messages (Q8); no large files, no force-push.