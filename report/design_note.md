# IRE Assignment 1 — Design Note (supplementary)

This note documents known interpretive and completeness gaps in the retrieval /
evaluation pipeline. It complements `AGENTS.md` and the assignment spec
(`Assignment1_v1.pdf`). Code behavior is unchanged unless noted; these are
documentation/labeling clarifications.

---

## a) MIND `recency` is always null (dataset limitation)

MIND's raw `behaviors.tsv` provides only a **titles-only history list** for each
user — it does **not** include per-history-click timestamps. Our unified history
schema therefore carries `click_time` / `read_time` / `recency` columns for MIND
impressions, but they are **always null** for MIND (real values exist only for
EB-NeRD, whose `history.parquet` records `impression_time_fixed` / `read_time_fixed`).

Consequence: any recency-aware feature or ranking that would consume MIND
`recency` cannot be computed from the raw data. This is a known limitation of the
MIND source, not a pipeline bug. EB-NeRD recency is real and used as-is.

---

## b) Beyond-accuracy metrics are computed at retrieval depth (label them)

`intra_list_diversity`, `novelty`, and `article/category_coverage` are currently
computed over the **full candidate list** produced by retrieval — i.e. whatever
top-`K` was used for candidate generation (e.g. `K=200`), **not** a smaller
"final recommendation" depth such as `@10`.

These must therefore be reported explicitly with their depth, e.g.:

- **ILD@200** (intra-list diversity at retrieval depth 200)
- **Novelty@200**
- **Coverage@200** (article and category coverage)

Reporting them unlabeled as "diversity / novelty / coverage" would overstate what
is measured, since the @200 catalog-truncation set is far larger than a realistic
served slate. If a smaller reporting depth is desired, an optional `report_k`
parameter can be added to `evaluate_candidates` that truncates `cands`/`scores`
(after the ranking fix in item 1) to the top `report_k` before computing
beyond-accuracy metrics. This is a labeling-level change, not needed for
correctness.

---

## c) MIND semantic embedding choice: `entity_mean` (a defensible third option)

The assignment spec lists two embedding options for semantic retrieval: (i) the
provided article-level embeddings, or (ii) a self-computed BERT/XLM-R text
embedding. MIND provides **neither** of these at the article level:

- MIND ships **no official article embedding artifact**;
- MIND news articles have **no body text** in `news.tsv` (only title + abstract),
  and the available text is sparse, making a from-scratch BERT/XLM-R article
  encoder a weak, extra-effort option with no ground-truth to validate against.

Instead we use **`entity_mean`**: the mean-pool of the Wikidata entity vectors
(`entity_embedding.vec`, 100-d) for the entities annotated on each article. This
is a **third, defensible approach given MIND's raw data** — it reuses the only
dense representation MIND actually provides and requires no auxiliary model.

Limitation and mitigation:

- Only articles with annotated entities get an embedding, so coverage is
  **~87%** (56 777 / 65 238 MIND articles). The remaining ~13% are unembeddable
  and unreachable by semantic retrieval.
- This is explicitly handled in `comparison.json` via the **`fair`** block: since
  semantic can only retrieve embedding-covered articles, direct recall@K would be
  unfair to semantic. We report coverage (`n_covered/n_catalog`) and recompute
  recall@K for **both** BM25 and semantic on the population of impressions whose
  `gt_clicked` are all embedding-covered, giving an equal recall ceiling.

A real BERT/XLM-R MIND article embedding was deliberately treated as a documented
scope decision, not implemented, given schedule constraints — `entity_mean` is
the pragmatic, data-grounded choice.

---

## d) Cold-start BM25 fallback is a deterministic zero-score fill (artifact)

For **cold-start** impressions (empty history -> empty query), BM25 cannot produce
a meaningful ranking. Our `Bm25Index.search` returns a **deterministic top-`K`
fill of the first `K` articles by catalog index** (score 0.0), not a learned or
relevance-ordered list.

Therefore any reported **cold-start BM25 recall / accuracy** is partly an artifact
of article ordering in the catalog, not retrieval quality. This caveat must be
stated explicitly next to any cold-start numbers in the report. By contrast,
semantic retrieval returns **empty** candidates for cold starts (no user vector to
search), so its cold-start recall is genuinely 0 and is not an artifact.

---

## e) Serving-time availability: indices are not publication-time aware

Both BM25 and semantic indices are built over the **full article catalog**
regardless of publication time. Consequently a query issued at time `t` can
retrieve articles that were **not yet published at `t`**. This is a known
limitation of the retrieval stage.

Important: this is **not** a Q9 leakage violation. Q9 leakage concerns using
**future click data**; serving-time article availability is a separate, milder
issue (it does not inject any user-click signal from the future). It is noted here
for completeness.

Possible extension: EB-NeRD exposes a real `published_time` column per article. A
serving-time-correct extension would restrict the candidate set to articles with
`published_time <= impression_time` before scoring. MIND lacks per-article
`published_time` (only impressions carry a timestamp), so the same extension is
not directly available for MIND without external publication metadata.

---

## f) Evaluation ranking correctness (item 1 fix)

`evaluate_candidates` / `_determine_ranking` now rank retrieved articles **above**
unretrieved ones regardless of score sign: the sort key is
`(retrieved_flag, -score_if_retrieved, retrieval_rank, inview_pos)`. This matters
for cosine semantic scores in `[-1, 1]` that are frequently negative — previously
a negatively-scored but genuinely-retrieved clicked article could be ranked below
unretrieved articles, silently deflating MRR / nDCG@5 / nDCG@10 for the semantic
method. See `tests/test_evaluation.py` for regression tests.

**Empirical note (item 1 verification):** on the current dev candidate files,
**all** retrieval scores are non-negative — BM25 is ≥ 0 by construction, and
semantic candidate scores (MIND `entity_mean`, EB-NeRD `word2vec`/`bert`) are all
in a high-positive band (e.g. MIND entity_mean ∈ [0.29, 1.0]; EB-NeRD word2vec ∈
[0.85, 1.0]) because top-`K` ANN retrieval returns only the *most similar*
(highest cosine) neighbors. With no negative retrieved scores, the bug can never
trigger, so re-running eval with the fixed ranking yields **identical**
MRR/nDCG@5/nDCG@10 as the buggy version on this data (MIND semantic val MRR =
0.25575 either way). The fix is a genuine correctness safeguard for any data /
retrieval configuration that can produce negative retrieved scores, but it does
**not** alter the current reported metrics. (The pre-existing
`semantic_entity_mean_val.json` MRR of 0.28382 seen at the start of this work
corresponds to an *older candidate-file generation* — candidate drift — and is
unrelated to this ranking fix; both code versions compute 0.25575 on the current
file.) AGENTS.md recall@K "gold numbers" are candidate-level and unaffected.
