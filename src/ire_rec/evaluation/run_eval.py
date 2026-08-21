from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import polars as pl

from ..config import load_config, processed_dir
from ..dataio import (
    ARTICLE_ID,
    CATEGORY,
    HISTORY,
    IMPRESSION_ID,
    IMPRESSION_ROW_ID,
    INVIEW,
    LABELS,
    N_CLICKS,
    N_INVIEWS,
    SPLIT,
    read_df,
)
from .metrics import (
    auc_score,
    bootstrap_coverage_ci,
    bootstrap_coverage_ci_cats,
    bootstrap_mean_ci,
    intra_list_diversity,
    mrr_from_ranked,
    ndcg_at_k_from_ranked,
    novelty_for_ids,
)

log = logging.getLogger(__name__)

UNKNOWN = "__UNKNOWN__"


def _history_len(impr: pl.DataFrame) -> pl.DataFrame:
    return impr.select([IMPRESSION_ROW_ID, HISTORY]).with_columns(
        pl.col(HISTORY).fill_null([]).list.len().alias("n_history")
    )


def _build_popularity(arts: pl.DataFrame):
    """Train-only popularity: n_inviews, fallback n_clicks, else 0 (Laplace)."""
    pop = (
        pl.when(arts[N_INVIEWS].is_not_null())
        .then(pl.col(N_INVIEWS).cast(pl.Float64))
        .otherwise(
            pl.when(arts[N_CLICKS].is_not_null())
            .then(pl.col(N_CLICKS).cast(pl.Float64))
            .otherwise(0.0)
        )
    ).alias("_pop")
    arts = arts.with_columns(pop)
    total = float((arts["_pop"] + 1.0).sum())
    ids = arts[ARTICLE_ID].to_list()
    id_to_idx = {a: i for i, a in enumerate(ids)}
    p_arr = ((arts["_pop"].to_numpy() + 1.0) / total).tolist()
    p_lookup = {a: p_arr[i] for i, a in enumerate(ids)}
    id_to_cat = {
        a: (c if c is not None else UNKNOWN)
        for a, c in zip(arts[ARTICLE_ID], arts[CATEGORY])
    }
    catalog_set = set(ids)
    catalog_cats = set(id_to_cat.values())
    return p_lookup, id_to_cat, catalog_set, catalog_cats


def _validate(cand: pl.DataFrame, split: str) -> None:
    if IMPRESSION_ROW_ID not in cand.columns:
        raise ValueError("candidate file missing 'impression_row_id'")
    if not cand[IMPRESSION_ROW_ID].is_unique().all():
        raise ValueError("impression_row_id is not unique within candidate file")
    if SPLIT not in cand.columns:
        raise ValueError("candidate file missing 'split'")
    if not (cand[SPLIT] == split).all():
        raise ValueError(
            f"candidate split mismatch: expected '{split}', "
            f"got {cand[SPLIT].unique().to_list()}"
        )
    if "candidates" not in cand.columns or "scores" not in cand.columns:
        raise ValueError("candidate file missing 'candidates'/'scores'")
    if "gt_clicked" not in cand.columns:
        raise ValueError("candidate file missing required 'gt_clicked' column")
    nc = cand.select(pl.col("candidates").list.len().alias("_nc"))
    ns = cand.select(pl.col("scores").list.len().alias("_ns"))
    if not (nc["_nc"].to_numpy() == ns["_ns"].to_numpy()).all():
        raise ValueError("candidate row has len(candidates) != len(scores)")


def _validate_joined(joined: pl.DataFrame, catalog_set: set, split: str) -> None:
    """Validation that requires the joined impressions (invariants A-E).

    Done with a single memory-bounded row iteration (no ``explode`` of the
    large candidate list columns) so it stays practical on the full corpus.
    Empty candidate lists (cold-start semantic rows) are valid and skipped.
    """
    # A. inview / labels present and equal length
    if INVIEW not in joined.columns or joined["inview"].null_count() > 0:
        raise ValueError(
            "candidate row(s) have no matching impression (inview missing/null)"
        )
    if LABELS not in joined.columns or joined["labels"].null_count() > 0:
        raise ValueError(
            "candidate row(s) have no matching impression (labels missing/null)"
        )
    len_ok = joined.select(
        (pl.col(INVIEW).list.len() == pl.col(LABELS).list.len()).alias("_eq")
    )
    if not len_ok["_eq"].all():
        raise ValueError("impression inview/labels length mismatch")

    # Per-row invariants (C, D, E, B)
    for row in joined.iter_rows(named=True):
        inv = row[INVIEW]
        lab = row[LABELS]
        cands = row["candidates"]
        scores = row["scores"]
        gt = row.get("gt_clicked")

        # C. every candidate id must exist in the catalog
        for c in cands:
            if c not in catalog_set:
                raise ValueError(f"candidate article ID not in catalog: {c!r}")

        # D. candidate ids unique within each impression
        if len(set(cands)) != len(cands):
            raise ValueError("candidate row has duplicate candidate article IDs")

        # E. scores finite
        for s in scores:
            if s is None or not math.isfinite(s):
                raise ValueError("candidate scores contain NaN/Inf (must be finite)")

        # B. gt_clicked must be present (non-null) and equal click-derived
        # ground truth (insertion order, matching Q2/Q3).
        if gt is None:
            raise ValueError("candidate gt_clicked is missing or null")
        derived = [a for a, l in zip(inv, [int(x) for x in lab]) if l == 1]
        if derived != gt:
            raise ValueError(
                "candidate gt_clicked does not match click-derived ground truth "
                "(inview/labels)"
            )


def _determine_ranking(inview, labels, cand_ids, cand_scores):
    """Deterministic INVIEW ranking (spec section 4).

    Returns (scores_arr aligned with inview, ranked_labels) where ranked_labels
    is ``labels`` ordered by (-score, retrieved?, retrieval_rank, inview_pos).
    Unretrieved inview articles get score 0.0 and retrieval_rank = +inf.
    """
    score_map: dict[str, tuple[float, int]] = {}
    for r, cid in enumerate(cand_ids):
        score_map[cid] = (float(cand_scores[r]), r)
    entries = []
    scores_arr = []
    for p, a in enumerate(inview):
        lab = int(labels[p])
        if a in score_map:
            s, rr = score_map[a]
            entries.append((-s, 0, rr, p, lab))
            scores_arr.append(s)
        else:
            entries.append((0.0, 1, float("inf"), p, lab))
            scores_arr.append(0.0)
    entries.sort(key=lambda e: (e[0], e[1], e[2], e[3]))
    ranked_labels = [e[4] for e in entries]
    return scores_arr, ranked_labels


def evaluate_candidates(
    cand: pl.DataFrame,
    impr: pl.DataFrame,
    p_lookup: dict[str, float],
    id_to_cat: dict[str, str],
    catalog_set: set[str],
    catalog_cats: set[str],
    bootstrap_runs: int,
    seed: int,
    split: str,
) -> dict:
    """Core evaluation of one candidate file against impressions.

    All joins/alignment are on ``impression_row_id`` (never ``impression_id``).
    Returns a result dict with metrics, beyond-accuracy and slices.
    """
    _validate(cand, split)

    hist = _history_len(impr)
    impr_small = impr.select(
        [IMPRESSION_ROW_ID, INVIEW, LABELS]
    ).join(hist, on=IMPRESSION_ROW_ID, how="left")

    joined = cand.join(impr_small, on=IMPRESSION_ROW_ID, how="left")
    _validate_joined(joined, catalog_set, split)

    auc_vals: list[float] = []
    mrr_vals: list[float] = []
    ndcg5_vals: list[float] = []
    ndcg10_vals: list[float] = []
    div_vals: list[float] = []
    nov_vals: list[float] = []
    rec_sets: list[set] = []
    rec_cat_sets: list[set] = []
    cold_rec_sets: list[set] = []
    cold_rec_cat_sets: list[set] = []
    warm_rec_sets: list[set] = []
    warm_rec_cat_sets: list[set] = []
    cold_auc: list[float] = []
    cold_mrr: list[float] = []
    cold_n5: list[float] = []
    cold_n10: list[float] = []
    warm_auc: list[float] = []
    warm_mrr: list[float] = []
    warm_n5: list[float] = []
    warm_n10: list[float] = []
    cold_div: list[float] = []
    warm_div: list[float] = []
    cold_nov: list[float] = []
    warm_nov: list[float] = []

    n_valid_acc = 0

    for row in joined.iter_rows(named=True):
        inview = row[INVIEW]
        labels = [int(x) for x in row[LABELS]]
        cands = row["candidates"]
        scores = row["scores"]
        nh = int(row["n_history"]) if row.get("n_history") is not None else 0
        n_pos = sum(labels)

        scores_arr, ranked_labels = _determine_ranking(inview, labels, cands, scores)

        # Accuracy metrics (only when >=1 click).
        if n_pos >= 1:
            n_valid_acc += 1
            a = auc_score(scores_arr, labels)
            m = mrr_from_ranked(ranked_labels)
            n5 = ndcg_at_k_from_ranked(ranked_labels, 5)
            n10 = ndcg_at_k_from_ranked(ranked_labels, 10)
            if a is not None:
                auc_vals.append(a)
            if m is not None:
                mrr_vals.append(m)
            if n5 is not None:
                ndcg5_vals.append(n5)
            if n10 is not None:
                ndcg10_vals.append(n10)
            if nh == 0:
                if a is not None:
                    cold_auc.append(a)
                if m is not None:
                    cold_mrr.append(m)
                if n5 is not None:
                    cold_n5.append(n5)
                if n10 is not None:
                    cold_n10.append(n10)
            else:
                if a is not None:
                    warm_auc.append(a)
                if m is not None:
                    warm_mrr.append(m)
                if n5 is not None:
                    warm_n5.append(n5)
                if n10 is not None:
                    warm_n10.append(n10)

        # Beyond-accuracy over the candidate (recommended) list, all impressions.
        cset = set(cands)
        rec_sets.append(cset)
        ccats = {id_to_cat.get(c, UNKNOWN) for c in cands}
        rec_cat_sets.append(ccats)
        div = intra_list_diversity([id_to_cat.get(c, UNKNOWN) for c in cands])
        nov = novelty_for_ids(list(cands), p_lookup)
        div_vals.append(div)
        nov_vals.append(nov)
        if nh == 0:
            cold_div.append(div)
            cold_nov.append(nov)
            cold_rec_sets.append(cset)
            cold_rec_cat_sets.append(ccats)
        else:
            warm_div.append(div)
            warm_nov.append(nov)
            warm_rec_sets.append(cset)
            warm_rec_cat_sets.append(ccats)

    # Cold/warm impression counts (counted once, per impression).
    n_cold = int((joined["n_history"] == 0).sum())
    n_warm = int((joined["n_history"] > 0).sum())

    acc = {
        "auc": bootstrap_mean_ci(auc_vals, bootstrap_runs, seed),
        "mrr": bootstrap_mean_ci(mrr_vals, bootstrap_runs, seed),
        "ndcg@5": bootstrap_mean_ci(ndcg5_vals, bootstrap_runs, seed),
        "ndcg@10": bootstrap_mean_ci(ndcg10_vals, bootstrap_runs, seed),
    }
    beyond = {
        "intra_list_diversity": bootstrap_mean_ci(div_vals, bootstrap_runs, seed),
        "novelty": bootstrap_mean_ci(nov_vals, bootstrap_runs, seed),
        "article_coverage": bootstrap_coverage_ci(rec_sets, len(catalog_set), bootstrap_runs, seed),
        "category_coverage": bootstrap_coverage_ci_cats(
            rec_cat_sets, len(catalog_cats), bootstrap_runs, seed
        ),
    }
    slices = {
        "cold_start_vs_warm": {
            "cold": {
                "n": n_cold,
                "auc": bootstrap_mean_ci(cold_auc, bootstrap_runs, seed),
                "mrr": bootstrap_mean_ci(cold_mrr, bootstrap_runs, seed),
                "ndcg@5": bootstrap_mean_ci(cold_n5, bootstrap_runs, seed),
                "ndcg@10": bootstrap_mean_ci(cold_n10, bootstrap_runs, seed),
                "intra_list_diversity": bootstrap_mean_ci(cold_div, bootstrap_runs, seed),
                "novelty": bootstrap_mean_ci(cold_nov, bootstrap_runs, seed),
                "article_coverage": bootstrap_coverage_ci(
                    cold_rec_sets, len(catalog_set), bootstrap_runs, seed
                ),
                "category_coverage": bootstrap_coverage_ci_cats(
                    cold_rec_cat_sets, len(catalog_cats), bootstrap_runs, seed
                ),
            },
            "warm": {
                "n": n_warm,
                "auc": bootstrap_mean_ci(warm_auc, bootstrap_runs, seed),
                "mrr": bootstrap_mean_ci(warm_mrr, bootstrap_runs, seed),
                "ndcg@5": bootstrap_mean_ci(warm_n5, bootstrap_runs, seed),
                "ndcg@10": bootstrap_mean_ci(warm_n10, bootstrap_runs, seed),
                "intra_list_diversity": bootstrap_mean_ci(warm_div, bootstrap_runs, seed),
                "novelty": bootstrap_mean_ci(warm_nov, bootstrap_runs, seed),
                "article_coverage": bootstrap_coverage_ci(
                    warm_rec_sets, len(catalog_set), bootstrap_runs, seed
                ),
                "category_coverage": bootstrap_coverage_ci_cats(
                    warm_rec_cat_sets, len(catalog_cats), bootstrap_runs, seed
                ),
            },
        }
    }
    return {
        "n_impressions": int(joined.height),
        "n_valid_for_accuracy_metrics": n_valid_acc,
        "metrics": acc,
        "beyond_accuracy": beyond,
        "slices": slices,
    }


def _discover_methods(dset_dir: Path, split: str) -> list[tuple[str, str | None, Path]]:
    out: list[tuple[str, str | None, Path]] = []
    bm25_dir = dset_dir / "retrieval" / "bm25"
    if (bm25_dir / f"candidates_{split}.parquet").exists():
        out.append(("bm25", None, bm25_dir / f"candidates_{split}.parquet"))
    sem_dir = dset_dir / "retrieval" / "semantic"
    if sem_dir.exists():
        for sub in sorted(sem_dir.iterdir()):
            if sub.is_dir() and (sub / f"candidates_{split}.parquet").exists():
                out.append((f"semantic_{sub.name}", sub.name, sub / f"candidates_{split}.parquet"))
    return out


def _select_candidate_rows(cand: pl.DataFrame, limit: int | None) -> pl.DataFrame:
    """Deterministic selection of candidate rows.

    Sorts by ``impression_row_id`` (never parquet row order) before applying a
    ``--limit``, so the same N smallest-``impression_row_id`` rows are always
    selected. Validation is applied by the caller afterwards."""
    cand = cand.sort(IMPRESSION_ROW_ID)
    if limit is not None:
        cand = cand.head(limit)
    return cand


def run_eval(
    cfg: dict,
    datasets: list[str],
    splits: tuple[str, ...],
    embedding_override: str | None = None,
    methods_filter: list[str] | None = None,
    limit: int | None = None,
) -> dict:
    t0 = time.time()
    ev_cfg = cfg.get("evaluation", {})
    bootstrap_runs = int(ev_cfg.get("bootstrap_runs", 1000))
    seed = int(ev_cfg.get("bootstrap_seed", 42))

    base = processed_dir(cfg)
    results: dict[str, Any] = {}
    for dataset in datasets:
        dset_dir = base / dataset
        if not (dset_dir / "articles.parquet").exists():
            raise ValueError(
                f"requested dataset '{dataset}' is missing articles.parquet "
                f"(expected at {dset_dir / 'articles.parquet'})"
            )
        arts = read_df(dset_dir / "articles.parquet")
        p_lookup, id_to_cat, catalog_set, catalog_cats = _build_popularity(arts)
        impr = read_df(dset_dir / "impressions.parquet").with_row_index(IMPRESSION_ROW_ID)

        for split in splits:
            discovered = _discover_methods(dset_dir, split)
            if not discovered:
                raise ValueError(
                    f"no candidate files discovered for dataset '{dataset}' split '{split}'"
                )
            evaluated = []
            for method_name, emb, cand_path in discovered:
                if methods_filter and method_name not in methods_filter:
                    continue
                if embedding_override and emb is not None and emb != embedding_override:
                    continue
                # bm25 (emb is None) is never filtered by an embedding override
                evaluated.append((method_name, emb, cand_path))
            if methods_filter:
                available = {m for m, _, _ in discovered}
                missing = [m for m in methods_filter if m not in available]
                if missing:
                    raise ValueError(
                        f"requested methods not found for {dataset}/{split}: {missing}"
                    )
            if not evaluated:
                raise ValueError(
                    f"no candidate methods to evaluate for {dataset}/{split} "
                    f"after filters"
                )
            for method_name, emb, cand_path in evaluated:
                cand = read_df(cand_path)
                cand = _select_candidate_rows(cand, limit)
                log.info("%s/%s/%s: evaluating %d rows", dataset, split, method_name, cand.height)
                res = evaluate_candidates(
                    cand,
                    impr,
                    p_lookup,
                    id_to_cat,
                    catalog_set,
                    catalog_cats,
                    bootstrap_runs,
                    seed,
                    split,
                )
                out_dir = dset_dir / "retrieval" / "eval"
                out_dir.mkdir(parents=True, exist_ok=True)
                payload = {
                    "dataset": dataset,
                    "split": split,
                    "method": method_name,
                    "embedding": emb,
                    **res,
                }
                with open(out_dir / f"{method_name}_{split}.json", "w") as f:
                    json.dump(payload, f, indent=2, default=str)
                results.setdefault(dataset, {})[f"{method_name}_{split}"] = payload
                log.info(
                    "  %s %s auc=%.4f mrr=%.4f ndcg@10=%.4f cov=%.4f",
                    method_name,
                    split,
                    (res["metrics"]["auc"]["value"] or 0.0),
                    (res["metrics"]["mrr"]["value"] or 0.0),
                    (res["metrics"]["ndcg@10"]["value"] or 0.0),
                    (res["beyond_accuracy"]["article_coverage"]["value"] or 0.0),
                )
    log.info("eval done in %.1fs", time.time() - t0)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Q4 - offline evaluation harness")
    parser.add_argument(
        "--datasets",
        default="MIND,EB-NeRD-demo,EB-NeRD-small",
        help="comma-separated dataset dirs",
    )
    parser.add_argument("--splits", default="val", help="comma-separated splits")
    parser.add_argument(
        "--embedding",
        default=None,
        help="restrict semantic evaluation to this embedding (word2vec|bert|entity_mean)",
    )
    parser.add_argument(
        "--methods", default=None, help="comma-separated methods (e.g. bm25,semantic_word2vec)"
    )
    parser.add_argument("--limit", type=int, default=None, help="cap impressions (debug)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config()
    datasets = [d.strip() for d in args.datasets.split(",")]
    splits = tuple(s.strip() for s in args.splits.split(","))
    methods_filter = [m.strip() for m in args.methods.split(",")] if args.methods else None
    run_eval(
        cfg,
        datasets,
        splits,
        embedding_override=args.embedding,
        methods_filter=methods_filter,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
