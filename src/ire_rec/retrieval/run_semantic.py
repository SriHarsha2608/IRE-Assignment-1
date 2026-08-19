from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import polars as pl

from ..config import load_config, processed_dir
from ..dataio import (
    ARTICLE_ID,
    HISTORY,
    IMPRESSION_ID,
    INVIEW,
    LABELS,
    SPLIT,
    read_df,
)
from .bm25 import recall_at_k
from .semantic import build_ann_index, load_embeddings, mean_pool_user_vector, search

log = logging.getLogger(__name__)


def _embedding_dir(cfg: dict, dataset: str, embedding: str | None) -> tuple[Path, str]:
    """Return (embeddings dir, embedding name) for a dataset."""
    if dataset == "MIND":
        return processed_dir(cfg) / "MIND" / "embeddings", "entity_mean"
    name = embedding or cfg["retrieval"]["semantic"].get("embedding", "word2vec")
    return processed_dir(cfg) / "EB-NeRD" / "embeddings", name


def _candidates_dir(cfg: dict, dataset: str, embedding: str) -> Path:
    return processed_dir(cfg) / dataset / "retrieval" / "semantic" / embedding


def _gt_clicked(labels, inview):
    """Ground-truth clicked articles for an impression's inview list."""
    if len(labels) != len(inview):
        raise ValueError(
            f"inview/labels length mismatch: {len(inview)} vs {len(labels)}"
        )
    return [aid for aid, lab in zip(inview, labels) if lab == 1]


def _mean_recall(df: pl.DataFrame, k: int) -> float | None:
    """Mean recall@k over ``df`` (which must have gt_clicked/candidates cols)."""
    if df.height == 0:
        return None
    out = df.select(
        pl.struct(["gt_clicked", "candidates"]).map_elements(
            lambda r: recall_at_k(r["gt_clicked"], r["candidates"], k),
            return_dtype=pl.Float64,
        ).alias("r")
    ).select(pl.col("r").mean())
    return round(float(out[0, 0]), 4)


def _read_metrics(out_dir: Path, top_k: list[int]) -> dict[str, dict[str, float]]:
    """Aggregate recall@K per split from on-disk candidate parquets."""
    metrics: dict[str, dict[str, float]] = {}
    for split_file in sorted(out_dir.glob("candidates_*.parquet")):
        sn = split_file.stem.split("_", 1)[1]
        if sn not in ("val", "test"):
            continue
        sub = pl.read_parquet(split_file).filter(pl.col("gt_clicked").list.len() > 0)
        if sub.height == 0:
            continue
        metrics[sn] = {}
        for k in sorted(top_k):
            r = _mean_recall(sub, k)
            if r is not None:
                metrics[sn][f"recall@{k}"] = r
    return metrics


def _cold_warm_recall(df: pl.DataFrame, k: int) -> dict:
    """recall@k sliced by history length (0 = cold start)."""
    sub = df.filter(pl.col("gt_clicked").list.len() > 0)
    cold = sub.filter(pl.col("n_history") == 0)
    warm = sub.filter(pl.col("n_history") > 0)
    return {
        "recall@K": k,
        "cold": _mean_recall(cold, k),
        "warm": _mean_recall(warm, k),
        "n_cold": int(cold.height),
        "n_warm": int(warm.height),
    }


def _semantic_cold_warm(out_dir: Path, k: int) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for split_file in sorted(out_dir.glob("candidates_*.parquet")):
        sn = split_file.stem.split("_", 1)[1]
        if sn not in ("val", "test"):
            continue
        df = pl.read_parquet(split_file).unique(
            subset=IMPRESSION_ID, keep="first"
        )
        out[sn] = _cold_warm_recall(df, k)
    return out


def _bm25_cold_warm(bm25_dir: Path, impressions: pl.DataFrame, k: int) -> dict[str, dict]:
    """Cold/warm recall@k from BM25 candidates (join history length).

    Some MIND dev rows reuse the same ``impression_id`` for genuinely distinct
    impressions, so both sides are deduplicated on ``impression_id`` (keep
    first) before the join to avoid row multiplication.  The history map is
    scoped to the split being sliced so ids that also appear in other splits
    resolve to their own split's history.
    """
    out: dict[str, dict] = {}
    for split_file in sorted(bm25_dir.glob("candidates_*.parquet")):
        sn = split_file.stem.split("_", 1)[1]
        if sn not in ("val", "test"):
            continue
        hist_map = (
            impressions.filter(pl.col(SPLIT) == sn)
            .select([IMPRESSION_ID, HISTORY])
            .unique(subset=IMPRESSION_ID, keep="first")
            .with_columns(pl.col(HISTORY).fill_null([]).list.len().alias("n_history"))
        )
        df = (
            pl.read_parquet(split_file)
            .unique(subset=IMPRESSION_ID, keep="first")
            .join(hist_map, on=IMPRESSION_ID, how="left")
        )
        out[sn] = _cold_warm_recall(df, k)
    return out


def _compare_bm25(
    cfg: dict,
    dataset: str,
    sem_dir: Path,
    impressions: pl.DataFrame,
    top_k: list[int],
    slice_k: int,
) -> dict:
    bm25_dir = processed_dir(cfg) / dataset / "retrieval" / "bm25"
    result: dict = {"top_k": top_k, "slice_k": slice_k}
    if not bm25_dir.exists():
        result["bm25"] = None
        return result

    def _load_splits(dir_: Path) -> dict:
        p = dir_ / "recall.json"
        if p.exists():
            with open(p) as f:
                return json.load(f).get("splits", {})
        return _read_metrics(dir_, top_k)

    result["bm25"] = {
        "recall": _load_splits(bm25_dir),
        "cold_vs_warm": _bm25_cold_warm(bm25_dir, impressions, slice_k),
    }
    result["semantic"] = {
        "recall": _load_splits(sem_dir),
        "cold_vs_warm": _semantic_cold_warm(sem_dir, slice_k),
    }
    return result


def run_semantic(
    cfg: dict,
    dataset: str,
    top_k: list[int],
    splits: tuple[str, ...],
    limit: int | None = None,
    embedding: str | None = None,
) -> dict:
    t0 = time.time()
    dset_dir = processed_dir(cfg) / dataset
    articles = read_df(dset_dir / "articles.parquet")
    impressions = read_df(dset_dir / "impressions.parquet")

    emb_dir, emb_name = _embedding_dir(cfg, dataset, embedding)
    normalize = cfg["retrieval"]["semantic"].get("normalize", True)
    ann = cfg["retrieval"]["semantic"].get("ann", "faiss")

    sel_impr = impressions.filter(pl.col(SPLIT).is_in(splits))
    if limit:
        sel_impr = sel_impr.head(limit)
    if sel_impr.height == 0:
        log.warning("%s: no impressions for splits %s", dataset, splits)
        return {"skip": True}

    catalog = set(articles[ARTICLE_ID].to_list())
    mat, ids = load_embeddings(emb_dir, emb_name, catalog=catalog)
    if mat.shape[0] == 0:
        raise RuntimeError(f"no {emb_name} embeddings for {dataset}")
    id_to_row = {aid: i for i, aid in enumerate(ids)}
    index = build_ann_index(mat, normalize=normalize)
    coverage = round(mat.shape[0] / articles.height, 4)
    max_k = max(top_k)
    log.info(
        "%s [%s]: indexing %d/%d articles (dim %d, coverage %s)",
        dataset,
        emb_name,
        mat.shape[0],
        articles.height,
        mat.shape[1],
        coverage,
    )

    user_rows = []
    for row in sel_impr.iter_rows(named=True):
        history = row[HISTORY] or []
        user_vec, n_used = mean_pool_user_vector(history, id_to_row, mat)
        if user_vec is None:
            candidates: list[str] = []
            scores: list[float] = []
        else:
            scores_arr, doc_idx = search(user_vec, index, max_k)
            candidates = [ids[int(i)] for i in doc_idx]
            scores = scores_arr.tolist()
        user_rows.append(
            {
                IMPRESSION_ID: row[IMPRESSION_ID],
                SPLIT: row[SPLIT],
                "gt_clicked": _gt_clicked(row[LABELS], row[INVIEW]),
                "candidates": candidates,
                "scores": scores,
                "n_history_used": n_used,
                "n_history": len(history),
            }
        )

    out = pl.DataFrame(
        user_rows,
        schema={
            IMPRESSION_ID: pl.String,
            SPLIT: pl.String,
            "gt_clicked": pl.List(pl.String),
            "candidates": pl.List(pl.String),
            "scores": pl.List(pl.Float64),
            "n_history_used": pl.Int32,
            "n_history": pl.Int32,
        },
    )

    out_dir = _candidates_dir(cfg, dataset, emb_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name in splits:
        sub = out.filter(pl.col(SPLIT) == split_name)
        if sub.height:
            sub.write_parquet(out_dir / f"candidates_{split_name}.parquet")

    slice_k = max(top_k)
    summary = {
        "dataset": dataset,
        "embedding": emb_name,
        "index_docs": int(mat.shape[0]),
        "article_catalog": int(articles.height),
        "embedding_coverage": coverage,
        "impressions": int(out.height),
        "ann": ann,
        "user_repr": "mean_pool",
        "normalize": normalize,
        "elastic_seconds": round(time.time() - t0, 1),
        "splits": _read_metrics(out_dir, top_k),
        "cold_vs_warm_recall@%d" % slice_k: _semantic_cold_warm(out_dir, slice_k),
    }
    with open(out_dir / "recall.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    comparison = _compare_bm25(cfg, dataset, out_dir, impressions, top_k, slice_k)
    with open(out_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Q3 - embedding/ANN semantic candidate generation"
    )
    parser.add_argument(
        "--datasets",
        default="MIND,EB-NeRD-demo,EB-NeRD-small",
        help="comma-separated dataset dirs",
    )
    parser.add_argument(
        "--splits",
        default="val",
        help="comma-separated split names to run (default: val)",
    )
    parser.add_argument(
        "--top-k",
        default=None,
        help="comma-separated K values (default: retrieval.semantic.top_k in config)",
    )
    parser.add_argument(
        "--embedding",
        default=None,
        help="EB-NeRD embedding name: word2vec | bert (MIND always uses entity_mean)",
    )
    parser.add_argument("--limit", type=int, default=None, help="cap impressions (debug)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config()
    top_k = (
        [int(x.strip()) for x in args.top_k.split(",")]
        if args.top_k
        else [int(k) for k in cfg["retrieval"]["semantic"].get("top_k", [50, 100, 200])]
    )
    splits = tuple(x.strip() for x in args.splits.split(","))
    for dataset in args.datasets.split(","):
        summary = run_semantic(
            cfg, dataset.strip(), top_k, splits, limit=args.limit, embedding=args.embedding
        )
        if summary.get("skip"):
            continue
        log.info("== %s [%s] ==", summary["dataset"], summary["embedding"])
        for sn, m in summary["splits"].items():
            log.info("  %-5s %s", sn, m)
    log.info("done")


if __name__ == "__main__":
    main()