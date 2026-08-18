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
from .bm25 import (
    Bm25Index,
    build_corpus,
    build_query_from_history,
    build_query_texts,
    recall_at_k,
    tokenize,
)

log = logging.getLogger(__name__)


def _candidates_dir(cfg: dict, dataset: str) -> Path:
    return processed_dir(cfg) / dataset / "retrieval" / "bm25"


def _gt_clicked(labels, inview):
    """Ground-truth clicked articles for an impression's inview list."""
    if len(labels) != len(inview):
        raise ValueError(
            f"inview/labels length mismatch: {len(inview)} vs {len(labels)}"
        )
    return [aid for aid, lab in zip(inview, labels) if lab == 1]


def _read_metrics(out_dir: Path, top_k: list[int]) -> dict[str, dict[str, float]]:
    """Aggregate recall@K per split from on-disk candidate parquets."""
    metrics: dict[str, dict[str, float]] = {}
    for split_name in sorted(out_dir.glob("candidates_*.parquet")):
        sn = split_name.stem.split("_", 1)[1]
        if sn not in ("val", "test"):
            continue
        sub = pl.read_parquet(split_name).filter(pl.col("gt_clicked").list.len() > 0)
        if sub.height == 0:
            continue
        metrics[sn] = {}
        for k in sorted(top_k):
            recall = sub.select(
                pl.struct(["gt_clicked", "candidates"]).map_elements(
                    lambda r: recall_at_k(r["gt_clicked"], r["candidates"], k),
                    return_dtype=pl.Float64,
                ).alias("r")
            ).select(pl.col("r").mean())
            metrics[sn][f"recall@{k}"] = round(float(recall[0, 0]), 4)
    return metrics


def run_bm25(
    cfg: dict,
    dataset: str,
    top_k: list[int],
    splits: tuple[str, ...],
    limit: int | None = None,
) -> dict:
    t0 = time.time()
    dset_dir = processed_dir(cfg) / dataset
    articles = read_df(dset_dir / "articles.parquet")
    impressions = read_df(dset_dir / "impressions.parquet")

    sel_impr = impressions.filter(pl.col(SPLIT).is_in(splits))
    if limit:
        sel_impr = sel_impr.head(limit)
    if sel_impr.height == 0:
        log.warning("%s: no impressions for splits %s", dataset, splits)
        return {"skip": True}

    k1 = cfg["retrieval"]["bm25"].get("k1", 1.5)
    b = cfg["retrieval"]["bm25"].get("b", 0.75)
    rm_sw = cfg["retrieval"]["bm25"].get("remove_stopwords", True)
    field = cfg["retrieval"]["bm25"].get("field", "title_abstract")
    query_field = cfg["retrieval"]["bm25"].get("query_field", "title")
    cap = cfg["retrieval"]["bm25"].get("history_query_cap", 20)
    max_k = max(top_k)

    corpus = build_corpus(articles, field=field, remove_stopwords=rm_sw)
    log.info("%s: indexing %d articles (%s)", dataset, articles.height, field)
    index = Bm25Index(corpus, k1=k1, b=b)
    query_texts = build_query_texts(articles, field=query_field)

    user_rows = []
    for row in sel_impr.iter_rows(named=True):
        query = build_query_from_history(
            row[HISTORY] or [],
            query_texts=query_texts,
            cap=cap,
            field=query_field,
        )
        scores, doc_idx = index.search(query, max_k)
        candidates = [articles[ARTICLE_ID][int(i)] for i in doc_idx]
        user_rows.append(
            {
                IMPRESSION_ID: row[IMPRESSION_ID],
                SPLIT: row[SPLIT],
                "gt_clicked": _gt_clicked(row[LABELS], row[INVIEW]),
                "candidates": candidates,
                "scores": scores.tolist(),
                "n_query_terms": len(tokenize(query)),
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
            "n_query_terms": pl.Int32,
        },
    )

    out_dir = _candidates_dir(cfg, dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    for (old,) in [("candidates.parquet",)]:
        p = out_dir / old
        if p.exists():
            p.unlink()
    for split_name in splits:
        sub = out.filter(pl.col(SPLIT) == split_name)
        if sub.height:
            sub.write_parquet(out_dir / f"candidates_{split_name}.parquet")

    summary = {
        "dataset": dataset,
        "index_docs": int(articles.height),
        "impressions": int(out.height),
        "k1": k1,
        "b": b,
        "field": field,
        "query_field": query_field,
        "history_query_cap": cap,
        "elastic_seconds": round(time.time() - t0, 1),
        "splits": _read_metrics(out_dir, top_k),
    }
    with open(out_dir / "recall.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Q2 - BM25 lexical candidate generation")
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
        help="comma-separated K values (default: retrieval.bm25.top_k in config)",
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
        else [int(k) for k in cfg["retrieval"]["bm25"].get("top_k", [50, 100, 200])]
    )
    splits = tuple(x.strip() for x in args.splits.split(","))
    for dataset in args.datasets.split(","):
        summary = run_bm25(cfg, dataset.strip(), top_k, splits, limit=args.limit)
        if summary.get("skip"):
            continue
        log.info("== %s ==", summary["dataset"])
        for sn, m in summary["splits"].items():
            log.info("  %-5s %s", sn, m)
    log.info("done")


if __name__ == "__main__":
    main()