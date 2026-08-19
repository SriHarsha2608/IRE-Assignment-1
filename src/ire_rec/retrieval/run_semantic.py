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
    IMPRESSION_ROW_ID,
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
    """Cold/warm recall@k per split from semantic candidate parquets.

    Rows are already unique per impression row (``impression_row_id`` is
    unique, unlike ``impression_id`` which MIND reuses across distinct rows),
    so no deduplication is applied here.
    """
    out: dict[str, dict] = {}
    for split_file in sorted(out_dir.glob("candidates_*.parquet")):
        sn = split_file.stem.split("_", 1)[1]
        if sn not in ("val", "test"):
            continue
        df = pl.read_parquet(split_file)
        out[sn] = _cold_warm_recall(df, k)
    return out


def _bm25_cold_warm(bm25_dir: Path, impressions: pl.DataFrame, k: int) -> dict[str, dict]:
    """Cold/warm recall@k from BM25 candidates (join history length).

    Both sides are keyed by ``impression_row_id`` (a stable row-level
    identifier carried in the candidate parquets), so MIND's reused
    ``impression_id`` values cannot multiply rows or misalign them.  The
    history map is scoped to the split being sliced.
    """
    out: dict[str, dict] = {}
    for split_file in sorted(bm25_dir.glob("candidates_*.parquet")):
        sn = split_file.stem.split("_", 1)[1]
        if sn not in ("val", "test"):
            continue
        hist_map = (
            impressions.filter(pl.col(SPLIT) == sn)
            .select([IMPRESSION_ROW_ID, HISTORY])
            .with_columns(pl.col(HISTORY).fill_null([]).list.len().alias("n_history"))
        )
        df = pl.read_parquet(split_file).join(hist_map, on=IMPRESSION_ROW_ID, how="left")
        out[sn] = _cold_warm_recall(df, k)
    return out


def _gt_all_covered(covered: set[str]) -> pl.Expr:
    """Row filter: impression's ground-truth clicks are all embedding-covered."""
    return pl.struct(["gt_clicked"]).map_elements(
        lambda r: all(a in covered for a in r["gt_clicked"]),
        return_dtype=pl.Boolean,
    )


def _fair_compare(
    bm25_dir: Path,
    sem_dir: Path,
    covered: set[str],
    catalog_n: int,
    top_k: list[int],
) -> dict:
    """Fair BM25-vs-semantic recall on the embedding-covered GT population.

    Direct recall@K is unfair when the two methods search different candidate
    universes: semantic retrieval can only ever retrieve articles that have an
    embedding, so ground-truth clicks on unembedded articles are unreachable
    for it.  To make the ceiling equal for both methods, recall is reported on
    impressions whose ground-truth clicked articles are ALL embedding-covered
    (``gt_clicked subseteq covered``).  Candidate universes still differ (BM25
    searches the full catalog, semantic searches the covered subset) and that
    difference is reported explicitly via ``coverage``.

    Both methods are evaluated on the SAME impression population: candidate
    files are intersected on ``impression_row_id`` (a stable row-level
    identifier, since MIND reuses some ``impression_id`` values across distinct
    rows) before the gt-coverage filter, so a partial/--limit run on one side
    cannot silently change the other side's recall population.
    ``n_gt_nonempty`` is the number of common impression rows with >=1 click;
    ``n_fair``/``n_fair_bm`` are the row counts of the gt-covered subset (equal
    when the two files carry the same rows for the common impressions).
    """
    out: dict = {
        "coverage": round(len(covered) / catalog_n, 4) if catalog_n else 0.0,
        "n_covered_articles": len(covered),
        "n_catalog_articles": catalog_n,
        "population": (
            "impressions with >=1 click whose gt_clicked are all "
            "embedding-covered (equal recall ceiling for both methods)"
        ),
        "note": (
            "BM25 searched the full article catalog; semantic searched only "
            "embedding-covered articles. Coverage is reported above so the "
            "smaller semantic candidate universe is not hidden. Recall is "
            "computed on impressions present in BOTH candidate files "
            "(intersected on the row-level impression_row_id) whose gt_clicked "
            "are all embedding-covered."
        ),
        "splits": {},
    }
    bm25_files = {
        f.stem.split("_", 1)[1]: f for f in bm25_dir.glob("candidates_*.parquet")
    } if bm25_dir.exists() else {}
    sem_files = {
        f.stem.split("_", 1)[1]: f for f in sem_dir.glob("candidates_*.parquet")
    } if sem_dir.exists() else {}
    for sn in sorted(set(bm25_files) | set(sem_files)):
        if sn not in ("val", "test"):
            continue
        if sn not in sem_files:
            continue
        sem_df = pl.read_parquet(sem_files[sn]).filter(
            pl.col("gt_clicked").list.len() > 0
        )
        common = set(sem_df[IMPRESSION_ROW_ID].to_list())
        bm_df = None
        if sn in bm25_files:
            bm_df = pl.read_parquet(bm25_files[sn]).filter(
                pl.col("gt_clicked").list.len() > 0
            )
            common &= set(bm_df[IMPRESSION_ROW_ID].to_list())
        # Both methods must be scored on the same impression ROWS: intersect on
        # the row-level identifier before filtering on gt coverage.
        sem_df = sem_df.filter(pl.col(IMPRESSION_ROW_ID).is_in(common))
        fair = sem_df.filter(_gt_all_covered(covered))
        entry: dict = {
            "n_gt_nonempty": len(common),
            "n_fair": int(fair.height),
            "semantic": {f"recall@{k}": _mean_recall(fair, k) for k in sorted(top_k)},
        }
        if bm_df is not None:
            fair_bm = bm_df.filter(pl.col(IMPRESSION_ROW_ID).is_in(common)).filter(
                _gt_all_covered(covered)
            )
            entry["n_fair_bm"] = int(fair_bm.height)
            entry["bm25"] = {
                f"recall@{k}": _mean_recall(fair_bm, k) for k in sorted(top_k)
            }
        out["splits"][sn] = entry
    return out


def _compare_bm25(
    dset_dir: Path,
    sem_dir: Path,
    impressions: pl.DataFrame,
    covered: set[str],
    catalog_n: int,
    top_k: list[int],
    slice_k: int,
) -> dict:
    bm25_dir = dset_dir / "retrieval" / "bm25"
    result: dict = {
        "top_k": top_k,
        "slice_k": slice_k,
        "fair": _fair_compare(bm25_dir, sem_dir, covered, catalog_n, top_k),
    }
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
    dset_dir: Path | None = None,
    emb_dir: Path | None = None,
) -> dict:
    t0 = time.time()
    dset_dir = Path(dset_dir) if dset_dir else processed_dir(cfg) / dataset
    articles = read_df(dset_dir / "articles.parquet")
    # Derive a stable row-level identifier from the FULL impressions frame
    # (before any split filter / --limit) so candidate rows align exactly with
    # the source impression row -- even when impression_id is reused (MIND dev).
    impressions = read_df(dset_dir / "impressions.parquet").with_row_index(
        IMPRESSION_ROW_ID
    )

    default_emb_dir, default_name = _embedding_dir(cfg, dataset, embedding)
    emb_dir = Path(emb_dir) if emb_dir else default_emb_dir
    emb_name = embedding or default_name
    normalize = cfg["retrieval"]["semantic"].get("normalize", True)

    sel_impr = impressions.filter(pl.col(SPLIT).is_in(splits))
    if limit is not None:
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
        user_vec, n_used = mean_pool_user_vector(
            history, id_to_row, mat, normalize=normalize
        )
        if user_vec is None:
            candidates: list[str] = []
            scores: list[float] = []
        else:
            scores_arr, doc_idx = search(user_vec, index, max_k)
            candidates = [ids[int(i)] for i in doc_idx]
            scores = scores_arr.tolist()
        user_rows.append(
            {
                IMPRESSION_ROW_ID: row[IMPRESSION_ROW_ID],
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
            IMPRESSION_ROW_ID: pl.UInt32,
            IMPRESSION_ID: pl.String,
            SPLIT: pl.String,
            "gt_clicked": pl.List(pl.String),
            "candidates": pl.List(pl.String),
            "scores": pl.List(pl.Float64),
            "n_history_used": pl.Int32,
            "n_history": pl.Int32,
        },
    )

    out_dir = dset_dir / "retrieval" / "semantic" / emb_name
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear candidate files first so a stale candidates_*.parquet from an
    # earlier run (e.g. a --limit debug run) can never be mixed with the
    # current invocation's results: recall.json / cold-warm / comparison.json
    # always correspond to exactly one coherent run.  Run --splits val,test
    # together to get both splits.
    for old in out_dir.glob("candidates*.parquet"):
        old.unlink()
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
        "ann": "IndexFlatIP",
        "user_repr": "mean_pool",
        "normalize": normalize,
        "elastic_seconds": round(time.time() - t0, 1),
        "splits": _read_metrics(out_dir, top_k),
        "cold_vs_warm_recall@%d" % slice_k: _semantic_cold_warm(out_dir, slice_k),
    }
    with open(out_dir / "recall.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    comparison = _compare_bm25(
        dset_dir, out_dir, impressions, set(ids), articles.height, top_k, slice_k
    )
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
    if not top_k or any(k <= 0 for k in top_k):
        parser.error("top_k values must be positive integers")
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