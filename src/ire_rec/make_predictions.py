"""Q5: generate Codabench prediction files.

For every requested dataset we rank each *test*-split impression's ``inview``
articles by a normalized fusion of the BM25 and semantic candidate scores, then
write the prediction in the format expected by each competition. Both
competitions require a **rank permutation** over the impression's ``inview``
list: for each impression we emit the 1-based rank of every article in the
order those articles appear in ``inview`` (NOT a reordered id list). E.g. if
``inview == [A, B, C]`` and our ranking puts B first, C second, A third, the
line for that impression is ``[3,1,2]``.

Output formats:

  * MIND (Codabench 13967): TSV, **no header**, one line per impression:
    ``<impression_id><TAB>[<rank_1>,<rank_2>,...,<rank_n>]``.
  * EB-NeRD / RecSys 2024 (Codabench 2469): CSV with header
    ``impression_id,prediction`` where prediction is the same
    ``[<rank_1>,<rank_2>,...]`` rank-permutation string.

The fusion normalizes each method's raw scores to [0, 1] (per impression,
min-max) so the two retrieval signals are comparable despite different scales;
articles that a method did not retrieve get a component of ``-1`` so retrieved
articles always outrank unretrieved ones. Final ranking ties are broken by the
original inview order for determinism.

The candidate frames are joined to the impressions on ``impression_row_id`` and
the per-impression fusion runs row-wise, so memory stays bounded to the size of
the candidate files (no giant intermediate dicts) -- important on small laptops.

Run via ``make predict`` (``python -m ire_rec.make_predictions``).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import polars as pl

from ire_rec import config as cfg_mod
from ire_rec.dataio import (
    IMPRESSION_ID,
    IMPRESSION_ROW_ID,
    INVIEW,
    SPLIT,
)

log = logging.getLogger("ire_rec.make_predictions")

CANDIDATES = "candidates"
SCORES = "scores"
BM25_C = "bm25_candidates"
BM25_S = "bm25_scores"
SEM_C = "sem_candidates"
SEM_S = "sem_scores"
RANKED = "ranked"
DEFAULT_DATASETS = ["MIND", "EB-NeRD-demo", "EB-NeRD-small"]


def _add_row_ids(impr: pl.DataFrame) -> pl.DataFrame:
    """Mirror run_bm25/run_semantic: row id is the index of the FULL frame,
    derived before any split filter, so it lines up with the candidate files."""
    if IMPRESSION_ROW_ID in impr.columns:
        return impr
    return impr.with_row_index(IMPRESSION_ROW_ID)


def _component(present: dict[str, float], inview: list[str], default: float = -1.0) -> dict[str, float]:
    """Min-max normalize ``present`` scores into [0, 1]; articles absent from
    ``present`` (unretrieved by this method) get ``default`` so retrieved
    articles always outrank unretrieved ones."""
    vals = list(present.values())
    if not vals:
        return {a: default for a in inview}
    lo, hi = min(vals), max(vals)
    rng = hi - lo
    norm = {a: (0.0 if rng == 0 else (s - lo) / rng) for a, s in present.items()}
    return {a: norm.get(a, default) for a in inview}


def _rank_inview(inview: list[str], fused: list[float]) -> list[int]:
    """Return the 1-based rank of each inview article by ``fused`` score
    (rank 1 = highest score), aligned with ``inview`` order. Competitions
    require a rank permutation over the inview list, not a reordered id list.
    Ties are broken by the original inview order for determinism."""
    order = sorted(range(len(inview)), key=lambda i: (-fused[i], i))
    ranks = [0] * len(inview)
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank
    return ranks


def _fuse_row(inview, bm25_c, bm25_s, sem_c, sem_s, w_bm25, w_sem) -> list[str]:
    if not inview:
        return []
    inview = [str(a) for a in inview]
    b_present = {str(a): float(s) for a, s in zip(bm25_c or [], bm25_s or [])}
    s_present = {str(a): float(s) for a, s in zip(sem_c or [], sem_s or [])}
    b_comp = _component(b_present, inview)
    s_comp = _component(s_present, inview)
    fused = [w_bm25 * b_comp[a] + w_sem * s_comp[a] for a in inview]
    return _rank_inview(inview, fused)


def _sem_emb(dataset: str, embedding: str | None, config: dict[str, Any]) -> str:
    if dataset.startswith("MIND"):
        return "entity_mean"  # MIND (incl. MIND-large) only has entity_mean semantic embeddings
    return embedding or config.get("retrieval", {}).get("semantic", {}).get("embedding", "word2vec")


def _fmt_for(dataset: str) -> str:
    return "mind" if dataset.startswith("MIND") else "ebnerd"


def _write_predictions(out_file: Path, rows: list[tuple[str, list[int]]], fmt: str) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "mind":
        text = "\n".join(
            f"{iid}\t[{','.join(str(r) for r in ranks)}]" for iid, ranks in rows
        )
    else:  # ebnerd: CSV with header
        text = "impression_id,prediction\n" + "\n".join(
            f"{iid},[{','.join(str(r) for r in ranks)}]" for iid, ranks in rows
        )
    out_file.write_text(text + "\n")


def make_predictions_for_dataset(
    base: Path,
    dataset: str,
    split: str,
    embedding: str,
    w_bm25: float,
    w_sem: float,
    out_root: Path | None = None,
) -> dict[str, Any]:
    dset_dir = base / dataset
    impr_path = dset_dir / "impressions.parquet"
    if not impr_path.exists():
        raise ValueError(
            f"requested dataset '{dataset}' is missing impressions.parquet "
            f"(expected at {impr_path})"
        )
    bm25_path = dset_dir / "retrieval" / "bm25" / f"candidates_{split}.parquet"
    sem_path = dset_dir / "retrieval" / "semantic" / embedding / f"candidates_{split}.parquet"

    impr = _add_row_ids(pl.read_parquet(impr_path)).filter(pl.col(SPLIT) == split)
    bm25 = (
        pl.read_parquet(bm25_path, columns=[IMPRESSION_ROW_ID, CANDIDATES, SCORES])
        .rename({CANDIDATES: BM25_C, SCORES: BM25_S})
        if bm25_path.exists()
        else None
    )
    sem = (
        pl.read_parquet(sem_path, columns=[IMPRESSION_ROW_ID, CANDIDATES, SCORES])
        .rename({CANDIDATES: SEM_C, SCORES: SEM_S})
        if sem_path.exists()
        else None
    )
    if bm25 is None:
        log.warning("%s/%s: no BM25 candidate file at %s; BM25 contributes nothing", dataset, split, bm25_path)
    if sem is None:
        log.warning("%s/%s: no semantic(%s) candidate file at %s; semantic contributes nothing", dataset, split, embedding, sem_path)

    joined = impr
    if bm25 is not None:
        joined = joined.join(bm25, on=IMPRESSION_ROW_ID, how="left")
    if sem is not None:
        joined = joined.join(sem, on=IMPRESSION_ROW_ID, how="left")

    # Guarantee the four candidate columns exist even when a method file was
    # absent (left join did not add them), so the fusion struct is well-formed.
    for col, dtype in [
        (BM25_C, pl.List(pl.String)),
        (BM25_S, pl.List(pl.Float64)),
        (SEM_C, pl.List(pl.String)),
        (SEM_S, pl.List(pl.Float64)),
    ]:
        if col not in joined.columns:
            joined = joined.with_columns(pl.lit(None, dtype=dtype).alias(col))

    ranked = joined.with_columns(
        pl.struct([INVIEW, BM25_C, BM25_S, SEM_C, SEM_S]).map_elements(
            lambda r: _fuse_row(
                r[INVIEW], r[BM25_C], r[BM25_S], r[SEM_C], r[SEM_S], w_bm25, w_sem
            ),
            return_dtype=pl.List(pl.Int64),
        ).alias(RANKED)
    )

    rows: list[tuple[str, list[int]]] = []
    dup_ids = 0
    seen_ids: set[str] = set()
    for row in ranked.iter_rows(named=True):
        ranks = row[RANKED] or []
        if not ranks:
            continue
        iid = str(row[IMPRESSION_ID])
        if iid in seen_ids:
            dup_ids += 1
        seen_ids.add(iid)
        rows.append((iid, ranks))

    fmt = _fmt_for(dataset)
    out_root = out_root or (cfg_mod.repo_root() / "predictions")
    out_file = out_root / dataset / (
        "prediction.txt" if fmt == "mind" else "prediction.csv"
    )
    _write_predictions(out_file, rows, fmt)

    meta = {
        "dataset": dataset,
        "split": split,
        "embedding": embedding,
        "format": fmt,
        "weights": {"bm25": w_bm25, "semantic": w_sem},
        "n_impressions": len(rows),
        "duplicate_impression_ids": dup_ids,
        "output": str(out_file),
    }
    (out_file.parent / "meta.json").write_text(json.dumps(meta, indent=2))
    if dup_ids:
        log.warning(
            "%s/%s: wrote %d predictions with %d duplicate impression_id values "
            "(multiple rows share an id; output is one line per impression row)",
            dataset, split, len(rows), dup_ids,
        )
    return meta


def run_predict(config: dict[str, Any], datasets: list[str] | None, split: str,
                embedding: str | None, w_bm25: float, w_sem: float,
                out_root: Path | None = None) -> dict[str, Any]:
    base = cfg_mod.processed_dir(config)
    cfg_datasets = config.get("datasets", {})
    if isinstance(cfg_datasets, dict):
        cfg_datasets = cfg_datasets.get("order", DEFAULT_DATASETS)
    datasets = datasets or cfg_datasets or DEFAULT_DATASETS
    results = {}
    t0 = time.time()
    for dataset in datasets:
        emb = _sem_emb(dataset, embedding, config)
        log.info("%s/%s: generating predictions (semantic=%s, weights bm25=%.2f sem=%.2f)",
                 dataset, split, emb, w_bm25, w_sem)
        meta = make_predictions_for_dataset(base, dataset, split, emb, w_bm25, w_sem, out_root)
        results[dataset] = meta
        log.info("  -> %s (%d impressions)", meta["output"], meta["n_impressions"])
    log.info("predict done in %.1fs", time.time() - t0)
    return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Q5: generate Codabench prediction files")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--datasets", type=str, default=None,
                   help="comma-separated dataset names (default: MIND,EB-NeRD-demo,EB-NeRD-small)")
    p.add_argument("--split", type=str, default="test", help="split to predict (default: test)")
    p.add_argument("--embedding", type=str, default=None,
                   help="EB-NeRD semantic embedding override (default: config retrieval.semantic.embedding)")
    p.add_argument("--w-bm25", type=float, default=0.5, help="BM25 fusion weight (default 0.5)")
    p.add_argument("--w-semantic", type=float, default=0.5, help="semantic fusion weight (default 0.5)")
    p.add_argument("--out", type=str, default=None,
                   help="output root dir for prediction files (default: <repo>/predictions)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = cfg_mod.load_config(Path(args.config) if args.config else None)
    config = cfg_mod.load_config(Path(args.config) if args.config else None)
    datasets = args.datasets.split(",") if args.datasets else None
    out_root = Path(args.out) if args.out else None
    return run_predict(config, datasets, args.split, args.embedding, args.w_bm25, args.w_semantic, out_root)


if __name__ == "__main__":
    main()
