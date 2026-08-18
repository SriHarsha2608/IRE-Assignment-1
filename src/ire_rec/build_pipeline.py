from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl

from . import config, dataio, download, split
from .datasets import ebnerd, mind
from .utils import clean_dir, find_file, unzip

log = logging.getLogger(__name__)
VERSION = "2"

EMBEDDINGS_KEY = "EB-NeRD-embeddings"


def _up_to_date(
    cfg: dict,
    manifest: dict,
    key: str,
    fingerprints: dict,
    files: list[str] | None = None,
    base_dir: Path | None = None,
) -> bool:
    prev = manifest.get(key)
    if not prev:
        return False
    if prev.get("fingerprints") != fingerprints:
        return False
    if files is None:
        files = [t + ".parquet" for t in ("articles", "impressions", "history")]
    base = base_dir or (config.processed_dir(cfg) / key)
    return all((base / f).exists() for f in files)


def build_mind(cfg: dict, force: bool, redownload: bool) -> dict:
    base = cfg["downloads"]["MIND"]
    raw_root = config.raw_dir(cfg) / "MIND"
    proc = config.processed_dir(cfg) / "MIND"
    archives = cfg["dataset_defaults"]["MIND"]["archives"]

    fingerprints = {}
    for arch in archives:
        download.ensure_file(base + "/" + arch, raw_root / arch, force=redownload)
        fingerprints[arch] = download.fingerprint(raw_root / arch)

    tmp = config.temp_dir(cfg) / "mind"
    clean_dir(tmp)
    for arch in archives:
        unzip(raw_root / arch, tmp / arch.replace(".zip", ""))

    news_parts = []
    beh_parts = []
    emb_parts = []
    for arch in archives:
        root = tmp / arch.replace(".zip", "")
        news_tsv = find_file(root, "news.tsv")
        if news_tsv is None:
            raise FileNotFoundError(f"news.tsv not found in {arch}")
        beh_tsv = find_file(root, "behaviors.tsv")
        if beh_tsv is None:
            raise FileNotFoundError(f"behaviors.tsv not found in {arch}")
        news_parts.append(mind.parse_mind_news(news_tsv))
        beh_parts.append(mind.parse_mind_behaviors(beh_tsv))
        vec_file = find_file(root, "entity_embedding.vec")
        if vec_file is not None:
            vec, ids = mind.parse_entity_embeddings(vec_file)
            emb_parts.append(dict(zip(ids, vec)))
        else:
            log.warning(
                "entity_embedding.vec missing in %s (entity vectors skipped for this archive)",
                arch,
            )

    articles = pl.concat(news_parts).unique(subset=dataio.ARTICLE_ID, keep="first")
    impressions = pl.concat(beh_parts)

    impressions, boundaries = split.add_temporal_split(
        impressions,
        val_days=cfg["temporal_split"]["val_days"],
        test_days=cfg["temporal_split"]["test_days"],
        fallback_ratios=tuple(cfg["temporal_split"]["fallback_ratios"]),
    )
    articles = dataio.add_popularity(articles, impressions)

    entity: dict[str, np.ndarray] = {}
    for part in emb_parts:
        for eid, vec in part.items():
            if eid in entity and not np.array_equal(entity[eid], vec):
                log.warning(
                    "entity %s has conflicting vectors across archives; keeping first",
                    eid,
                )
            entity.setdefault(eid, vec)
    entity_ids = list(entity)
    arr = np.asarray(list(entity.values()), dtype=np.float32)
    emb_mat, emb_ids = mind.build_entity_article_embeddings(arr, entity_ids, articles)

    embed_dir = proc / "embeddings"
    embed_dir.mkdir(parents=True, exist_ok=True)
    np.save(embed_dir / "entity_mean.npy", emb_mat)
    pl.DataFrame({dataio.ARTICLE_ID: pl.Series(emb_ids)}).write_parquet(
        embed_dir / "entity_mean_ids.parquet"
    )
    coverage = round(len(emb_ids) / articles.height, 4) if articles.height else 0.0

    dataio.write_df(articles, proc / "articles.parquet")
    dataio.write_df(impressions, proc / "impressions.parquet")
    dataio.write_df(mind.build_mind_history(impressions), proc / "history.parquet")

    return {
        "fingerprints": fingerprints,
        "articles": int(articles.height),
        "impressions": int(impressions.height),
        "splits": boundaries,
        "embeddings": {
            "dim": int(arr.shape[1]),
            "n_articles": int(len(emb_ids)),
            "coverage": coverage,
            "name": "entity_mean",
        },
    }


def build_ebnerd_bundle(cfg: dict, bundle: str, force: bool, redownload: bool) -> dict:
    base = cfg["downloads"]["EB-NeRD"]
    arch = cfg["dataset_defaults"]["EB-NeRD"]["archives"][bundle]
    raw_path = config.raw_dir(cfg) / "EB-NeRD" / arch
    download.ensure_file(base + "/" + arch, raw_path, force=redownload)
    fingerprints = {arch: download.fingerprint(raw_path)}

    proc = config.processed_dir(cfg) / f"EB-NeRD-{bundle}"
    tmp = config.temp_dir(cfg) / f"ebnerd_{bundle}"
    clean_dir(tmp)
    unzip(raw_path, tmp / bundle)
    base_dir = find_file(tmp / bundle, "articles.parquet")
    if base_dir is None:
        raise FileNotFoundError("articles.parquet not found in bundle archive")
    base_dir = base_dir.parent

    articles = ebnerd.parse_ebnerd_articles(base_dir / "articles.parquet")

    impressions_parts = []
    history_parts = []
    for split_name in ("train", "validation"):
        behaviors_path = base_dir / split_name / "behaviors.parquet"
        history_path = base_dir / split_name / "history.parquet"
        if not behaviors_path.exists() or not history_path.exists():
            continue
        imp, hist = ebnerd.parse_ebnerd_behaviors(
            behaviors_path, history_path, history_size=cfg["history"]["size"]
        )
        impressions_parts.append(imp)
        history_parts.append(hist)

    impressions = pl.concat(impressions_parts)
    impressions, boundaries = split.add_temporal_split(
        impressions,
        val_days=cfg["temporal_split"]["val_days"],
        test_days=cfg["temporal_split"]["test_days"],
        fallback_ratios=tuple(cfg["temporal_split"]["fallback_ratios"]),
    )
    articles = dataio.add_popularity(articles, impressions)
    history = pl.concat(history_parts)

    dataio.write_df(articles, proc / "articles.parquet")
    dataio.write_df(impressions, proc / "impressions.parquet")
    dataio.write_df(history, proc / "history.parquet")

    return {
        "fingerprints": fingerprints,
        "articles": int(articles.height),
        "impressions": int(impressions.height),
        "history_rows": int(history.height),
        "splits": boundaries,
    }


def build_ebnerd_embeddings(cfg: dict, force: bool, redownload: bool) -> dict:
    base = cfg["downloads"]["EB-NeRD"]
    emb_cfg = cfg["dataset_defaults"]["EB-NeRD"]["embeddings"]
    embed_dir = config.processed_dir(cfg) / "EB-NeRD" / "embeddings"
    artifacts = {
        "word2vec": ("Ekstra_Bladet_word2vec", "document_vector.parquet"),
        "bert": ("google_bert_base_multilingual_cased", "bert_base_multilingual_cased.parquet"),
    }
    fingerprints = {}
    results = {}
    for name, (zip_name, parquet_name) in artifacts.items():
        raw_path = config.raw_dir(cfg) / "EB-NeRD" / f"{zip_name}.zip"
        download.ensure_file(base + "/" + emb_cfg[name], raw_path, force=redownload)
        fingerprints[name] = download.fingerprint(raw_path)
        tmp = config.temp_dir(cfg) / f"emb_{name}"
        clean_dir(tmp)
        unzip(raw_path, tmp / zip_name)
        parquet_path = find_file(tmp / zip_name, parquet_name)
        if parquet_path is None:
            raise FileNotFoundError(f"{parquet_name} not found in {zip_name}.zip")
        results[name] = ebnerd.consolidate_embedding_parquet(parquet_path, embed_dir, name)
    return {"fingerprints": fingerprints, "artifacts": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="IRE Assignment 1 - data pipeline (Q1)")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="re-extract/reparse/re-split from existing raw files (one-command rebuild)",
    )
    parser.add_argument(
        "--redownload",
        action="store_true",
        help="force re-download of raw archives first",
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help="comma-separated subset of {MIND, EB-NeRD-demo, EB-NeRD-small}",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="skip EB-NeRD embedding consolidation (memory-saving; Q3 needs it later)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = config.load_config()
    build_all = args.datasets == "all"
    wanted = None if build_all else set(args.datasets.split(","))

    manifest_path = config.processed_dir(cfg) / "manifest.json"
    manifest = dataio.load_manifest(manifest_path)
    if manifest.get("version") != VERSION:
        log.info(
            "manifest version %s != current %s; treating store as stale",
            manifest.get("version"),
            VERSION,
        )
        manifest = {}
    manifest.setdefault("version", VERSION)
    started = time.time()

    if build_all or "MIND" in wanted:
        if args.rebuild or not _up_to_date(cfg, manifest, "MIND", _mind_fp(cfg)):
            log.info("building MIND")
            manifest["MIND"] = build_mind(cfg, args.rebuild, args.redownload)
            dataio.write_manifest(manifest_path, manifest)
        else:
            log.info("MIND up to date (use --rebuild to force)")

    wants_ebnerd = build_all or any(k.startswith("EB-NeRD") for k in wanted or ())
    if wants_ebnerd and not args.skip_embeddings:
        key = EMBEDDINGS_KEY
        build_emb = build_all or (wanted and "EB-NeRD-embeddings" in wanted)
        if build_emb or any(
            not _up_to_date(cfg, manifest, b, _ebnerd_fp(cfg, b)) for b in ("demo", "small")
        ):
            if args.rebuild or not _up_to_date(
                cfg,
                manifest,
                key,
                _emb_fp(cfg),
                files=["embeddings/word2vec.npy", "embeddings/bert.npy"],
                base_dir=config.processed_dir(cfg) / "EB-NeRD",
            ):
                log.info("building EB-NeRD article embeddings")
                manifest[key] = build_ebnerd_embeddings(cfg, args.rebuild, args.redownload)
                dataio.write_manifest(manifest_path, manifest)

    for bundle in ("demo", "small"):
        key = f"EB-NeRD-{bundle}"
        if build_all or key in wanted:
            if args.rebuild or not _up_to_date(cfg, manifest, key, _ebnerd_fp(cfg, bundle)):
                log.info("building %s", key)
                manifest[key] = build_ebnerd_bundle(cfg, bundle, args.rebuild, args.redownload)
                dataio.write_manifest(manifest_path, manifest)
            else:
                log.info("%s up to date (use --rebuild to force)", key)

    dataio.write_manifest(manifest_path, manifest)
    log.info("all done in %.1fs", time.time() - started)


def _mind_fp(cfg: dict) -> dict:
    raw = config.raw_dir(cfg) / "MIND"
    out = {}
    for arch in cfg["dataset_defaults"]["MIND"]["archives"]:
        p = raw / arch
        if p.exists():
            out[arch] = download.fingerprint(p)
    return out


def _ebnerd_fp(cfg: dict, bundle: str) -> dict:
    arch = cfg["dataset_defaults"]["EB-NeRD"]["archives"][bundle]
    p = config.raw_dir(cfg) / "EB-NeRD" / arch
    return {arch: download.fingerprint(p)} if p.exists() else {}


def _emb_fp(cfg: dict) -> dict:
    emb_cfg = cfg["dataset_defaults"]["EB-NeRD"]["embeddings"]
    out = {}
    for name in ("word2vec", "bert"):
        p = config.raw_dir(cfg) / "EB-NeRD" / emb_cfg[name].split("/")[-1]
        if p.exists():
            out[name] = download.fingerprint(p)
    return out


if __name__ == "__main__":
    main()