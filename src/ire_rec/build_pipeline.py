from __future__ import annotations

import argparse
import hashlib
import json
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

# Files the MIND pipeline promises to produce; the up-to-date check requires
# every one of them (Q3 depends on the entity_mean embedding outputs).
MIND_OUTPUT_FILES = [
    "articles.parquet",
    "impressions.parquet",
    "history.parquet",
    "embeddings/entity_mean.npy",
    "embeddings/entity_mean_ids.parquet",
]

# Files the EB-NeRD embedding stage promises to produce; the up-to-date check
# requires every one of them (the Q3 loader needs both the .npy matrices and
# the matching *_ids.parquet row-order files).
EMBEDDING_OUTPUT_FILES = [
    "embeddings/word2vec.npy",
    "embeddings/word2vec_ids.parquet",
    "embeddings/bert.npy",
    "embeddings/bert_ids.parquet",
]

# Top-level config sections that materially affect the generated feature store.
# Changing any of these invalidates the pipeline cache.
CONFIG_SIGNATURE_SECTIONS = ("downloads", "dataset_defaults", "temporal_split", "history")


def _config_signature(cfg: dict) -> str:
    """Stable hash of the config values that affect pipeline outputs.

    Only behavior-affecting sections are included (download URLs, dataset
    defaults/archives, temporal split parameters, history size).  Retrieval
    config is excluded because it does not change the feature store.  The
    pipeline ``VERSION`` is folded in, so bumping it (a parser/schema
    implementation change) invalidates every stage's cache.  The hash is
    deterministic and independent of dict insertion order.
    """
    relevant = {k: cfg.get(k, {}) for k in CONFIG_SIGNATURE_SECTIONS}
    blob = json.dumps({"version": VERSION, "config": relevant}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


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
    if prev.get("config_hash") != _config_signature(cfg):
        return False
    if files is None:
        files = [t + ".parquet" for t in ("articles", "impressions", "history")]
    base = base_dir or (config.processed_dir(cfg) / key)
    return all((base / f).exists() for f in files)


def _write_manifest(manifest_path: Path, manifest: dict, cfg: dict, key: str) -> None:
    """Persist the manifest, recording the config that produced one stage.

    The config hash is stored on the stage entry itself (``manifest[key]``),
    not globally, so a partial rebuild of one dataset never refreshes the cache
    status of other stages that were built under an older config.
    """
    manifest[key]["config_hash"] = _config_signature(cfg)
    dataio.write_manifest(manifest_path, manifest)


def _require_mind_entity_vectors(emb_parts: list[dict], archives: list[str]) -> None:
    """Fail clearly if no MIND archive provides Wikidata entity vectors.

    The Q3 entity_mean article embeddings are built from these vectors; without
    them there is nothing to pool and the store would be incomplete.  A hard
    error with an actionable message is preferable to a confusing downstream
    shape error, and we must never fabricate embeddings.
    """
    if not emb_parts:
        raise RuntimeError(
            "MIND entity embeddings unavailable: entity_embedding.vec was not "
            f"found in any MIND archive ({', '.join(archives)}). The entity_mean "
            "article embeddings that Q3 depends on cannot be built. Provide the "
            "entity vector file (or an archive that includes it)."
        )


def dataset_spec(cfg: dict, name: str) -> dict:
    """Central dataset configuration mapping (single source of truth).

    Resolves a dataset key to its kind, raw directory, processed output
    directory, raw archive filename(s), and embedding location, so dataset-
    specific path logic is not scattered across the pipeline."""
    if name.startswith("MIND"):
        dd = cfg["dataset_defaults"][name]
        return {
            "kind": "mind",
            "raw_root": config.raw_dir(cfg) / "MIND",
            "proc": config.processed_dir(cfg) / name,
            "archives": list(dd["archives"]),
            "embeddings": ("entity_mean", config.processed_dir(cfg) / name / "embeddings"),
        }
    if name == "EB-NeRD-embeddings":
        return {
            "kind": "ebnerd-embeddings",
            "proc": config.processed_dir(cfg) / "EB-NeRD" / "embeddings",
        }
    if name.startswith("EB-NeRD-"):
        bundle = name.split("EB-NeRD-", 1)[1]
        dd = cfg["dataset_defaults"]["EB-NeRD"]
        return {
            "kind": "ebnerd",
            "raw_root": config.raw_dir(cfg) / "EB-NeRD",
            "proc": config.processed_dir(cfg) / name,
            "archives": [dd["archives"][bundle]],
            "embeddings": ("shared", config.processed_dir(cfg) / "EB-NeRD" / "embeddings"),
        }
    raise ValueError(f"unknown dataset key: {name}")


def _gated_hint(label: str) -> str:
    if label.startswith("MIND"):
        return (
            " The official MIND dataset on Hugging Face (yjw1029/MIND) is gated: "
            "obtain access, authenticate externally (e.g. `huggingface-cli login`), "
            "and place the official archive(s) in the raw directory, then re-run. "
            "Do not substitute an unrelated mirror for the official assignment dataset."
        )
    return " Supply the official raw archive in the raw directory, then re-run."


def _ensure_archives(cfg: dict, spec: dict, force: bool, redownload: bool, label: str) -> dict:
    """Download/verify raw archives for a dataset spec, failing clearly.

    Authentication/data availability is treated as an external prerequisite: if an
    archive is missing and cannot be fetched, raise a RuntimeError with an
    actionable message (including the gated-dataset note for MIND)."""
    base = cfg["downloads"]["MIND"] if spec["kind"] == "mind" else cfg["downloads"]["EB-NeRD"]
    raw_root = spec["raw_root"]
    fps: dict[str, int] = {}
    for arch in spec["archives"]:
        dest = raw_root / arch
        try:
            download.ensure_file(base + "/" + arch, dest, force=redownload)
        except Exception as e:  # network/HTTP/IO failure -> actionable error
            raise RuntimeError(
                f"Required raw archive '{arch}' for {label} is missing from "
                f"{raw_root} and could not be downloaded ({type(e).__name__}: {e})."
                + _gated_hint(label)
            ) from e
        fps[arch] = download.fingerprint(dest)
    return fps


def build_mind(cfg: dict, key: str, force: bool, redownload: bool) -> dict:
    spec = dataset_spec(cfg, key)
    raw_root = spec["raw_root"]
    proc = spec["proc"]
    archives = spec["archives"]

    fingerprints = _ensure_archives(cfg, spec, force, redownload, label=key)

    tmp = config.temp_dir(cfg) / f"mind_{key}"
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

    _require_mind_entity_vectors(emb_parts, archives)

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
    key = f"EB-NeRD-{bundle}"
    spec = dataset_spec(cfg, key)
    raw_path = spec["raw_root"] / spec["archives"][0]
    fingerprints = _ensure_archives(cfg, spec, force, redownload, label=key)

    proc = config.processed_dir(cfg) / key
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

    # NOTE (Q8 / reviewer concern): each split-folder's impressions+history is
    # parsed by a SEPARATE parse_ebnerd_behaviors call, so every impression's
    # per-impression HISTORY list is computed from ONLY that folder's own
    # history.parquet (see _compute_per_impression_history). The concat below
    # only stacks finalized rows; histories are never recomputed across
    # train/validation. Therefore a single impression's history list CANNOT
    # contain a duplicate click traceable to the train/validation concat. The
    # long-form history.parquet is just an artifact and is not read by retrieval
    # (run_bm25/run_semantic use the HISTORY column on impressions.parquet).
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
        help="comma-separated subset (default 'all' = MIND, EB-NeRD-demo, "
             "EB-NeRD-small). Large variants MIND-large / EB-NeRD-large are "
             "supported but must be requested explicitly (MIND Large is gated "
             "on Hugging Face; EB-NeRD Large needs substantial memory).",
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

    # MIND family. The default "all" builds only MIND (small); Large is gated
    # and must be requested explicitly (e.g. --datasets MIND-large). This avoids
    # silently attempting the gated download or an oversized run.
    mind_keys = ["MIND", "MIND-large"]
    if build_all:
        requested_mind = ["MIND"]
    else:
        requested_mind = [k for k in mind_keys if k in wanted]

    for key in requested_mind:
        if args.rebuild or not _up_to_date(
            cfg, manifest, key, _mind_fp(cfg, key), files=MIND_OUTPUT_FILES
        ):
            log.info("building %s", key)
            manifest[key] = build_mind(cfg, key, args.rebuild, args.redownload)
            _write_manifest(manifest_path, manifest, cfg, key)
        else:
            log.info("%s up to date (use --rebuild to force)", key)

    # EB-NeRD family. Default "all" builds demo + small only; "large" is
    # supported but must be requested explicitly (it needs far more memory than
    # this environment provides).
    ebnerd_bundles = ["demo", "small", "large"]
    if build_all:
        active_bundles = ["demo", "small"]
    else:
        active_bundles = [b for b in ebnerd_bundles if f"EB-NeRD-{b}" in wanted]

    wants_ebnerd = build_all or any(f"EB-NeRD-{b}" in (wanted or ()) for b in ebnerd_bundles)
    if wants_ebnerd and not args.skip_embeddings:
        key = EMBEDDINGS_KEY
        build_emb = build_all or (wanted and "EB-NeRD-embeddings" in wanted)
        emb_base = config.processed_dir(cfg) / "EB-NeRD"
        emb_up_to_date = _up_to_date(
            cfg,
            manifest,
            key,
            _emb_fp(cfg),
            files=EMBEDDING_OUTPUT_FILES,
            base_dir=emb_base,
        )
        # Rebuild embeddings when requested, when this stage's own outputs are
        # stale/missing, or when any active bundle needs rebuilding (so the
        # shared embedding stage is not left incomplete for that bundle).
        if build_emb or not emb_up_to_date or any(
            not _up_to_date(cfg, manifest, f"EB-NeRD-{b}", _ebnerd_fp(cfg, b))
            for b in active_bundles
        ):
            if args.rebuild or not emb_up_to_date:
                log.info("building EB-NeRD article embeddings")
                manifest[key] = build_ebnerd_embeddings(cfg, args.rebuild, args.redownload)
                _write_manifest(manifest_path, manifest, cfg, key)

    for bundle in active_bundles:
        key = f"EB-NeRD-{bundle}"
        if build_all or key in wanted:
            if args.rebuild or not _up_to_date(cfg, manifest, key, _ebnerd_fp(cfg, bundle)):
                log.info("building %s", key)
                manifest[key] = build_ebnerd_bundle(cfg, bundle, args.rebuild, args.redownload)
                _write_manifest(manifest_path, manifest, cfg, key)
            else:
                log.info("%s up to date (use --rebuild to force)", key)

    # Persist without re-stamping any config hash: hashes are only updated when
    # a stage is actually rebuilt (see _write_manifest), so a config change for
    # stages outside this run's --datasets scope stays stale and triggers a
    # rebuild on the next full run.
    dataio.write_manifest(manifest_path, manifest)
    log.info("all done in %.1fs", time.time() - started)


def _mind_fp(cfg: dict, key: str = "MIND") -> dict:
    raw = config.raw_dir(cfg) / "MIND"
    out = {}
    for arch in cfg["dataset_defaults"][key]["archives"]:
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