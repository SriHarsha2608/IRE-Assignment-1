from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ire_rec.dataio import add_popularity
from ire_rec.datasets import ebnerd, mind
from ire_rec.split import add_temporal_split


@pytest.fixture
def news_tsv(tmp_path):
    content = (
        "N1\tnews\tworld\tTitle one\tAbstract with \"quoted\" text\t"
        "https://e.com/1\t[{\"Label\": \"X\", \"WikidataId\": \"Q1\", "
        "\"SurfaceForms\": [\"X\"]}]\t[]\n"
        "N2\tsports\tcricket\tTitle two\t\t"
        "https://e.com/2\t[]\t[{\"Label\": \"Y\", \"WikidataId\": \"Q2\", "
        "\"SurfaceForms\": [\"Y\", \"Y2\"]}]\n"
    )
    p = tmp_path / "news.tsv"
    p.write_text(content)
    return p


@pytest.fixture
def behaviors_tsv(tmp_path):
    content = (
        "I1\tU1\t11/15/2019 12:37:50 PM\tN1 N2\tN1-1 N3-0 N2-1\n"
        "I2\tU2\t11/14/2019 7:11 AM\tN5\tN2-0 N5-1 N4-0 N7-1\n"
    )
    p = tmp_path / "behaviors.tsv"
    p.write_text(content)
    return p


def test_parse_mind_news(news_tsv):
    df = mind.parse_mind_news(news_tsv)
    assert df.height == 2
    assert df["abstract"].to_list()[0].startswith("Abstract with")
    assert df["body"].to_list() == [None, None]
    assert df["published_time"].to_list() == [None, None]
    assert df["entities"].to_list() == [["X"], ["Y", "Y2"]]
    assert df["entity_ids"].to_list() == [["Q1"], ["Q2"]]


def test_parse_mind_behaviors(behaviors_tsv):
    df = mind.parse_mind_behaviors(behaviors_tsv)
    assert df.height == 2
    assert df["user_id"].to_list() == ["U1", "U2"]
    assert df["impression_time"].to_list()[0] == dt.datetime(2019, 11, 15, 12, 37, 50)
    assert df["impression_time"].to_list()[1] == dt.datetime(2019, 11, 14, 7, 11)
    assert df["history"].to_list()[0] == ["N1", "N2"]
    assert df["inview"].to_list()[0] == ["N1", "N3", "N2"]
    assert df["labels"].to_list()[0] == [1, 0, 1]


def test_entity_embeddings_and_article_pooling(tmp_path):
    vec_path = tmp_path / "e.vec"
    vec_path.write_text(
        "Q1 1.0 0.0 0.0\nQ2 0.0 1.0 0.0\nQ3 0.0 0.0 3.0\n"
    )
    vec, ids = mind.parse_entity_embeddings(vec_path)
    assert vec.shape == (3, 3)
    assert ids == ["Q1", "Q2", "Q3"]

    articles = pl.DataFrame(
        {
            "article_id": ["A1", "A2", "A3"],
            "entity_ids": [["Q1", "Q2"], ["Q3"], []],
        }
    )
    mat, covered = mind.build_entity_article_embeddings(vec, ids, articles)
    assert covered == ["A1", "A2"]
    assert mat.shape == (2, 3)
    assert np.allclose(mat[0], [0.5, 0.5, 0.0])
    assert np.allclose(mat[1], [0.0, 0.0, 3.0])


def test_temporal_split_days():
    base = dt.datetime(2020, 1, 10, 12, 0, 0)
    df = pl.DataFrame(
        {
            "impression_id": [str(i) for i in range(6)],
            "impression_time": [base + dt.timedelta(days=d) for d in (0, 1, 3, 4, 5, 6)],
        }
    )
    out, bounds = add_temporal_split(df, val_days=2, test_days=2)
    counts = {s: out.filter(pl.col("split") == s).height for s in ("train", "val", "test")}
    assert counts["test"] == 3 and counts["val"] == 1 and counts["train"] == 2
    assert bounds["method"] == "days"
    train_max = out.filter(pl.col("split") == "train")["impression_time"].max()
    val_min = out.filter(pl.col("split") == "val")["impression_time"].min()
    assert train_max < val_min


def test_temporal_split_fallback():
    base = dt.datetime(2020, 1, 10, 12, 0, 0)
    df = pl.DataFrame(
        {
            "impression_id": [str(i) for i in range(10)],
            "impression_time": [base + dt.timedelta(days=d) for d in (0, 0, 0, 0, 0, 0, 0, 0, 6, 6)],
        }
    )
    out, bounds = add_temporal_split(df, val_days=2, test_days=2)
    assert bounds["method"] == "fallback_proportional"
    assert bounds["b_val_start"] is None
    assert bounds["b_test_start"] is None
    counts = {s: out.filter(pl.col("split") == s).height for s in ("train", "val", "test")}
    assert counts["train"] == 8 and counts["val"] == 0 and counts["test"] == 2
    all_splits = out.sort("impression_time")["split"].to_list()
    # identical timestamps (both day-6 rows) must end up in the SAME split
    assert all_splits == ["train"] * 8 + ["test", "test"]


def test_temporal_split_fallback_keeps_identical_timestamps_together():
    # 5 rows at day0 + 5 rows at day1; fallback ratios (0.6, 0.2, 0.2)
    # would otherwise split the day1 group across train/val/test by row index.
    base = dt.datetime(2020, 1, 10, 12, 0, 0)
    df = pl.DataFrame(
        {
            "impression_id": [str(i) for i in range(10)],
            "impression_time": [base + dt.timedelta(days=d) for d in (0, 0, 0, 0, 0, 1, 1, 1, 1, 1)],
        }
    )
    out, bounds = add_temporal_split(
        df, val_days=2, test_days=2, fallback_ratios=(0.6, 0.2, 0.2)
    )
    assert bounds["method"] == "fallback_proportional"
    grouped = (
        out.group_by("impression_time")
        .agg(pl.col("split").n_unique())
        .filter(pl.col("split") > 1)
    )
    assert grouped.height == 0  # no timestamp group is divided across splits
    # determinism: same input yields the same split assignment
    out2, _ = add_temporal_split(
        df, val_days=2, test_days=2, fallback_ratios=(0.6, 0.2, 0.2)
    )
    assert out.sort("impression_id")["split"].to_list() == out2.sort("impression_id")["split"].to_list()


def _write_ebnerd_inputs(tmp_path, history_size=2):
    behaviors = pl.DataFrame(
        {
            "impression_id": [1, 2, 3],
            "user_id": [10, 10, 20],
            "impression_time": [
                dt.datetime(2023, 5, 1, 12),
                dt.datetime(2023, 5, 1, 14),
                dt.datetime(2023, 5, 1, 13),
            ],
            "article_ids_inview": [["a", "b"], ["a", "c"], ["b", "d"]],
            "article_ids_clicked": [["a"], ["c"], ["d"]],
            "device_type": [1, 1, 2],
            "is_sso_user": [True, False, True],
            "is_subscriber": [False, False, True],
            "gender": [0, 1, None],
            "age": [25, 34, None],
        }
    )
    history = pl.DataFrame(
        {
            "user_id": [10, 20],
            "impression_time_fixed": [
                [dt.datetime(2023, 5, 1, 8), dt.datetime(2023, 5, 1, 13, 30)],
                [dt.datetime(2023, 5, 1, 12, 30)],
            ],
            "article_id_fixed": [[100, 101], [200]],
            "read_time_fixed": [[10.0, 5.0], [7.0]],
        }
    )
    b = tmp_path / "behaviors.parquet"
    h = tmp_path / "history.parquet"
    behaviors.write_parquet(b)
    history.write_parquet(h)
    return b, h


def test_ebnerd_labels_and_causal_history(tmp_path):
    b, h = _write_ebnerd_inputs(tmp_path)
    imp, hist = ebnerd.parse_ebnerd_behaviors(b, h, history_size=2)

    assert imp.select(pl.col("labels").list.sum()).sum().to_series()[0] == 3
    rows = imp.sort("impression_id").to_dicts()
    assert rows[0]["labels"] == [1, 0]
    assert rows[1]["labels"] == [0, 1]
    assert rows[2]["labels"] == [0, 1]

    lookup = {r["impression_id"]: r["history"] for r in rows}
    assert lookup["1"] == ["100"]
    assert lookup["2"] == ["100", "101"]
    assert lookup["3"] == ["200"]

    joined = hist.join(
        imp.select(["impression_id", "impression_time"]), on="impression_id"
    )
    assert (joined["click_time"] < joined["impression_time"]).all()
    assert (joined["recency"] > 0).all()


def test_ebnerd_history_capped(tmp_path):
    b, h = _write_ebnerd_inputs(tmp_path)
    imp, _ = ebnerd.parse_ebnerd_behaviors(b, h, history_size=1)
    rows = imp.sort("impression_id").to_dicts()
    assert rows[1]["history"] == ["101"]


HISTORY_SCHEMA = [
    "user_id",
    "impression_id",
    "impression_time",
    "article_id",
    "click_time",
    "read_time",
    "recency",
]


def test_history_schema_unified(tmp_path):
    imp_m = pl.DataFrame(
        {
            "user_id": ["U1", "U2"],
            "impression_id": ["I1", "I2"],
            "impression_time": [dt.datetime(2019, 11, 15, 12), dt.datetime(2019, 11, 15, 13)],
            "history": [["N1", "N2"], None],
        }
    )
    hm = mind.build_mind_history(imp_m)
    b, h = _write_ebnerd_inputs(tmp_path)
    _, he = ebnerd.parse_ebnerd_behaviors(b, h, history_size=2)

    assert hm.columns == HISTORY_SCHEMA
    assert he.columns == HISTORY_SCHEMA
    assert [str(d) for d in hm.dtypes] == [str(d) for d in he.dtypes]
    assert hm["impression_time"].to_list() == [dt.datetime(2019, 11, 15, 12)] * 2
    assert (he["click_time"] < he["impression_time"]).all()


def test_mind_cold_start_history_empty(tmp_path):
    p = tmp_path / "cold.tsv"
    p.write_text("I1\tU1\t11/15/2019 12:37:50 PM\t\tN1-1 N2-0\n")
    df = mind.parse_mind_behaviors(p)
    assert df["history"].to_list() == [[]]


def test_ebnerd_impression_before_first_click_is_cold_start(tmp_path):
    behaviors = pl.DataFrame(
        {
            "impression_id": [1],
            "user_id": [30],
            "impression_time": [dt.datetime(2023, 5, 1, 10)],
            "article_ids_inview": [["a", "b"]],
            "article_ids_clicked": [["a"]],
            "device_type": [1],
            "is_sso_user": [True],
            "is_subscriber": [False],
            "gender": [0],
            "age": [20],
        }
    )
    history = pl.DataFrame(
        {
            "user_id": [30],
            "impression_time_fixed": [[dt.datetime(2023, 5, 1, 11)]],
            "article_id_fixed": [[100]],
            "read_time_fixed": [[5.0]],
        }
    )
    b = tmp_path / "b.parquet"
    h = tmp_path / "h.parquet"
    behaviors.write_parquet(b)
    history.write_parquet(h)
    imp, _ = ebnerd.parse_ebnerd_behaviors(b, h, history_size=50)
    assert imp["history"].to_list() == [[]]


def test_add_popularity_train_only():
    articles = pl.DataFrame({"article_id": ["A", "B", "C"]})
    impressions = pl.DataFrame(
        {
            "inview": [["A", "B"], ["B", "C"]],
            "labels": [[1, 0], [0, 1]],
            "split": ["train", "test"],
        }
    )
    feat = add_popularity(articles, impressions)
    got = {r["article_id"]: r for r in feat.to_dicts()}
    assert got["A"]["n_inviews"] == 1 and got["A"]["n_clicks"] == 1
    assert got["B"]["n_inviews"] == 1 and got["B"]["n_clicks"] is None
    assert got["C"]["n_inviews"] is None and got["C"]["n_clicks"] is None


def _minimal_cfg(tmp_path):
    """A config dict with tmp paths; used for pipeline cache/up-to-date tests."""
    return {
        "paths": {
            "raw_dir": str(tmp_path / "raw"),
            "processed_dir": str(tmp_path / "proc"),
            "temp_dir": str(tmp_path / "tmp"),
        },
        "downloads": {"MIND": "https://x", "EB-NeRD": "https://y"},
        "dataset_defaults": {
            "MIND": {
                "version": "MINDsmall",
                "archives": ["MINDsmall_train.zip", "MINDsmall_dev.zip"],
            },
            "EB-NeRD": {
                "version": "2024",
                "bundles": ["demo", "small"],
                "archives": {"demo": "ebnerd_demo.zip", "small": "ebnerd_small.zip"},
                "embeddings": {"word2vec": "w2v.zip", "bert": "bert.zip"},
            },
        },
        "temporal_split": {
            "method": "days",
            "val_days": 2,
            "test_days": 2,
            "fallback_ratios": [0.8, 0.1, 0.1],
        },
        "history": {"size": 50},
    }


def test_config_signature_changes_with_behavior_config(tmp_path):
    from ire_rec.build_pipeline import _config_signature

    cfg_a = _minimal_cfg(tmp_path)
    # temporal split change matters
    cfg_b = dict(cfg_a)
    cfg_b["temporal_split"] = dict(cfg_a["temporal_split"], val_days=5)
    assert _config_signature(cfg_a) != _config_signature(cfg_b)
    # history size change matters
    cfg_c = dict(cfg_a)
    cfg_c["history"] = {"size": 10}
    assert _config_signature(cfg_a) != _config_signature(cfg_c)
    # retrieval config does NOT affect the feature store
    cfg_d = dict(cfg_a)
    cfg_d["retrieval"] = {"bm25": {"k1": 99}}
    assert _config_signature(cfg_a) == _config_signature(cfg_d)
    # deterministic across calls
    assert _config_signature(cfg_a) == _config_signature(cfg_a)


def test_up_to_date_requires_matching_config_hash(tmp_path):
    from ire_rec.build_pipeline import _config_signature, _up_to_date

    cfg = _minimal_cfg(tmp_path)
    base = Path(cfg["paths"]["processed_dir"]) / "MIND"
    base.mkdir(parents=True)
    for f in ("articles.parquet", "impressions.parquet", "history.parquet"):
        (base / f).write_bytes(b"x")
    fp = {"a.zip": "fp1"}
    manifest = {
        "version": "2",
        "MIND": {"fingerprints": fp, "config_hash": _config_signature(cfg)},
    }
    assert _up_to_date(cfg, manifest, "MIND", fp) is True
    # changing a behavior-affecting config invalidates the cache
    cfg_changed = dict(cfg)
    cfg_changed["temporal_split"] = dict(cfg["temporal_split"], test_days=4)
    assert _up_to_date(cfg_changed, manifest, "MIND", fp) is False
    # a missing output file is also stale
    (base / "history.parquet").unlink()
    assert _up_to_date(cfg, manifest, "MIND", fp) is False


def test_up_to_date_mind_requires_embedding_outputs(tmp_path):
    from ire_rec.build_pipeline import (
        MIND_OUTPUT_FILES,
        _config_signature,
        _up_to_date,
    )

    cfg = _minimal_cfg(tmp_path)
    base = Path(cfg["paths"]["processed_dir"]) / "MIND"
    emb = base / "embeddings"
    emb.mkdir(parents=True)
    for f in ("articles.parquet", "impressions.parquet", "history.parquet"):
        (base / f).write_bytes(b"x")
    for f in ("entity_mean.npy", "entity_mean_ids.parquet"):
        (emb / f).write_bytes(b"x")
    fp = {"a.zip": "fp1"}
    manifest = {
        "version": "2",
        "MIND": {"fingerprints": fp, "config_hash": _config_signature(cfg)},
    }
    assert _up_to_date(cfg, manifest, "MIND", fp, files=MIND_OUTPUT_FILES) is True
    # deleting either embedding artifact must make MIND stale
    (emb / "entity_mean.npy").unlink()
    assert _up_to_date(cfg, manifest, "MIND", fp, files=MIND_OUTPUT_FILES) is False
    (emb / "entity_mean.npy").write_bytes(b"x")
    (emb / "entity_mean_ids.parquet").unlink()
    assert _up_to_date(cfg, manifest, "MIND", fp, files=MIND_OUTPUT_FILES) is False


def test_up_to_date_embeddings_requires_ids_parquets(tmp_path):
    """The EB-NeRD embedding stage needs the *_ids.parquet files too."""
    from ire_rec.build_pipeline import (
        EMBEDDING_OUTPUT_FILES,
        _config_signature,
        _up_to_date,
    )

    cfg = _minimal_cfg(tmp_path)
    base = Path(cfg["paths"]["processed_dir"]) / "EB-NeRD"
    for f in EMBEDDING_OUTPUT_FILES:
        (base / f).parent.mkdir(parents=True, exist_ok=True)
        (base / f).write_bytes(b"x")
    fp = {"w2v": "fp1", "bert": "fp2"}
    manifest = {
        "version": "2",
        "EB-NeRD-embeddings": {"fingerprints": fp, "config_hash": _config_signature(cfg)},
    }
    assert _up_to_date(
        cfg, manifest, "EB-NeRD-embeddings", fp, files=EMBEDDING_OUTPUT_FILES, base_dir=base
    ) is True
    # deleting any single artifact (especially an *_ids.parquet) makes it stale
    for f in ("embeddings/word2vec_ids.parquet", "embeddings/bert_ids.parquet", "embeddings/word2vec.npy", "embeddings/bert.npy"):
        (base / f).unlink()
        assert _up_to_date(
            cfg, manifest, "EB-NeRD-embeddings", fp, files=EMBEDDING_OUTPUT_FILES, base_dir=base
        ) is False
        (base / f).write_bytes(b"x")


def test_fingerprint_content_based_not_mtime(tmp_path):
    import ire_rec.download as dl

    d = tmp_path / "f.zip"
    d.write_bytes(b"identical content here")
    fp1 = dl.fingerprint(d)
    # Changing only mtime must NOT change the fingerprint (no spurious rebuild).
    old = d.stat().st_mtime
    try:
        d.touch()
    except OSError:
        pass
    fp2 = dl.fingerprint(d)
    assert fp1 == fp2
    assert fp1["sha256"] == fp2["sha256"]
    # Different content must produce a different fingerprint.
    d.write_bytes(b"different content here")
    fp3 = dl.fingerprint(d)
    assert fp3["sha256"] != fp1["sha256"]


def test_pipeline_orchestration_config_change_triggers_rebuild(monkeypatch, tmp_path):
    import sys

    import numpy as np

    import ire_rec.build_pipeline as bp

    cfg = _minimal_cfg(tmp_path)
    raw_dir = Path(cfg["paths"]["raw_dir"]) / "MIND"
    raw_dir.mkdir(parents=True)
    dummy = raw_dir / "MINDsmall_train.zip"
    dummy.write_bytes(b"rawdata")
    raw_fp = bp.download.fingerprint(dummy)

    def fake_build_mind(cfg_, key, force, redownload):
        proc = bp.config.processed_dir(cfg_) / key
        emb = proc / "embeddings"
        emb.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {"article_id": ["N1"], "title": ["t"], "abstract": ["a"]}
        ).write_parquet(proc / "articles.parquet")
        pl.DataFrame(
            {
                "impression_id": ["I1"],
                "user_id": ["U1"],
                "impression_time": [dt.datetime(2023, 1, 1)],
                "split": ["val"],
                "history": [[]],
                "inview": [["N1"]],
                "labels": [[1]],
            }
        ).write_parquet(proc / "impressions.parquet")
        pl.DataFrame({"article_id": ["N1"]}).write_parquet(proc / "history.parquet")
        np.save(emb / "entity_mean.npy", np.zeros((1, 3), dtype=np.float32))
        pl.DataFrame({"article_id": ["N1"]}).write_parquet(
            emb / "entity_mean_ids.parquet"
        )
        return {
            "fingerprints": {"MINDsmall_train.zip": raw_fp},
            "articles": 1,
            "impressions": 1,
            "splits": {},
            "embeddings": {},
        }

    calls = {"n": 0}

    def counting_build_mind(cfg_, key, force, redownload):
        calls["n"] += 1
        return fake_build_mind(cfg_, key, force, redownload)

    monkeypatch.setattr(bp.config, "load_config", lambda: cfg)
    monkeypatch.setattr(bp, "build_mind", counting_build_mind)
    monkeypatch.setattr(bp, "build_ebnerd_bundle", lambda *a, **k: {})
    monkeypatch.setattr(bp, "build_ebnerd_embeddings", lambda *a, **k: {})

    monkeypatch.setattr(sys, "argv", ["ire_rec.build_pipeline", "--datasets", "MIND"])
    bp.main()
    assert calls["n"] == 1

    # unchanged config + all outputs present -> up to date, no rebuild
    bp.main()
    assert calls["n"] == 1

    # changing a behavior-affecting config value -> cache invalidated -> rebuild
    cfg["temporal_split"]["val_days"] = 5
    bp.main()
    assert calls["n"] == 2

    # manifest reflects the new config (per-stage hash, not a global one)
    import json as _json

    m = _json.loads(
        (Path(cfg["paths"]["processed_dir"]) / "manifest.json").read_text()
    )
    assert m["MIND"]["config_hash"] == bp._config_signature(cfg)
    assert "config_hash" not in m  # no global hash remains


def test_pipeline_per_stage_config_hash_invalidation(monkeypatch, tmp_path):
    """A partial rebuild must not refresh other stages' cache status.

    build all under config A -> change config -> rebuild only MIND -> EB-NeRD
    stages must STILL be stale -> rebuild EB-NeRD -> they become current.
    """
    import sys

    import numpy as np

    import ire_rec.build_pipeline as bp

    cfg = _minimal_cfg(tmp_path)
    proc = bp.config.processed_dir(cfg)

    def _write_bundle_files(dset_dir: Path):
        dset_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"article_id": ["E1"], "title": ["t"], "abstract": ["a"]}).write_parquet(
            dset_dir / "articles.parquet"
        )
        pl.DataFrame(
            {
                "impression_id": ["I1"],
                "user_id": ["U1"],
                "impression_time": [dt.datetime(2023, 1, 1)],
                "split": ["val"],
                "history": [[]],
                "inview": [["E1"]],
                "labels": [[1]],
            }
        ).write_parquet(dset_dir / "impressions.parquet")
        pl.DataFrame({"article_id": ["E1"]}).write_parquet(dset_dir / "history.parquet")

    def fake_build_mind(cfg_, key, force, redownload):
        dset = proc / key
        emb = dset / "embeddings"
        emb.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"article_id": ["N1"], "title": ["t"], "abstract": ["a"]}).write_parquet(
            dset / "articles.parquet"
        )
        pl.DataFrame(
            {
                "impression_id": ["I1"],
                "user_id": ["U1"],
                "impression_time": [dt.datetime(2023, 1, 1)],
                "split": ["val"],
                "history": [[]],
                "inview": [["N1"]],
                "labels": [[1]],
            }
        ).write_parquet(dset / "impressions.parquet")
        pl.DataFrame({"article_id": ["N1"]}).write_parquet(dset / "history.parquet")
        np.save(emb / "entity_mean.npy", np.zeros((1, 3), dtype=np.float32))
        pl.DataFrame({"article_id": ["N1"]}).write_parquet(emb / "entity_mean_ids.parquet")
        return {"fingerprints": {}, "articles": 1, "impressions": 1, "splits": {}, "embeddings": {}}

    def fake_build_ebnerd_bundle(cfg_, bundle, force, redownload):
        _write_bundle_files(proc / f"EB-NeRD-{bundle}")
        return {"fingerprints": {}, "articles": 1, "impressions": 1, "history_rows": 1, "splits": {}}

    def fake_build_ebnerd_embeddings(cfg_, force, redownload):
        emb = proc / "EB-NeRD" / "embeddings"
        emb.mkdir(parents=True, exist_ok=True)
        for name in ("word2vec", "bert"):
            np.save(emb / f"{name}.npy", np.zeros((1, 3), dtype=np.float32))
            pl.DataFrame({"article_id": ["E1"]}).write_parquet(emb / f"{name}_ids.parquet")
        return {"fingerprints": {}, "artifacts": {}}

    calls = {"mind": 0, "demo": 0, "small": 0, "emb": 0}

    def _count(name, fn):
        def wrapper(*a, **k):
            calls[name] += 1
            return fn(*a, **k)

        return wrapper

    monkeypatch.setattr(bp.config, "load_config", lambda: cfg)
    monkeypatch.setattr(bp, "build_mind", _count("mind", fake_build_mind))
    monkeypatch.setattr(
        bp,
        "build_ebnerd_bundle",
        lambda *a, **k: _count("demo", fake_build_ebnerd_bundle)(*a, **k)
        if a[1] == "demo"
        else _count("small", fake_build_ebnerd_bundle)(*a, **k),
    )
    monkeypatch.setattr(bp, "build_ebnerd_embeddings", _count("emb", fake_build_ebnerd_embeddings))

    import json as _json

    def _manifest():
        return _json.loads((proc / "manifest.json").read_text())

    # 1) build all under config A
    monkeypatch.setattr(sys, "argv", ["ire_rec.build_pipeline", "--datasets", "all"])
    bp.main()
    assert calls == {"mind": 1, "demo": 1, "small": 1, "emb": 1}
    sig_a = bp._config_signature(cfg)
    m = _manifest()
    assert m["MIND"]["config_hash"] == sig_a
    assert m["EB-NeRD-demo"]["config_hash"] == sig_a
    assert m["EB-NeRD-small"]["config_hash"] == sig_a
    assert m["EB-NeRD-embeddings"]["config_hash"] == sig_a

    # 2) unchanged config -> nothing rebuilds
    bp.main()
    assert calls == {"mind": 1, "demo": 1, "small": 1, "emb": 1}

    # 3) config B: rebuild ONLY MIND
    cfg["temporal_split"]["val_days"] = 5
    sig_b = bp._config_signature(cfg)
    monkeypatch.setattr(sys, "argv", ["ire_rec.build_pipeline", "--datasets", "MIND"])
    bp.main()
    assert calls["mind"] == 2
    assert calls["demo"] == 1 and calls["small"] == 1 and calls["emb"] == 1
    m = _manifest()
    assert m["MIND"]["config_hash"] == sig_b  # MIND refreshed
    # EB-NeRD stages were built under config A -> must STILL be stale
    assert m["EB-NeRD-demo"]["config_hash"] == sig_a
    assert m["EB-NeRD-small"]["config_hash"] == sig_a
    assert m["EB-NeRD-embeddings"]["config_hash"] == sig_a
    assert bp._up_to_date(cfg, m, "EB-NeRD-demo", {}) is False

    # 4) rebuild EB-NeRD stages under config B -> they become current
    monkeypatch.setattr(
        sys, "argv", ["ire_rec.build_pipeline", "--datasets", "EB-NeRD-demo,EB-NeRD-small"]
    )
    bp.main()
    assert calls["demo"] == 2 and calls["small"] == 2 and calls["emb"] == 2
    m = _manifest()
    for key in ("EB-NeRD-demo", "EB-NeRD-small", "EB-NeRD-embeddings"):
        assert m[key]["config_hash"] == sig_b
    assert bp._up_to_date(cfg, m, "EB-NeRD-demo", {}) is True

    # 5) full run now does nothing (all stages current under config B)
    monkeypatch.setattr(sys, "argv", ["ire_rec.build_pipeline", "--datasets", "all"])
    bp.main()
    assert calls == {"mind": 2, "demo": 2, "small": 2, "emb": 2}


# ---------------------------------------------------------------------------
# Q5 / Large-dataset support (structural; no real Large data required)
# ---------------------------------------------------------------------------
def test_dataset_spec_mind_small_vs_large_distinct():
    import ire_rec.build_pipeline as bp
    from ire_rec.build_pipeline import dataset_spec

    cfg = bp.config.load_config()
    s_small = dataset_spec(cfg, "MIND")
    s_large = dataset_spec(cfg, "MIND-large")
    # MIND and MIND-large share the raw dir but have distinct output dirs/archives
    assert s_small["raw_root"] == s_large["raw_root"]
    assert s_small["proc"] != s_large["proc"]
    assert s_small["proc"].name == "MIND"
    assert s_large["proc"].name == "MIND-large"
    assert "MINDsmall_train.zip" in s_small["archives"]
    assert "MINDlarge_train.zip" in s_large["archives"]
    assert s_small["archives"] != s_large["archives"]
    assert s_small["embeddings"][0] == "entity_mean"


def test_dataset_spec_ebnerd_small_vs_large_distinct():
    import ire_rec.build_pipeline as bp
    from ire_rec.build_pipeline import dataset_spec

    cfg = bp.config.load_config()
    e_small = dataset_spec(cfg, "EB-NeRD-small")
    e_large = dataset_spec(cfg, "EB-NeRD-large")
    assert e_small["proc"].name == "EB-NeRD-small"
    assert e_large["proc"].name == "EB-NeRD-large"
    assert e_small["proc"] != e_large["proc"]
    assert e_small["archives"][0] == "ebnerd_small.zip"
    assert e_large["archives"][0] == "ebnerd_large.zip"
    # EB-NeRD embeddings are shared (EB-NeRD/embeddings), not per-bundle
    assert e_small["embeddings"][1] == e_large["embeddings"][1]
    assert e_small["embeddings"][1].name == "embeddings"


def test_missing_mind_large_archives_clear_error(monkeypatch):
    import urllib.error

    import ire_rec.build_pipeline as bp

    cfg = bp.config.load_config()

    def boom(*a, **k):
        raise urllib.error.URLError("401 GatedRepo")

    monkeypatch.setattr(bp.download, "ensure_file", boom)
    with pytest.raises(RuntimeError) as exc:
        bp.build_mind(cfg, "MIND-large", False, False)
    msg = str(exc.value)
    assert "MINDlarge_train.zip" in msg
    assert "gated" in msg.lower() or "Hugging Face" in msg


def test_missing_ebnerd_large_archives_clear_error(monkeypatch):
    import urllib.error

    import ire_rec.build_pipeline as bp

    cfg = bp.config.load_config()

    def boom(*a, **k):
        raise urllib.error.URLError("not found")

    monkeypatch.setattr(bp.download, "ensure_file", boom)
    with pytest.raises(RuntimeError) as exc:
        bp.build_ebnerd_bundle(cfg, "large", False, False)
    msg = str(exc.value)
    assert "ebnerd_large.zip" in msg


def test_large_cache_isolation(tmp_path):
    from ire_rec.build_pipeline import MIND_OUTPUT_FILES, _config_signature, _up_to_date

    cfg = _minimal_cfg(tmp_path)
    mind_base = Path(cfg["paths"]["processed_dir"]) / "MIND"
    large_base = Path(cfg["paths"]["processed_dir"]) / "MIND-large"
    for base in (mind_base, large_base):
        base.mkdir(parents=True)
        for f in MIND_OUTPUT_FILES:
            p = base / f
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x")
    fp_small = {"MINDsmall_train.zip": "fp1", "MINDsmall_dev.zip": "fp2"}
    fp_large = {"MINDlarge_train.zip": "fpl1", "MINDlarge_dev.zip": "fpl2"}
    manifest = {
        "version": "2",
        "MIND": {"fingerprints": fp_small, "config_hash": _config_signature(cfg)},
        "MIND-large": {"fingerprints": fp_large, "config_hash": _config_signature(cfg)},
    }
    # each key is up to date only under its own fingerprints / output dir
    assert _up_to_date(cfg, manifest, "MIND", fp_small, files=MIND_OUTPUT_FILES) is True
    assert _up_to_date(cfg, manifest, "MIND", fp_large, files=MIND_OUTPUT_FILES) is False
    assert _up_to_date(cfg, manifest, "MIND-large", fp_large, files=MIND_OUTPUT_FILES) is True
    assert _up_to_date(cfg, manifest, "MIND-large", fp_small, files=MIND_OUTPUT_FILES) is False
    # a present MIND (small) cache must NOT satisfy MIND-large
    small_only = {
        "version": "2",
        "MIND": {"fingerprints": fp_small, "config_hash": _config_signature(cfg)},
    }
    assert _up_to_date(cfg, small_only, "MIND-large", fp_large, files=MIND_OUTPUT_FILES) is False
