from __future__ import annotations

import datetime as dt
import json

import numpy as np
import polars as pl
import pytest

from ire_rec.retrieval.semantic import (
    build_ann_index,
    load_embeddings,
    mean_pool_user_vector,
    search,
)


def test_load_embeddings_restricts_to_catalog(tmp_path):
    emb_dir = tmp_path / "embeddings"
    emb_dir.mkdir()
    mat = np.arange(12, dtype=np.float32).reshape(4, 3)
    np.save(emb_dir / "w2v.npy", mat)
    pl.DataFrame({"article_id": ["a", "b", "c", "d"]}).write_parquet(
        emb_dir / "w2v_ids.parquet"
    )
    m, ids = load_embeddings(emb_dir, "w2v", catalog={"b", "d"})
    assert ids == ["b", "d"]
    assert m.shape == (2, 3)
    np.testing.assert_allclose(m[0], mat[1])
    np.testing.assert_allclose(m[1], mat[3])


def test_load_embeddings_no_catalog_keeps_all(tmp_path):
    emb_dir = tmp_path / "embeddings"
    emb_dir.mkdir()
    np.save(emb_dir / "x.npy", np.eye(2, dtype=np.float32))
    pl.DataFrame({"article_id": ["a", "b"]}).write_parquet(emb_dir / "x_ids.parquet")
    m, ids = load_embeddings(emb_dir, "x")
    assert ids == ["a", "b"]
    assert m.shape == (2, 2)


def test_load_embeddings_dim_mismatch_raises(tmp_path):
    emb_dir = tmp_path / "embeddings"
    emb_dir.mkdir()
    np.save(emb_dir / "x.npy", np.zeros((3, 4), dtype=np.float32))
    pl.DataFrame({"article_id": ["a", "b"]}).write_parquet(emb_dir / "x_ids.parquet")
    with pytest.raises(ValueError, match="rows"):
        load_embeddings(emb_dir, "x")


def test_load_embeddings_duplicate_ids_raise(tmp_path):
    emb_dir = tmp_path / "embeddings"
    emb_dir.mkdir()
    np.save(emb_dir / "x.npy", np.zeros((3, 4), dtype=np.float32))
    pl.DataFrame({"article_id": ["a", "a", "b"]}).write_parquet(
        emb_dir / "x_ids.parquet"
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_embeddings(emb_dir, "x")


def test_build_ann_index_normalizes_copy_and_retrieves_cosine():
    rng = np.random.default_rng(1)
    mat = rng.normal(size=(50, 8)).astype(np.float32)
    original = mat.copy()
    index = build_ann_index(mat, normalize=True)
    assert index.ntotal == 50
    np.testing.assert_allclose(mat, original)  # input is never mutated
    # a normalized query's nearest neighbour is itself with cosine ~1
    q = mat[3] / np.linalg.norm(mat[3])
    scores, idx = search(q, index, top_k=3)
    assert idx[0] == 3
    assert scores[0] == pytest.approx(1.0, abs=1e-5)


def test_build_ann_index_normalize_false_keeps_raw_inner_product():
    mat = np.eye(3, dtype=np.float32) * 2.0
    original = mat.copy()
    index = build_ann_index(mat, normalize=False)
    np.testing.assert_allclose(mat, original)  # untouched
    scores, idx = search(mat[0], index, top_k=1)
    assert idx[0] == 0
    assert scores[0] == pytest.approx(4.0)  # raw dot(self) = 4, dot(others) = 0


def test_mean_pool_user_vector_pools_only_known():
    mat = np.eye(3, dtype=np.float32)
    id_to_row = {"a": 0, "b": 1, "c": 2}
    vec, n = mean_pool_user_vector(["a", "c", "missing"], id_to_row, mat)
    assert n == 2
    np.testing.assert_allclose(
        vec, [0.7071068, 0.0, 0.7071068], atol=1e-6
    )


def test_mean_pool_raw_when_normalize_false():
    mat = np.eye(3, dtype=np.float32) * 2.0
    id_to_row = {"a": 0, "b": 1}
    vec, n = mean_pool_user_vector(["a", "b"], id_to_row, mat, normalize=False)
    assert n == 2
    np.testing.assert_allclose(vec, [1.0, 1.0, 0.0])  # raw mean, no renormalizing


def test_mean_pool_zero_norm_returns_none():
    mat = np.array(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    id_to_row = {"a": 0, "b": 1, "c": 2}
    vec, n = mean_pool_user_vector(["a", "b"], id_to_row, mat, normalize=True)
    assert vec is None and n == 2  # zero query never sent to FAISS


def test_mean_pool_cold_start_returns_none():
    mat = np.eye(3, dtype=np.float32)
    id_to_row = {"a": 0}
    vec, n = mean_pool_user_vector([], id_to_row, mat)
    assert vec is None and n == 0
    vec, n = mean_pool_user_vector(["unknown"], id_to_row, mat)
    assert vec is None and n == 0


def test_search_handles_fewer_results_than_k():
    mat = np.eye(3, dtype=np.float32)
    index = build_ann_index(mat.copy(), normalize=True)
    scores, idx = search(mat[0], index, top_k=10)
    assert len(idx) == 3
    assert len(scores) == 3


def test_run_semantic_end_to_end(tmp_path):
    from ire_rec.config import load_config
    from ire_rec.retrieval.run_semantic import run_semantic

    store = tmp_path / "store"
    emb_dir = tmp_path / "embeddings"
    emb_dir.mkdir()
    store.mkdir()

    n = 8
    mat = np.eye(n, dtype=np.float32)
    mat[2, 0] = 1.0  # a2 = a0 + a1 -> mean(a0, a1) normalizes to a2's direction
    mat[2, 1] = 1.0
    np.save(emb_dir / "w2v.npy", mat)
    pl.DataFrame({"article_id": [f"a{i}" for i in range(n)]}).write_parquet(
        emb_dir / "w2v_ids.parquet"
    )

    pl.DataFrame({"article_id": [f"a{i}" for i in range(n)]}).write_parquet(
        store / "articles.parquet"
    )
    pl.DataFrame(
        {
            "impression_id": ["I1", "I2"],
            "user_id": ["U1", "U2"],
            "impression_time": [
                dt.datetime(2023, 5, 1, 12),
                dt.datetime(2023, 5, 1, 13),
            ],
            "split": ["val", "val"],
            "history": [["a0", "a1"], []],
            "inview": [["a2", "a3"], ["a5", "a6"]],
            "labels": [[1, 0], [1, 0]],
        }
    ).write_parquet(store / "impressions.parquet")

    cfg = load_config()
    summary = run_semantic(
        cfg,
        "SYN",
        [50, 100, 200],
        ("val",),
        embedding="w2v",
        dset_dir=store,
        emb_dir=emb_dir,
    )
    assert summary["embedding_coverage"] == 1.0
    assert summary["splits"]["val"]["recall@50"] == 0.5
    assert summary["splits"]["val"]["recall@100"] == 0.5
    assert summary["splits"]["val"]["recall@200"] == 0.5

    out_dir = store / "retrieval" / "semantic" / "w2v"
    assert (out_dir / "candidates_val.parquet").exists()
    rows = pl.read_parquet(out_dir / "candidates_val.parquet").sort(
        "impression_id"
    ).to_dicts()
    assert rows[0]["candidates"][0] == "a2"  # warm impression -> a2 top-1
    assert rows[1]["candidates"] == []       # cold start -> empty candidates
    assert rows[1]["n_history"] == 0 and rows[1]["n_history_used"] == 0

    comp = json.loads((out_dir / "comparison.json").read_text())
    assert comp["fair"]["coverage"] == 1.0
    assert comp["fair"]["splits"]["val"]["n_fair"] == 2
    assert comp["bm25"] is None  # no bm25 candidates in the synthetic store