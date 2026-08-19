from __future__ import annotations

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


def test_build_ann_index_normalizes_and_retrieves_cosine():
    rng = np.random.default_rng(1)
    mat = rng.normal(size=(50, 8)).astype(np.float32)
    index = build_ann_index(mat, normalize=True)
    assert index.ntotal == 50
    norms = np.linalg.norm(mat, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    scores, idx = search(mat[3], index, top_k=3)
    assert idx[0] == 3
    assert scores[0] == pytest.approx(1.0, abs=1e-5)


def test_mean_pool_user_vector_pools_only_known():
    mat = np.eye(3, dtype=np.float32)
    id_to_row = {"a": 0, "b": 1, "c": 2}
    vec, n = mean_pool_user_vector(["a", "c", "missing"], id_to_row, mat)
    assert n == 2
    np.testing.assert_allclose(
        vec, [0.7071068, 0.0, 0.7071068], atol=1e-6
    )


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