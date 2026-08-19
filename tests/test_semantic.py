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
    mat = np.zeros((n, n), dtype=np.float32)
    mat[0, 0] = 1.0                      # a0 = e0
    mat[1, 1] = 2.0                      # a1 = 2*e1 (magnitude differs from a0)
    mat[2, 0], mat[2, 1] = 1.0, 2.0      # a2 = e0+2*e1 = raw mean(a0,a1) direction
    mat[3, 0], mat[3, 1] = 1.0, 1.0      # a3 = e0+e1 = normalized-mean direction
    for i in range(4, n):
        mat[i, i] = 1.0
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


def test_gt_all_covered_partial_coverage():
    """Partial embedding coverage: only fully-covered GT impressions pass."""
    from ire_rec.retrieval.run_semantic import _gt_all_covered

    covered = {"a", "b", "c"}
    df = pl.DataFrame(
        {
            "impression_id": ["I1", "I2", "I3", "I4"],
            "gt_clicked": [["a"], ["d"], ["a", "d"], ["a", "b"]],
        }
    )
    kept = df.filter(_gt_all_covered(covered))["impression_id"].to_list()
    assert kept == ["I1", "I4"]  # GT=[a] in, GT=[d] out, GT=[a,d] out, GT=[a,b] in


def test_fair_compare_intersects_on_common_impressions(tmp_path):
    """Fair comparison must evaluate both methods on the SAME impressions.

    Semantic and BM25 candidate files may diverge (e.g. --limit on one side);
    recall must only be computed on impressions present in both files, not on
    each file's own population.
    """
    from ire_rec.retrieval.run_semantic import _fair_compare

    bm25_dir = tmp_path / "bm25"
    sem_dir = tmp_path / "sem"
    bm25_dir.mkdir()
    sem_dir.mkdir()

    def _write(d: Path, rows):
        pl.DataFrame(
            {
                "impression_row_id": [r[0] for r in rows],
                "impression_id": [r[1] for r in rows],
                "split": ["val"] * len(rows),
                "gt_clicked": [r[2] for r in rows],
                "candidates": [r[3] for r in rows],
                "scores": [[0.0]] * len(rows),
            }
        ).write_parquet(d / "candidates_val.parquet")

    covered = {"a", "b", "c"}
    # semantic file: rid 0 (covered GT), rid 3 (covered GT), rid 1 (uncovered GT)
    _write(
        sem_dir,
        [
            (0, "I1", ["a"], ["a"]),
            (3, "I4", ["a"], ["a"]),
            (1, "I2", ["d"], ["d"]),
        ],
    )
    # bm25 file: rid 0, rid 1, rid 4 (rid 3 missing, rid 4 extra -> {0, 1} common)
    _write(
        bm25_dir,
        [
            (0, "I1", ["a"], ["a"]),
            (1, "I2", ["d"], ["d"]),
            (4, "I5", ["a"], ["a"]),
        ],
    )

    out = _fair_compare(bm25_dir, sem_dir, covered, catalog_n=4, top_k=[50])
    entry = out["splits"]["val"]
    assert entry["n_gt_nonempty"] == 2        # common impression rows {0, 1}
    assert entry["n_fair"] == 1               # only rid 0 is gt-covered in common
    assert entry["n_fair_bm"] == 1
    assert entry["semantic"]["recall@50"] == 1.0
    assert entry["bm25"]["recall@50"] == 1.0


def test_fair_compare_duplicate_impression_ids(tmp_path):
    """Duplicate impression_id rows must be aligned by row identity.

    MIND reuses some impression_id values for distinct rows.  Aligning on
    impression_id would conflate them; the fair comparison must use the
    row-level impression_row_id so both rows stay distinct and --limit cannot
    accidentally align the wrong duplicate row.
    """
    from ire_rec.retrieval.run_semantic import _fair_compare

    bm25_dir = tmp_path / "bm25"
    sem_dir = tmp_path / "sem"
    bm25_dir.mkdir()
    sem_dir.mkdir()

    def _write(d: Path, rows):
        pl.DataFrame(
            {
                "impression_row_id": [r[0] for r in rows],
                "impression_id": [r[1] for r in rows],
                "split": ["val"] * len(rows),
                "gt_clicked": [r[2] for r in rows],
                "candidates": [r[3] for r in rows],
                "scores": [[0.0]] * len(rows),
            }
        ).write_parquet(d / "candidates_val.parquet")

    covered = {"a", "b"}
    # both rows share impression_id "X" but are genuinely distinct impressions
    _write(sem_dir, [(0, "X", ["a"], ["a"]), (1, "X", ["b"], ["b"])])
    _write(bm25_dir, [(0, "X", ["a"], ["a"]), (1, "X", ["b"], ["b"])])

    out = _fair_compare(bm25_dir, sem_dir, covered, catalog_n=2, top_k=[50])
    entry = out["splits"]["val"]
    assert entry["n_gt_nonempty"] == 2   # both rows distinct, none silently dropped
    assert entry["n_fair"] == 2
    assert entry["n_fair_bm"] == 2
    assert entry["semantic"]["recall@50"] == 1.0
    assert entry["bm25"]["recall@50"] == 1.0

    # a --limit semantic run carrying only row 0 must align with BM25's row 0,
    # NOT with BM25's row 1 (same impression_id "X")
    _write(sem_dir, [(0, "X", ["a"], ["a"])])
    out = _fair_compare(bm25_dir, sem_dir, covered, catalog_n=2, top_k=[50])
    entry = out["splits"]["val"]
    assert entry["n_gt_nonempty"] == 1
    assert entry["n_fair"] == 1 and entry["n_fair_bm"] == 1
    assert entry["semantic"]["recall@50"] == 1.0


def test_load_embeddings_ndim_and_finite_validation(tmp_path):
    emb_dir = tmp_path / "embeddings"
    emb_dir.mkdir()
    # 1-D matrix -> error
    np.save(emb_dir / "x.npy", np.zeros(4, dtype=np.float32))
    pl.DataFrame({"article_id": ["a", "b", "c", "d"]}).write_parquet(
        emb_dir / "x_ids.parquet"
    )
    with pytest.raises(ValueError, match="ndim"):
        load_embeddings(emb_dir, "x")
    # zero embedding dimension -> error
    np.save(emb_dir / "y.npy", np.zeros((3, 0), dtype=np.float32))
    pl.DataFrame({"article_id": ["a", "b", "c"]}).write_parquet(
        emb_dir / "y_ids.parquet"
    )
    with pytest.raises(ValueError, match="dimension"):
        load_embeddings(emb_dir, "y")
    # non-finite values -> error
    np.save(emb_dir / "z.npy", np.array([[1.0, np.nan]], dtype=np.float32))
    pl.DataFrame({"article_id": ["a"]}).write_parquet(emb_dir / "z_ids.parquet")
    with pytest.raises(ValueError, match="non-finite"):
        load_embeddings(emb_dir, "z")


def test_run_semantic_partial_run_does_not_mix_stale_splits(tmp_path):
    """Semantic runs must clear stale candidates like run_bm25 does.

    A later --limit run on one split must not keep another split's stale
    candidates around (they would leak into recall.json / cold-warm /
    comparison.json).
    """
    import json as _json

    from ire_rec.config import load_config
    from ire_rec.retrieval.run_semantic import run_semantic

    store = tmp_path / "store"
    emb_dir = tmp_path / "embeddings"
    emb_dir.mkdir()
    store.mkdir()

    n = 4
    mat = np.eye(n, dtype=np.float32)
    np.save(emb_dir / "w2v.npy", mat)
    pl.DataFrame({"article_id": [f"a{i}" for i in range(n)]}).write_parquet(
        emb_dir / "w2v_ids.parquet"
    )
    pl.DataFrame({"article_id": [f"a{i}" for i in range(n)]}).write_parquet(
        store / "articles.parquet"
    )
    pl.DataFrame(
        {
            "impression_id": ["I1", "I2", "I3"],
            "user_id": ["U1", "U2", "U3"],
            "impression_time": [
                dt.datetime(2023, 1, 1),
                dt.datetime(2023, 1, 2),
                dt.datetime(2023, 1, 3),
            ],
            "split": ["val", "val", "test"],
            "history": [["a0"], ["a1"], ["a2"]],
            "inview": [["a0"], ["a1"], ["a2"]],
            "labels": [[1], [1], [1]],
        }
    ).write_parquet(store / "impressions.parquet")

    cfg = load_config()
    out_dir = store / "retrieval" / "semantic" / "w2v"

    # full run on test only -> recall.json has test
    run_semantic(cfg, "SYN", [50], ("test",), embedding="w2v", dset_dir=store, emb_dir=emb_dir)
    rec = _json.loads((out_dir / "recall.json").read_text())
    assert set(rec["splits"].keys()) == {"test"}

    # a later partial --limit run on val must not mix the stale test metrics in
    run_semantic(cfg, "SYN", [50], ("val",), limit=1, embedding="w2v", dset_dir=store, emb_dir=emb_dir)
    rec = _json.loads((out_dir / "recall.json").read_text())
    assert set(rec["splits"].keys()) == {"val"}
    # the stale test candidate file is cleared so nothing stale can be combined
    assert not (out_dir / "candidates_test.parquet").exists()
    # the current val file reflects exactly the limited invocation
    assert pl.read_parquet(out_dir / "candidates_val.parquet").height == 1