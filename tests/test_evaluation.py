from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ire_rec.dataio import IMPRESSION_ROW_ID
from ire_rec.evaluation import metrics as M
from ire_rec.evaluation.run_eval import (
    _build_popularity,
    _determine_ranking,
    _select_candidate_rows,
    _validate,
    evaluate_candidates,
    run_eval,
)


# ---------------------------------------------------------------------------
# AUC
# ---------------------------------------------------------------------------
def test_auc_perfect_separation():
    assert M.auc_score([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == pytest.approx(1.0)


def test_auc_all_tied_is_half():
    # every score equal -> average-rank AUC = 0.5
    assert M.auc_score([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == pytest.approx(0.5)


def test_auc_tied_scores_handcomputed():
    # scores [1,1,0,0], labels [1,0,1,0]  (positives at idx 0,2)
    # avg ranks: {0,1}->1.5, {2,3}->3.5 ; sum pos ranks = 5 ; (5-3)/4 = 0.5
    assert M.auc_score([1.0, 1.0, 0.0, 0.0], [1, 0, 1, 0]) == pytest.approx(0.5)


def test_auc_skips_single_class():
    assert M.auc_score([0.1, 0.2, 0.3], [1, 1, 1]) is None
    assert M.auc_score([0.1, 0.2, 0.3], [0, 0, 0]) is None


# ---------------------------------------------------------------------------
# MRR
# ---------------------------------------------------------------------------
def test_mrr_first_click():
    assert M.mrr_from_ranked([1, 0, 0]) == pytest.approx(1.0)


def test_mrr_third_click():
    assert M.mrr_from_ranked([0, 0, 1, 0]) == pytest.approx(1 / 3)


def test_mrr_no_click():
    assert M.mrr_from_ranked([0, 0, 0]) is None


# ---------------------------------------------------------------------------
# nDCG
# ---------------------------------------------------------------------------
def test_ndcg_perfect():
    assert M.ndcg_at_k_from_ranked([1, 0, 0], 5) == pytest.approx(1.0)


def test_ndcg_rank2_k5():
    # click at rank 2 -> gain 1/log2(3) / 1
    assert M.ndcg_at_k_from_ranked([0, 1, 0], 5) == pytest.approx(1 / np.log2(3))


def test_ndcg_k_smaller_than_list():
    # click at rank 2, K=1 -> 0
    assert M.ndcg_at_k_from_ranked([0, 1, 0], 1) == pytest.approx(0.0)


def test_ndcg_fewer_than_k():
    assert M.ndcg_at_k_from_ranked([1, 0], 10) == pytest.approx(1.0)


def test_ndcg_no_positive():
    assert M.ndcg_at_k_from_ranked([0, 0, 0], 5) is None


# ---------------------------------------------------------------------------
# diversity
# ---------------------------------------------------------------------------
def test_diversity_empty_and_singleton():
    assert M.intra_list_diversity([]) == 0.0
    assert M.intra_list_diversity(["a"]) == 0.0


def test_diversity_all_same():
    assert M.intra_list_diversity(["a", "a"]) == 0.0


def test_diversity_all_distinct():
    assert M.intra_list_diversity(["a", "b"]) == pytest.approx(1.0)
    assert M.intra_list_diversity(["a", "b", "c"]) == pytest.approx(1.0)


def test_diversity_mixed():
    # a2,b2 -> same=4 ; L=4 -> 1 - 4/12 = 0.6667
    assert M.intra_list_diversity(["a", "a", "b", "b"]) == pytest.approx(2 / 3)


def test_diversity_unknown_bucket_kept():
    # unknown counts as its own category, not dropped
    assert M.intra_list_diversity(["__UNKNOWN__", "__UNKNOWN__"]) == 0.0
    assert M.intra_list_diversity(["a", "__UNKNOWN__"]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# novelty
# ---------------------------------------------------------------------------
def test_novelty_monotonic_popularity():
    p = {"popular": 0.9, "niche": 0.05}
    assert M.novelty_for_ids(["niche"], p) > M.novelty_for_ids(["popular"], p)


def test_novelty_empty_list():
    assert M.novelty_for_ids([], {"a": 0.5}) == 0.0


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------
def test_bootstrap_mean_ci_reproducible():
    v = np.arange(1, 101, dtype=float)
    a = M.bootstrap_mean_ci(v, 200, 42)
    b = M.bootstrap_mean_ci(v, 200, 42)
    assert a == b
    assert a["ci_low"] <= a["value"] <= a["ci_high"]


def test_bootstrap_mean_ci_empty():
    assert M.bootstrap_mean_ci([], 100, 1) == {
        "value": None,
        "ci_low": None,
        "ci_high": None,
    }


def test_bootstrap_coverage_reproducible():
    sets = [{"a"}, {"b"}, {"c"}]
    a = M.bootstrap_coverage_ci(sets, 5, 200, 7)
    b = M.bootstrap_coverage_ci(sets, 5, 200, 7)
    assert a == b
    assert a["value"] == pytest.approx(3 / 5)


# ---------------------------------------------------------------------------
# INVIEW ranking
# ---------------------------------------------------------------------------
def test_ranking_unretrieved_gets_zero_score():
    inview = ["a", "b", "c"]
    labels = [0, 0, 0]
    cands = ["a"]
    scores = [0.5]
    s_arr, ranked = _determine_ranking(inview, labels, cands, scores)
    assert s_arr == [0.5, 0.0, 0.0]


def test_ranking_deterministic_unretrieved_order():
    # a retrieved (score 0.5); b,c unretrieved -> keep inview order b then c
    inview = ["a", "b", "c"]
    labels = [0, 1, 0]
    cands = ["a"]
    scores = [0.5]
    s_arr, ranked = _determine_ranking(inview, labels, cands, scores)
    # ranking tuples: (-0.5,0,0,0) for a ; (0,1,inf,1) b ; (0,1,inf,2) c
    assert ranked == [0, 1, 0]  # a first, then b (clicked) before c


def test_ranking_retrieved_ties_keep_retrieval_order():
    # two retrieved with equal score -> retrieval order (first candidate wins)
    inview = ["x", "y"]
    labels = [0, 0]
    cands = ["y", "x"]  # retrieval order: y then x, both score 0.0
    scores = [0.0, 0.0]
    s_arr, ranked = _determine_ranking(inview, labels, cands, scores)
    # y (status0, rank0) precedes x (status0, rank1)
    assert ranked == [0, 0]
    assert s_arr == [0.0, 0.0]


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def _cand_df(rows):
    return pl.DataFrame(
        rows,
        schema={
            "impression_row_id": pl.UInt32,
            "impression_id": pl.String,
            "split": pl.String,
            "gt_clicked": pl.List(pl.String),
            "candidates": pl.List(pl.String),
            "scores": pl.List(pl.Float64),
            "n_query_terms": pl.Int32,
        },
    )


def test_validate_ok():
    _validate(_cand_df([
        {"impression_row_id": 0, "impression_id": "i", "split": "val",
         "gt_clicked": [], "candidates": ["a"], "scores": [0.5], "n_query_terms": 1}
    ]), "val")


def test_validate_nonunique_row_id():
    with pytest.raises(ValueError):
        _validate(_cand_df([
            {"impression_row_id": 0, "impression_id": "i", "split": "val",
             "gt_clicked": [], "candidates": ["a"], "scores": [0.5], "n_query_terms": 1},
            {"impression_row_id": 0, "impression_id": "i2", "split": "val",
             "gt_clicked": [], "candidates": ["b"], "scores": [0.5], "n_query_terms": 1},
        ]), "val")


def test_validate_split_mismatch():
    with pytest.raises(ValueError):
        _validate(_cand_df([
            {"impression_row_id": 0, "impression_id": "i", "split": "test",
             "gt_clicked": [], "candidates": ["a"], "scores": [0.5], "n_query_terms": 1}
        ]), "val")


def test_validate_length_mismatch():
    with pytest.raises(ValueError):
        _validate(_cand_df([
            {"impression_row_id": 0, "impression_id": "i", "split": "val",
             "gt_clicked": [], "candidates": ["a", "b"], "scores": [0.5], "n_query_terms": 1}
        ]), "val")


# ---------------------------------------------------------------------------
# end-to-end evaluate_candidates
# ---------------------------------------------------------------------------
def _make_inputs():
    arts = pl.DataFrame({
        "article_id": ["a", "b", "c", "d", "e"],
        "category": ["cat1", "cat1", "cat2", None, "cat2"],
        "n_inviews": [100.0, 10.0, 1.0, None, 0.0],
        "n_clicks": [50.0, 5.0, 0.0, None, 0.0],
    })
    p_lookup, id_to_cat, catalog_set, catalog_cats = _build_popularity(arts)

    impr = pl.DataFrame({
        "impression_row_id": [0, 1, 2],
        "impression_id": ["I0", "I1", "I1"],  # duplicate impression_id!
        "inview": [["a", "b", "c"], ["d", "e"], ["a", "e"]],
        "labels": [[1, 0, 0], [0, 1], [1, 1]],
        "history": [["x"], [], ["y", "z"]],  # row 1 = cold
    }).with_columns(pl.col("history").cast(pl.List(pl.String)))

    cand = _cand_df([
        # row 0: a retrieved; b,c unretrieved
        {"impression_row_id": 0, "impression_id": "I0", "split": "val",
         "gt_clicked": ["a"], "candidates": ["a"], "scores": [0.9], "n_query_terms": 1},
        # row 1: d retrieved, e unretrieved
        {"impression_row_id": 1, "impression_id": "I1", "split": "val",
         "gt_clicked": ["e"], "candidates": ["d"], "scores": [0.7], "n_query_terms": 1},
        # row 2: both a,e retrieved (e has higher score)
        {"impression_row_id": 2, "impression_id": "I1", "split": "val",
         "gt_clicked": ["a", "e"], "candidates": ["e", "a"], "scores": [0.6, 0.4],
         "n_query_terms": 2},
    ])
    return cand, impr, p_lookup, id_to_cat, catalog_set, catalog_cats


def test_evaluate_end_to_end_keys_and_counts():
    cand, impr, p_lookup, id_to_cat, cs, cc = _make_inputs()
    res = evaluate_candidates(cand, impr, p_lookup, id_to_cat, cs, cc, 50, 42, "val")
    assert res["n_impressions"] == 3
    assert res["n_valid_for_accuracy_metrics"] == 3  # all rows have >=1 click
    # all metric sections present with CIs (incl. coverage)
    for k in ("auc", "mrr", "ndcg@5", "ndcg@10"):
        assert res["metrics"][k]["value"] is not None
        assert res["metrics"][k]["ci_low"] is not None
    for k in ("intra_list_diversity", "novelty", "article_coverage", "category_coverage"):
        assert res["beyond_accuracy"][k]["value"] is not None
        assert res["beyond_accuracy"][k]["ci_low"] is not None
    # cold/warm slice
    sl = res["slices"]["cold_start_vs_warm"]
    assert sl["cold"]["n"] == 1
    assert sl["warm"]["n"] == 2


def test_evaluate_duplicate_impression_id_not_collapsed():
    cand, impr, p_lookup, id_to_cat, cs, cc = _make_inputs()
    res = evaluate_candidates(cand, impr, p_lookup, id_to_cat, cs, cc, 50, 42, "val")
    # rows with impression_id "I1" (row_id 1 and 2) must stay distinct
    assert res["n_impressions"] == 3


def test_evaluate_missing_candidate_row_fails():
    cand, impr, p_lookup, id_to_cat, cs, cc = _make_inputs()
    bad = _cand_df([
        {"impression_row_id": 999, "impression_id": "Z", "split": "val",
         "gt_clicked": [], "candidates": ["a"], "scores": [0.1], "n_query_terms": 1}
    ])
    with pytest.raises(ValueError):
        evaluate_candidates(bad, impr, p_lookup, id_to_cat, cs, cc, 50, 42, "val")


def test_evaluate_no_positive_labels():
    arts = pl.DataFrame({
        "article_id": ["a", "b"],
        "category": ["c1", "c2"],
        "n_inviews": [1.0, 1.0],
        "n_clicks": [1.0, 1.0],
    })
    p_lookup, id_to_cat, cs, cc = _build_popularity(arts)
    impr = pl.DataFrame({
        "impression_row_id": [0],
        "impression_id": ["I0"],
        "inview": [["a", "b"]],
        "labels": [[0, 0]],
        "history": [["x"]],
    }).with_columns(pl.col("history").cast(pl.List(pl.String)))
    cand = _cand_df([
        {"impression_row_id": 0, "impression_id": "I0", "split": "val",
         "gt_clicked": [], "candidates": ["a"], "scores": [0.5], "n_query_terms": 1}
    ])
    res = evaluate_candidates(cand, impr, p_lookup, id_to_cat, cs, cc, 50, 42, "val")
    assert res["n_valid_for_accuracy_metrics"] == 0
    assert res["metrics"]["auc"]["value"] is None
    assert res["metrics"]["mrr"]["value"] is None
    # beyond-accuracy still computed
    assert res["beyond_accuracy"]["intra_list_diversity"]["value"] is not None


# ---------------------------------------------------------------------------
# extended validation (spec 1 A-E)
# ---------------------------------------------------------------------------
def _build_case(impr_data: dict, cand_data: list, arts=None):
    if arts is None:
        arts = pl.DataFrame({
            "article_id": ["a", "b", "c", "d", "e"],
            "category": ["cat1", "cat1", "cat2", "cat2", "cat2"],
            "n_inviews": [100.0, 10.0, 1.0, 5.0, 2.0],
            "n_clicks": [50.0, 5.0, 0.0, 2.0, 1.0],
        })
    p_lookup, id_to_cat, cs, cc = _build_popularity(arts)
    impr = pl.DataFrame(impr_data).with_columns(
        pl.col("history").cast(pl.List(pl.String))
    )
    cand = _cand_df(cand_data)
    return cand, impr, p_lookup, id_to_cat, cs, cc


def test_validate_inview_labels_length_mismatch():
    cand, impr, p, ic, cs, cc = _build_case(
        {"impression_row_id": [0], "impression_id": ["I0"], "inview": [["a", "b"]],
         "labels": [[1, 0, 0]], "history": [["x"]]},
        [{"impression_row_id": 0, "impression_id": "I0", "split": "val",
          "gt_clicked": ["a"], "candidates": ["a"], "scores": [0.5], "n_query_terms": 1}],
    )
    with pytest.raises(ValueError):
        evaluate_candidates(cand, impr, p, ic, cs, cc, 20, 1, "val")


def test_validate_gt_clicked_mismatch():
    cand, impr, p, ic, cs, cc = _build_case(
        {"impression_row_id": [0], "impression_id": ["I0"], "inview": [["a", "b"]],
         "labels": [[1, 0]], "history": [["x"]]},
        [{"impression_row_id": 0, "impression_id": "I0", "split": "val",
          "gt_clicked": ["b"], "candidates": ["a"], "scores": [0.5], "n_query_terms": 1}],
    )
    with pytest.raises(ValueError):
        evaluate_candidates(cand, impr, p, ic, cs, cc, 20, 1, "val")


def test_validate_candidate_id_not_in_catalog():
    cand, impr, p, ic, cs, cc = _build_case(
        {"impression_row_id": [0], "impression_id": ["I0"], "inview": [["z"]],
         "labels": [[1]], "history": [["x"]]},
        [{"impression_row_id": 0, "impression_id": "I0", "split": "val",
          "gt_clicked": ["z"], "candidates": ["z"], "scores": [0.5], "n_query_terms": 1}],
    )
    with pytest.raises(ValueError):
        evaluate_candidates(cand, impr, p, ic, cs, cc, 20, 1, "val")


def test_validate_duplicate_candidate_ids():
    cand, impr, p, ic, cs, cc = _build_case(
        {"impression_row_id": [0], "impression_id": ["I0"], "inview": [["a", "b"]],
         "labels": [[0, 0]], "history": [["x"]]},
        [{"impression_row_id": 0, "impression_id": "I0", "split": "val",
          "gt_clicked": [], "candidates": ["a", "a"], "scores": [0.5, 0.5], "n_query_terms": 1}],
    )
    with pytest.raises(ValueError):
        evaluate_candidates(cand, impr, p, ic, cs, cc, 20, 1, "val")


def test_validate_nonfinite_score():
    cand, impr, p, ic, cs, cc = _build_case(
        {"impression_row_id": [0], "impression_id": ["I0"], "inview": [["a"]],
         "labels": [[1]], "history": [["x"]]},
        [{"impression_row_id": 0, "impression_id": "I0", "split": "val",
          "gt_clicked": ["a"], "candidates": ["a"], "scores": [float("nan")], "n_query_terms": 1}],
    )
    with pytest.raises(ValueError):
        evaluate_candidates(cand, impr, p, ic, cs, cc, 20, 1, "val")


def test_select_candidate_rows_deterministic_by_row_id():
    rows = [
        {"impression_row_id": 10, "impression_id": "i", "split": "val",
         "gt_clicked": [], "candidates": ["a"], "scores": [0.5], "n_query_terms": 1},
        {"impression_row_id": 3, "impression_id": "i", "split": "val",
         "gt_clicked": [], "candidates": ["a"], "scores": [0.5], "n_query_terms": 1},
        {"impression_row_id": 7, "impression_id": "i", "split": "val",
         "gt_clicked": [], "candidates": ["a"], "scores": [0.5], "n_query_terms": 1},
        {"impression_row_id": 1, "impression_id": "i", "split": "val",
         "gt_clicked": [], "candidates": ["a"], "scores": [0.5], "n_query_terms": 1},
    ]
    cand = _cand_df(rows)
    sel = _select_candidate_rows(cand, 2)
    assert sel[IMPRESSION_ROW_ID].to_list() == [1, 3]


def test_slice_contains_coverage_metrics():
    cand, impr, p_lookup, id_to_cat, cs, cc = _make_inputs()
    res = evaluate_candidates(cand, impr, p_lookup, id_to_cat, cs, cc, 50, 42, "val")
    sl = res["slices"]["cold_start_vs_warm"]
    for part in ("cold", "warm"):
        for k in ("article_coverage", "category_coverage"):
            assert sl[part][k]["value"] is not None
            assert sl[part][k]["ci_low"] is not None
            assert sl[part][k]["ci_high"] is not None


# ---------------------------------------------------------------------------
# method discovery failures (spec 3)
# ---------------------------------------------------------------------------
def _write_tmp_dataset(base: Path, name: str, with_bm25=True,
                       with_sem_entity=True, with_bert=False):
    dset = base / name
    dset.mkdir(parents=True, exist_ok=True)
    arts = pl.DataFrame({
        "article_id": ["a", "b", "c", "d", "e"],
        "category": ["cat1", "cat1", "cat2", "cat2", "cat2"],
        "n_inviews": [100.0, 10.0, 1.0, 5.0, 2.0],
        "n_clicks": [50.0, 5.0, 0.0, 2.0, 1.0],
    })
    arts.write_parquet(dset / "articles.parquet")
    impr = pl.DataFrame({
        "impression_id": ["I0"],
        "inview": [["a", "b", "c"]],
        "labels": [[1, 0, 0]],
        "history": [["x"]],
    }).with_columns(pl.col("history").cast(pl.List(pl.String)))
    impr.write_parquet(dset / "impressions.parquet")
    schema = {
        "impression_row_id": pl.UInt32,
        "impression_id": pl.String,
        "split": pl.String,
        "gt_clicked": pl.List(pl.String),
        "candidates": pl.List(pl.String),
        "scores": pl.List(pl.Float64),
        "n_query_terms": pl.Int32,
    }

    def _write(subdir: str):
        d = dset / "retrieval" / subdir
        d.mkdir(parents=True, exist_ok=True)
        pl.DataFrame([{
            "impression_row_id": 0, "impression_id": "I0", "split": "val",
            "gt_clicked": ["a"], "candidates": ["a"], "scores": [0.5],
            "n_query_terms": 1,
        }], schema=schema).write_parquet(d / "candidates_val.parquet")

    if with_bm25:
        _write("bm25")
    if with_sem_entity:
        _write("semantic/entity_mean")
    if with_bert:
        _write("semantic/bert")
    return dset


def _tmp_cfg(base: Path, runs=20):
    return {
        "paths": {"raw_dir": "x", "processed_dir": str(base)},
        "evaluation": {"bootstrap_runs": runs, "bootstrap_seed": 1},
    }


def test_discovery_missing_requested_method_fails(tmp_path):
    _write_tmp_dataset(tmp_path, "D")
    with pytest.raises(ValueError):
        run_eval(_tmp_cfg(tmp_path), ["D"], ("val",), methods_filter=["semantic_bert"])


def test_discovery_present_requested_method_ok(tmp_path):
    _write_tmp_dataset(tmp_path, "D")
    res = run_eval(_tmp_cfg(tmp_path), ["D"], ("val",), methods_filter=["bm25"])
    assert "bm25_val" in res["D"]


def test_discovery_no_candidates_fails(tmp_path):
    _write_tmp_dataset(tmp_path, "D", with_bm25=False,
                       with_sem_entity=False, with_bert=False)
    with pytest.raises(ValueError):
        run_eval(_tmp_cfg(tmp_path), ["D"], ("val",))
