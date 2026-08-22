import json
import polars as pl
import pytest

import ire_rec.make_predictions as MP

IMP_ID = "impression_id"
INVIEW = "inview"
SPLIT = "split"
RID = "impression_row_id"
CAND = "candidates"
SCORES = "scores"


def _write_dataset(base, name, impr_rows, bm25=None, sem=None, emb="entity_mean"):
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    impr = pl.DataFrame({
        IMP_ID: [r[IMP_ID] for r in impr_rows],
        INVIEW: [r[INVIEW] for r in impr_rows],
        SPLIT: [r[SPLIT] for r in impr_rows],
    })
    impr.write_parquet(d / "impressions.parquet")
    if bm25 is not None:
        bd = d / "retrieval" / "bm25"
        bd.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({
            RID: [r[RID] for r in bm25],
            CAND: [r[CAND] for r in bm25],
            SCORES: [r[SCORES] for r in bm25],
        }).write_parquet(bd / "candidates_test.parquet")
    if sem is not None:
        sd = d / "retrieval" / "semantic" / emb
        sd.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({
            RID: [r[RID] for r in sem],
            CAND: [r[CAND] for r in sem],
            SCORES: [r[SCORES] for r in sem],
        }).write_parquet(sd / "candidates_test.parquet")
    return d


def _cfg(base):
    return {"paths": {"processed_dir": str(base)}}


def _run(tmp_path, datasets, split="test", emb=None, wb=0.5, ws=0.5):
    out = tmp_path / "predictions"
    MP.run_predict(_cfg(tmp_path), datasets, split, emb, wb, ws, out_root=out)
    return out


def test_fusion_retrieved_above_unretrieved(tmp_path):
    _write_dataset(
        tmp_path, "MIND",
        [{IMP_ID: "i0", INVIEW: ["a", "b", "c"], SPLIT: "test"}],
        bm25=[{RID: 0, CAND: ["a", "b"], SCORES: [5.0, 1.0]}],
        sem=None,
    )
    out = _run(tmp_path, ["MIND"])
    text = (out / "MIND" / "prediction.txt").read_text().strip()
    # a (bm25 5) > b (bm25 1) > c (unretrieved, -1) => ranks [1,2,3]
    assert text == "i0\t[1,2,3]"


def test_semantic_only_dominates(tmp_path):
    _write_dataset(
        tmp_path, "MIND",
        [{IMP_ID: "i0", INVIEW: ["a", "b"], SPLIT: "test"}],
        bm25=None,
        sem=[{RID: 0, CAND: ["a", "b"], SCORES: [2.0, 10.0]}],  # b preferred
    )
    out = _run(tmp_path, ["MIND"])
    text = (out / "MIND" / "prediction.txt").read_text().strip()
    # normalized semantic: a=0, b=1 => b rank1, a rank2 => [2,1]
    assert text == "i0\t[2,1]"


def test_weights_flip_order(tmp_path):
    _write_dataset(
        tmp_path, "MIND",
        [{IMP_ID: "i0", INVIEW: ["a", "b"], SPLIT: "test"}],
        bm25=[{RID: 0, CAND: ["a", "b"], SCORES: [10.0, 1.0]}],   # bm25 prefers a
        sem=[{RID: 0, CAND: ["a", "b"], SCORES: [1.0, 10.0]}],    # semantic prefers b
    )
    out = _run(tmp_path, ["MIND"], emb=None, wb=1.0, ws=0.0)
    a_first = (out / "MIND" / "prediction.txt").read_text().strip()
    assert a_first == "i0\t[1,2]"
    out = _run(tmp_path, ["MIND"], emb=None, wb=0.0, ws=1.0)
    b_first = (out / "MIND" / "prediction.txt").read_text().strip()
    assert b_first == "i0\t[2,1]"


def test_mind_output_format_tsv_no_header(tmp_path):
    _write_dataset(
        tmp_path, "MIND",
        [{IMP_ID: "i0", INVIEW: ["a", "b"], SPLIT: "test"}],
        bm25=[{RID: 0, CAND: ["a"], SCORES: [1.0]}],
    )
    out = _run(tmp_path, ["MIND"])
    f = out / "MIND" / "prediction.txt"
    lines = f.read_text().strip().split("\n")
    assert len(lines) == 1
    assert "\t" in lines[0]
    assert lines[0].split("\t")[1] == "[1,2]"  # a retrieved rank1, b unretrieved rank2
    assert lines[0].startswith("i0\t[1,2]")


def test_ebnerd_output_format_csv_header(tmp_path):
    _write_dataset(
        tmp_path, "EB-NeRD-demo",
        [{IMP_ID: "i0", INVIEW: ["a", "b"], SPLIT: "test"}],
        bm25=[{RID: 0, CAND: ["a"], SCORES: [1.0]}],
    )
    out = _run(tmp_path, ["EB-NeRD-demo"])
    f = out / "EB-NeRD-demo" / "prediction.csv"
    lines = f.read_text().strip().split("\n")
    assert lines[0] == "impression_id,prediction"
    assert lines[1].startswith("i0,")
    assert lines[1].split(",", 1)[1] == "[1,2]"


def test_missing_dataset_raises(tmp_path):
    with pytest.raises(ValueError):
        MP.run_predict(_cfg(tmp_path), ["GHOST"], "test", None, 0.5, 0.5)


def test_meta_written(tmp_path):
    _write_dataset(
        tmp_path, "MIND",
        [{IMP_ID: "i0", INVIEW: ["a", "b"], SPLIT: "test"}],
        bm25=[{RID: 0, CAND: ["a"], SCORES: [1.0]}],
    )
    out = _run(tmp_path, ["MIND"])
    import json
    meta = json.loads((out / "MIND" / "meta.json").read_text())
    assert meta["dataset"] == "MIND"
    assert meta["format"] == "mind"
    assert meta["embedding"] == "entity_mean"
    assert meta["n_impressions"] == 1


def test_mind_large_routes_to_entity_mean_and_mind_format(tmp_path):
    _write_dataset(
        tmp_path, "MIND-large",
        [{IMP_ID: "i0", INVIEW: ["a", "b"], SPLIT: "test"}],
        bm25=[{RID: 0, CAND: ["a"], SCORES: [1.0]}],
    )
    out = _run(tmp_path, ["MIND-large"])
    f = out / "MIND-large" / "prediction.txt"
    lines = f.read_text().strip().split("\n")
    assert "\t" in lines[0] and lines[0].split("\t")[1] == "[1,2]"
    meta = json.loads((out / "MIND-large" / "meta.json").read_text())
    assert meta["format"] == "mind"
    assert meta["embedding"] == "entity_mean"


def test_rank_permutation_example(tmp_path):
    # inview = [A, B, C]; ranking puts B first, C second, A third.
    # Expected output permutation (ranks aligned with inview order) = [3,1,2].
    _write_dataset(
        tmp_path, "MIND",
        [{IMP_ID: "i0", INVIEW: ["A", "B", "C"], SPLIT: "test"}],
        bm25=None,
        sem=[{RID: 0, CAND: ["A", "B", "C"], SCORES: [1.0, 3.0, 2.0]}],  # B>C>A
    )
    out = _run(tmp_path, ["MIND"])
    text = (out / "MIND" / "prediction.txt").read_text().strip()
    assert text == "i0\t[3,1,2]"


def test_rank_permutation_ebnerd_csv(tmp_path):
    _write_dataset(
        tmp_path, "EB-NeRD-demo",
        [{IMP_ID: "i0", INVIEW: ["A", "B", "C"], SPLIT: "test"}],
        bm25=None,
        sem=[{RID: 0, CAND: ["A", "B", "C"], SCORES: [1.0, 3.0, 2.0]}],  # B>C>A
        emb="word2vec",
    )
    out = _run(tmp_path, ["EB-NeRD-demo"])
    lines = (out / "EB-NeRD-demo" / "prediction.csv").read_text().strip().split("\n")
    assert lines[0] == "impression_id,prediction"
    assert lines[1] == "i0,[3,1,2]"


def test_rank_permutation_fusion_matches_inview_order(tmp_path):
    # Retrieved articles always outrank unretrieved; ranks are 1-based and
    # aligned to inview order regardless of how many were retrieved.
    _write_dataset(
        tmp_path, "MIND",
        [{IMP_ID: "i0", INVIEW: ["x", "y", "z", "w"], SPLIT: "test"}],
        bm25=[{RID: 0, CAND: ["y", "z"], SCORES: [0.9, 0.1]}],  # y>z, x/w unretrieved
        sem=None,
    )
    out = _run(tmp_path, ["MIND"])
    text = (out / "MIND" / "prediction.txt").read_text().strip()
    # normalized bm25: y=1.0, z=0.0, x=w=-1.0 -> y(1), z(2), x(3), w(4)
    assert text == "i0\t[3,1,2,4]"
