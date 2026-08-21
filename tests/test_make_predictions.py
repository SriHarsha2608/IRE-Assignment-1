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
    assert text == "i0\ta b c"  # a>b by BM25; c unretrieved ranks last


def test_semantic_only_dominates(tmp_path):
    _write_dataset(
        tmp_path, "MIND",
        [{IMP_ID: "i0", INVIEW: ["a", "b"], SPLIT: "test"}],
        bm25=None,
        sem=[{RID: 0, CAND: ["a", "b"], SCORES: [2.0, 10.0]}],  # b preferred
    )
    out = _run(tmp_path, ["MIND"])
    text = (out / "MIND" / "prediction.txt").read_text().strip()
    assert text == "i0\tb a"


def test_weights_flip_order(tmp_path):
    _write_dataset(
        tmp_path, "MIND",
        [{IMP_ID: "i0", INVIEW: ["a", "b"], SPLIT: "test"}],
        bm25=[{RID: 0, CAND: ["a", "b"], SCORES: [10.0, 1.0]}],   # bm25 prefers a
        sem=[{RID: 0, CAND: ["a", "b"], SCORES: [1.0, 10.0]}],    # semantic prefers b
    )
    out = _run(tmp_path, ["MIND"], emb=None, wb=1.0, ws=0.0)
    a_first = (out / "MIND" / "prediction.txt").read_text().strip()
    assert a_first == "i0\ta b"
    out = _run(tmp_path, ["MIND"], emb=None, wb=0.0, ws=1.0)
    b_first = (out / "MIND" / "prediction.txt").read_text().strip()
    assert b_first == "i0\tb a"


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
    assert "," not in lines[0].split("\t")[0]
    assert lines[0].split("\t")[1] == "a b"


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
    assert lines[1].split(",", 1)[1] == "a b"


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
    assert "\t" in lines[0] and lines[0].split("\t")[1] == "a b"
    meta = json.loads((out / "MIND-large" / "meta.json").read_text())
    assert meta["format"] == "mind"
    assert meta["embedding"] == "entity_mean"
