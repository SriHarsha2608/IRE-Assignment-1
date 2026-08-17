from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

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
    counts = {s: out.filter(pl.col("split") == s).height for s in ("train", "val", "test")}
    assert counts["train"] == 8 and counts["val"] == 1 and counts["test"] == 1
    all_splits = out.sort("impression_time")["split"].to_list()
    assert all_splits == ["train"] * 8 + ["val", "test"]


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
