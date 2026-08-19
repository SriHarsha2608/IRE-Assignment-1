from __future__ import annotations

import datetime as dt
import json

import numpy as np
import polars as pl

from ire_rec.retrieval.bm25 import (
    Bm25Index,
    build_corpus,
    build_query_from_history,
    build_query_texts,
    recall_at_k,
    tokenize,
)


def test_tokenize_english_and_stopwords():
    assert tokenize("The Quick Brown Fox!") == ["quick", "brown", "fox"]
    assert tokenize("hunden løber på skole") == ["hunden", "løber", "skole"]
    assert tokenize("") == []
    assert tokenize(None) == []


def test_bm25_index_returns_most_similar_doc_first():
    corpus = [
        tokenize("the cat sat on the mat"),
        tokenize("the dog ran after the ball"),
        tokenize("global climate change report released"),
    ]
    index = Bm25Index(corpus)
    scores, idx = index.search("cat mat", top_k=2)
    assert idx[0] == 0
    assert scores.size == 2  # zero-overlap docs fill the remaining slot
    assert scores[0] > 0
    assert scores[1] == 0


def test_bm25_empty_query_returns_topk_zero_score_docs():
    index = Bm25Index([tokenize("hello world")])
    scores, idx = index.search("", top_k=5)
    # an empty query has no lexical signal; the deterministic top-K over the
    # full corpus is returned (all scores 0), capped at the corpus size
    assert scores.size == 1 and idx.size == 1
    assert scores[0] == 0 and idx[0] == 0


def test_bm25_zero_score_docs_fill_tail_deterministically():
    corpus = [
        tokenize("cat mat"),
        tokenize("dog ball"),
        tokenize("bird nest"),
        tokenize("fish pond"),
    ]
    index = Bm25Index(corpus)
    scores, idx = index.search("cat mat", top_k=4)
    assert len(idx) == 4  # exactly K returned though only 1 doc overlaps
    assert scores[0] > 0
    assert np.all(scores[1:] == 0)  # tail filled with zero-score docs
    assert list(idx) == [0, 1, 2, 3]  # deterministic ascending-index tie-break
    # a zero-overlap ground-truth doc within K is theoretically retrievable
    scores2, idx2 = index.search("cat mat", top_k=2)
    assert list(idx2) == [0, 1]
    assert scores2[1] == 0


def test_bm25_zero_fill_no_duplicates_and_positive_first():
    corpus = [
        tokenize("cat mat"),
        tokenize("dog ball"),
        tokenize("cat"),
        tokenize("fish"),
    ]
    index = Bm25Index(corpus)
    scores, idx = index.search("cat", top_k=4)
    assert len(set(idx.tolist())) == len(idx)  # never duplicate doc indices
    assert set(idx.tolist()) == {0, 1, 2, 3}   # genuinely top-K over full corpus
    assert np.all(scores[:2] > 0) and np.all(scores[2:] == 0)  # positive first


def test_bm25_corpus_smaller_than_k_returns_all():
    corpus = [tokenize("cat mat"), tokenize("dog ball")]
    index = Bm25Index(corpus)
    scores, idx = index.search("cat mat", top_k=5)
    assert len(idx) == 2  # cannot return more than the corpus size
    assert list(idx) == [0, 1]
    assert scores[0] > 0 and scores[1] == 0


def test_negative_idf_is_floored_to_epsilon():
    rare = ["alpha"]
    common = "beta"
    # beta appears in every doc -> raw idf negative (freq > N - freq + 1);
    # without the epsilon floor a query term would drive scores DOWN.
    corpus = [rare + [common]] + [[common] for _ in range(9)]
    index = Bm25Index(corpus)
    assert index.postings[common]
    assert index.idf["beta"] >= 0

    score_alone, _ = index.search(["alpha"], top_k=10)
    score_plus, idx_plus = index.search(["alpha", "beta"], top_k=10)
    hits = {int(d): float(s) for s, d in zip(score_plus, idx_plus)}
    assert hits[0] >= float(score_alone[0])


def test_build_query_from_history_caps_and_uses_titles():
    articles = pl.DataFrame(
        {
            "article_id": ["A1", "A2", "A3"],
            "title": ["First Title", "Second Title", "Third Title"],
            "abstract": ["abs one", "abs two", "abs three"],
        }
    )
    texts = build_query_texts(articles, field="title")
    q = build_query_from_history(["A1", "A2", "A3"], texts, cap=2)
    assert q == "Second Title Third Title"
    texts_ta = build_query_texts(articles, field="title_abstract")
    q2 = build_query_from_history(["A1", "A2", "A3"], texts_ta, cap=2, field="title_abstract")
    assert "abs two" in q2 and "abs three" in q2


def test_build_query_skips_unknown_articles():
    articles = pl.DataFrame({"article_id": ["A1"], "title": ["Only"], "abstract": ["abs"]})
    texts = build_query_texts(articles, field="title")
    q = build_query_from_history(["A1", "NX"], texts, cap=5)
    assert q == "Only"


def test_recall_at_k():
    assert recall_at_k(["A", "B"], ["B", "C", "A"], k=2) == 0.5
    assert recall_at_k(["A", "B"], [], k=5) == 0.0
    assert recall_at_k([], ["A"], k=5) == 0.0
    assert recall_at_k(["A", "B", "C"], ["A", "B", "C", "D"], k=3) == 1.0


def test_build_corpus_field_variants():
    articles = pl.DataFrame(
        {"article_id": ["1"], "title": ["Title here"], "abstract": ["Abstract here"]}
    )
    t = build_corpus(articles, field="title")
    assert "abstract" not in t[0]
    ta = build_corpus(articles, field="title_abstract")
    assert "abstract" in ta[0] and "title" in ta[0]


def test_run_bm25_partial_run_does_not_mix_stale_splits(tmp_path):
    from ire_rec.retrieval.run_bm25 import run_bm25

    cfg = {
        "paths": {
            "processed_dir": str(tmp_path),
            "raw_dir": str(tmp_path / "raw"),
            "temp_dir": str(tmp_path / "tmp"),
        },
        "retrieval": {
            "bm25": {
                "k1": 1.5,
                "b": 0.75,
                "remove_stopwords": True,
                "field": "title_abstract",
                "query_field": "title",
                "history_query_cap": 20,
            }
        },
    }
    dset = tmp_path / "SYN"
    dset.mkdir()
    pl.DataFrame(
        {
            "article_id": ["A1", "A2", "A3"],
            "title": ["cat mat", "dog ball", "bird nest"],
            "abstract": ["", "", ""],
        }
    ).write_parquet(dset / "articles.parquet")
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
            "history": [["A1"], ["A2"], ["A3"]],
            "inview": [["A1", "A2"], ["A2", "A3"], ["A1", "A3"]],
            "labels": [[1, 0], [1, 0], [1, 0]],
        }
    ).write_parquet(dset / "impressions.parquet")

    # full run on test only -> recall.json has test
    run_bm25(cfg, "SYN", [50], ("test",))
    out_dir = dset / "retrieval" / "bm25"
    rec = json.loads((out_dir / "recall.json").read_text())
    assert set(rec["splits"].keys()) == {"test"}

    # a later partial --limit run on val must NOT mix the stale test metrics in
    run_bm25(cfg, "SYN", [50], ("val",), limit=1)
    rec = json.loads((out_dir / "recall.json").read_text())
    assert set(rec["splits"].keys()) == {"val"}
    # the stale test candidate file is cleared so nothing stale can be combined
    assert not (out_dir / "candidates_test.parquet").exists()
    # the current val file reflects exactly the limited invocation
    assert pl.read_parquet(out_dir / "candidates_val.parquet").height == 1