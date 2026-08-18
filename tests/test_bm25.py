from __future__ import annotations

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
        "the cat sat on the mat".split(),
        "the dog ran after the ball".split(),
        "global climate change report released".split(),
    ]
    index = Bm25Index(corpus)
    scores, idx = index.search("cat mat", top_k=2)
    assert idx[0] == 0
    assert scores.size == 1
    assert scores[0] > 0


def test_bm25_empty_query_returns_empty():
    index = Bm25Index(["hello world".split()])
    scores, idx = index.search("", top_k=5)
    assert scores.size == 0 and idx.size == 0


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