from __future__ import annotations

import re
from collections import Counter
from math import log
from typing import Iterable

import numpy as np

from ..dataio import ARTICLE_ID, ABSTRACT, TITLE

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

_EN_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her",
    "his", "i", "if", "in", "into", "is", "it", "its", "me", "my", "no",
    "not", "of", "on", "one", "or", "our", "out", "s", "she", "so", "some",
    "t", "than", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "those", "to", "up", "us", "was", "we", "were", "what",
    "when", "where", "which", "who", "will", "with", "you", "your",
}

_DA_STOPWORDS = {
    "alle", "at", "blevet", "bliver", "da", "de", "dem", "den", "denne",
    "der", "derefter", "det", "dig", "din", "dine", "disse", "do", "dog",
    "du", "efter", "eller", "en", "end", "er", "et", "få", "for", "fra",
    "ham", "han", "hans", "har", "havde", "have", "hende", "hendes",
    "her", "hun", "hvis", "i", "ikke", "ind", "jeg", "jer", "jeres",
    "kan", "kun", "lige", "man", "mange", "med", "meget", "men", "mere",
    "mig", "min", "mine", "mit", "mod", "når", "ned", "noget", "nogle",
    "nu", "ny", "og", "op", "over", "på", "selv", "sig", "sin", "sine",
    "sit", "skal", "skulle", "som", "så", "sådan", "til", "under", "være",
    "været", "ved", "vi", "vil", "ville", "vor", "vores", "øv", "år",
}

STOPWORDS = _EN_STOPWORDS | _DA_STOPWORDS


def tokenize(text: str | None, remove_stopwords: bool = True) -> list[str]:
    """Lowercase unicode-word tokenizer (English + Danish friendly)."""
    if not text:
        return []
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    if remove_stopwords:
        tokens = [t for t in tokens if len(t) > 1 and t not in STOPWORDS]
    return tokens


class Bm25Index:
    """BM25 with an inverted index.

    Scoring follows the standard BM25 (ATIRE-idf floor variant used by
    rank-bm25) but retrieval only scores documents in the *union of the
    postings* for the query terms, instead of scanning the whole corpus.

    Postings are stored as parallel numpy arrays per term so scoring a query
    is a few array ops (np.add.at over candidate docs) rather than python
    loops, keeping retrieval fast at 60k+ document corpora.

    Args:
        corpus: iterable of token lists (one per document).
        k1, b: BM25 hyper-parameters.
        epsilon: idf floor for terms whose raw idf is negative
            (idf = ln((N - df + 0.5) / (df + 0.5)) + 1; if that value is
            negative, it is floored to ``epsilon * average_idf``, the same
            ATIRE floor rank-bm25 applies, so that a query term can never
            *decrease* a document's score).
    """

    def __init__(
        self,
        corpus: Iterable[Iterable[str]],
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        docs = [list(d) for d in corpus]
        self.corpus_size = len(docs)
        self.doc_len = np.asarray([len(d) for d in docs], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if docs else 1.0

        df: Counter[str] = Counter()
        post: dict[str, list[tuple[int, int]]] = {}
        for i, doc in enumerate(docs):
            for term, count in Counter(doc).items():
                df[term] += 1
                post.setdefault(term, []).append((i, count))
        n = self.corpus_size
        self.idf: dict[str, float] = {}
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        raw = {
            term: log((n - df[term] + 0.5) / (df[term] + 0.5)) + 1
            for term in post
        }
        average_idf = sum(raw.values()) / len(raw) if raw else 0.0
        eps = self.epsilon * average_idf
        for term, pairs in post.items():
            idf_raw = raw[term]
            self.idf[term] = idf_raw if idf_raw >= 0 else eps
            arr = np.asarray(pairs, dtype=np.int32)
            self.postings[term] = (arr[:, 0], arr[:, 1])

    def search(
        self, query: str | list[str], top_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, doc_indices) for the top-``top_k`` documents.

        Scores are BM25 similarity to ``query``; only documents sharing at
        least one term with the query are scored via the inverted index.  The
        result is the genuine top-K over the FULL corpus: when fewer than
        ``top_k`` documents have a positive score, the remaining slots are
        filled with zero-score documents in deterministic (ascending doc-index)
        order.  This keeps zero-overlap documents eligible for retrieval (a
        ground-truth article with no lexical overlap can still fall within the
        requested K) and never returns fewer than ``min(top_k, corpus_size)``
        candidates or duplicate indices.
        """
        tokens = query if isinstance(query, list) else tokenize(query)
        scores = np.zeros(self.corpus_size, dtype=np.float64)
        doc_len = self.doc_len
        b = self.b
        for term in set(tokens):
            posting = self.postings.get(term)
            if posting is None:
                continue
            di, tf = posting
            denom = tf + self.k1 * (1 - b + b * doc_len[di] / self.avgdl)
            term_score = self.idf[term] * (tf * (self.k1 + 1)) / denom
            np.add.at(scores, di, term_score)
        n_take = min(top_k, self.corpus_size)
        nz = np.flatnonzero(scores)
        if nz.size >= n_take:
            # common case: enough overlapping documents; rank them by score
            sc = scores[nz]
            order = np.argsort(-sc, kind="stable")[:n_take]
            return sc[order], nz[order]
        # scarce overlap: take all positive-scoring docs (score-sorted) and fill
        # the rest with zero-score docs in ascending doc-index order (ties at 0
        # are broken deterministically, so the result is the true top-K).
        if nz.size == 0:
            return (
                np.zeros(n_take, dtype=np.float64),
                np.arange(n_take, dtype=np.int64),
            )
        sc = scores[nz]
        order = np.argsort(-sc, kind="stable")
        picked = nz[order]
        fill = np.flatnonzero(scores == 0.0)[: n_take - picked.size]
        return (
            np.concatenate([sc[order], np.zeros(fill.size, dtype=np.float64)]),
            np.concatenate([picked, fill]),
        )


def build_corpus(
    articles,
    field: str = "title_abstract",
    remove_stopwords: bool = True,
) -> list[str]:
    """Build the tokenized corpus from an articles frame.

    ``field`` is one of ``title``, ``abstract``, ``title_abstract``.
    Returns a list of token lists aligned with ``articles`` row order.
    """
    title = articles[TITLE].fill_null("").to_list()
    if field == "title":
        texts = title
    elif field == "abstract":
        texts = articles[ABSTRACT].fill_null("").to_list()
    else:
        abstract = articles[ABSTRACT].fill_null("").to_list()
        texts = [f"{t} {a}" for t, a in zip(title, abstract)]
    return [tokenize(t, remove_stopwords=remove_stopwords) for t in texts]


def build_query_texts(
    articles,
    field: str = "title",
    remove_stopwords: bool = True,
) -> "dict[str, str]":
    """Map article_id -> prepared query text (title[+abstract])."""
    title = articles[TITLE].fill_null("").to_list()
    if field == "title_abstract":
        abstract = articles[ABSTRACT].fill_null("").to_list()
        return {
            aid: f"{t} {a}"
            for aid, t, a in zip(articles[ARTICLE_ID].to_list(), title, abstract)
        }
    return {aid: t for aid, t in zip(articles[ARTICLE_ID].to_list(), title)}


def build_query_from_history(
    history_articles: list[str],
    query_texts: dict[str, str] | list[str] | None = None,
    articles=None,
    cap: int = 20,
    field: str = "title",
    remove_stopwords: bool = True,
) -> str:
    """Concatenate titles (``field``) of the user's recently clicked articles.

    ``history_articles`` is the user's click history in chronological order
    (oldest first); the ``cap`` most recent clicks are used.  Pass a pre-built
    ``query_texts`` mapping (article_id -> text) for speed; otherwise it is
    derived from ``articles``.
    """
    if not history_articles:
        return ""
    if query_texts is None:
        if articles is None:
            raise ValueError(
                "build_query_from_history: pass either query_texts or articles "
                "(both are None)"
            )
        query_texts = build_query_texts(articles, field=field)
    texts = []
    for aid in history_articles[-cap:]:
        text = query_texts.get(aid)
        if text:
            texts.append(text)
    return " ".join(texts)


def recall_at_k(
    gt_clicked: list[str], candidates: list[str], k: int
) -> float:
    """Fraction of ground-truth clicked articles in the top-``k`` candidates."""
    if not gt_clicked:
        return 0.0
    top = set(candidates[:k])
    hits = sum(1 for a in gt_clicked if a in top)
    return hits / len(gt_clicked)