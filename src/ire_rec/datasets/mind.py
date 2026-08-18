from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

log = logging.getLogger(__name__)

from ..dataio import (
    ABSTRACT,
    ARTICLE_ID,
    BODY,
    CATEGORY,
    CLICK_TIME,
    ENTITIES,
    ENTITY_IDS,
    HISTORY,
    IMPRESSION_ID,
    IMPRESSION_TIME,
    INVIEW,
    LABELS,
    PUBLISHED_TIME,
    READ_TIME,
    RECENCY,
    SUBCATEGORY,
    TITLE,
    URL,
    USER_ID,
)

MIND_NEWS_COLS = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]

MIND_BEHAVIOR_COLS = ["impression_id", "user_id", "time", "history", "impressions"]


def _merge_entities(title_raw: str | None, abstract_raw: str | None) -> dict:
    ents: list[str] = []
    ids: list[str] = []
    for raw in (title_raw, abstract_raw):
        if not raw:
            continue
        try:
            items = json.loads(raw)
        except Exception:
            items = []
        for it in items or []:
            ents.extend(it.get("SurfaceForms") or [])
            if it.get("WikidataId"):
                ids.append(it["WikidataId"])
    return {"ents": ents, "ids": ids}


def _parse_impressions(raw: str | None) -> dict:
    if not raw:
        return {"inv": None, "labels": None}
    inv: list[str] = []
    labels: list[int] = []
    for token in raw.split():
        article, label = token.rsplit("-", 1)
        inv.append(article)
        labels.append(int(label))
    return {"inv": inv, "labels": labels}


def parse_mind_news(path: Path) -> pl.DataFrame:
    df = pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        new_columns=MIND_NEWS_COLS,
        quote_char=None,
    )
    parsed = pl.struct(["title_entities", "abstract_entities"]).map_elements(
        lambda r: _merge_entities(r["title_entities"], r["abstract_entities"]),
        return_dtype=pl.Struct({"ents": pl.List(pl.String), "ids": pl.List(pl.String)}),
    )
    out = (
        df.with_columns(
            pl.col("abstract").replace("", None).alias(ABSTRACT),
            parsed.alias("_e"),
        )
        .with_columns(
            pl.col("_e").struct.field("ents").alias(ENTITIES),
            pl.col("_e").struct.field("ids").alias(ENTITY_IDS),
        )
        .select(
            pl.col("news_id").cast(pl.String).alias(ARTICLE_ID),
            pl.col("category").alias(CATEGORY),
            pl.col("subcategory").alias(SUBCATEGORY),
            pl.col("title").alias(TITLE),
            pl.col(ABSTRACT),
            pl.lit(None, dtype=pl.String).alias(BODY),
            pl.lit(None, dtype=pl.Datetime("us")).alias(PUBLISHED_TIME),
            pl.col("url").alias(URL),
            pl.col(ENTITIES),
            pl.col(ENTITY_IDS),
        )
    )
    return out


def parse_mind_behaviors(path: Path) -> pl.DataFrame:
    df = pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        new_columns=MIND_BEHAVIOR_COLS,
        quote_char=None,
    )
    parsed = pl.struct(["history", "impressions"]).map_elements(
        lambda r: {
            "history": r["history"].split() if r["history"] else [],
            **_parse_impressions(r["impressions"]),
        },
        return_dtype=pl.Struct(
            {
                "history": pl.List(pl.String),
                "inv": pl.List(pl.String),
                "labels": pl.List(pl.Int8),
            }
        ),
    )
    t_sec = pl.col("time").str.to_datetime("%m/%d/%Y %I:%M:%S %p", strict=False)
    t_min = pl.col("time").str.to_datetime("%m/%d/%Y %I:%M %p", strict=False)
    n_before = df.height
    out = (
        df.with_columns(
            pl.coalesce(t_sec, t_min).alias(IMPRESSION_TIME),
            parsed.alias("_p"),
        )
        .with_columns(
            pl.col("_p").struct.field("history").alias(HISTORY),
            pl.col("_p").struct.field("inv").alias(INVIEW),
            pl.col("_p").struct.field("labels").alias(LABELS),
        )
        .select(
            pl.col("impression_id").cast(pl.String).alias(IMPRESSION_ID),
            pl.col("user_id").cast(pl.String).alias(USER_ID),
            pl.col(IMPRESSION_TIME),
            pl.col(HISTORY),
            pl.col(INVIEW),
            pl.col(LABELS),
        )
        .drop_nulls(IMPRESSION_TIME)
    )
    dropped = n_before - out.height
    if dropped:
        log.warning("dropped %d MIND behavior rows with unparseable time", dropped)
    return out


def parse_entity_embeddings(path: Path) -> tuple[np.ndarray, list[str]]:
    ids: list[str] = []
    vectors: list[list[float]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split()
            ids.append(parts[0])
            vectors.append([float(x) for x in parts[1:]])
    return np.asarray(vectors, dtype=np.float32), ids


def build_entity_article_embeddings(
    vec: np.ndarray, ids: list[str], articles: pl.DataFrame
) -> tuple[np.ndarray, list[str]]:
    lookup = {eid: v for eid, v in zip(ids, vec)}
    matrix: list[np.ndarray] = []
    covered: list[str] = []
    for article_id, entity_ids in zip(
        articles[ARTICLE_ID].to_list(), articles[ENTITY_IDS].to_list()
    ):
        embs = [lookup[e] for e in (entity_ids or []) if e in lookup]
        if embs:
            matrix.append(np.mean(embs, axis=0))
            covered.append(article_id)
    if not matrix:
        return np.zeros((0, vec.shape[1]), dtype=np.float32), []
    return np.stack(matrix).astype(np.float32), covered


def build_mind_history(impressions: pl.DataFrame) -> pl.DataFrame:
    return (
        impressions.select([USER_ID, IMPRESSION_ID, IMPRESSION_TIME, HISTORY])
        .explode(HISTORY, empty_as_null=True)
        .rename({HISTORY: ARTICLE_ID})
        .drop_nulls(ARTICLE_ID)
        .with_columns(
            pl.lit(None, dtype=pl.Datetime("us")).alias(CLICK_TIME),
            pl.lit(None, dtype=pl.Float32).alias(READ_TIME),
            pl.lit(None, dtype=pl.Float64).alias(RECENCY),
        )
        .select(
            [
                USER_ID,
                IMPRESSION_ID,
                IMPRESSION_TIME,
                ARTICLE_ID,
                CLICK_TIME,
                READ_TIME,
                RECENCY,
            ]
        )
    )