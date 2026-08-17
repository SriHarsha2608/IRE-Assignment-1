from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

ARTICLE_ID = "article_id"
TITLE = "title"
ABSTRACT = "abstract"
BODY = "body"
CATEGORY = "category"
SUBCATEGORY = "subcategory"
PUBLISHED_TIME = "published_time"
URL = "url"
ENTITIES = "entities"
ENTITY_IDS = "entity_ids"
N_INVIEWS = "n_inviews"
N_CLICKS = "n_clicks"

IMPRESSION_ID = "impression_id"
USER_ID = "user_id"
IMPRESSION_TIME = "impression_time"
HISTORY = "history"
INVIEW = "inview"
LABELS = "labels"
SPLIT = "split"
CLICK_TIME = "click_time"
READ_TIME = "read_time"
RECENCY = "recency"

SPLITS = ("train", "val", "test")


def write_df(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def read_df(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path)


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def add_popularity(articles: pl.DataFrame, impressions: pl.DataFrame) -> pl.DataFrame:
    inv = (
        impressions.select(pl.col(INVIEW))
        .explode(INVIEW)
        .group_by(INVIEW)
        .agg(pl.len().alias(N_INVIEWS))
    )
    clk = (
        impressions.select(pl.col(INVIEW), pl.col(LABELS))
        .explode([INVIEW, LABELS])
        .filter(pl.col(LABELS) == 1)
        .group_by(INVIEW)
        .agg(pl.len().alias(N_CLICKS))
    )
    stats = inv.join(clk, on=INVIEW, how="left").rename({INVIEW: ARTICLE_ID})
    return articles.join(stats, on=ARTICLE_ID, how="left")