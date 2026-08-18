from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

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

_ARTICLE_RENAMES = {
    "subtitle": ABSTRACT,
    "entity_groups": ENTITIES,
}

_EXTRA_ARTICLE_COLS = [
    "last_modified_time",
    "total_inviews",
    "total_pageviews",
    "total_read_time",
    "premium",
    "article_type",
    "sentiment_score",
    "sentiment_label",
]


def parse_ebnerd_articles(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    subcategory = (
        pl.col("subcategory")
        .list.eval(pl.element().cast(pl.String))
        .list.join(", ")
        .replace("", None)
        .alias(SUBCATEGORY)
    )
    out = (
        df.rename(_ARTICLE_RENAMES)
        .with_columns(
            pl.col(ARTICLE_ID).cast(pl.String),
            pl.coalesce("category_str", pl.col(CATEGORY).cast(pl.String)).alias(CATEGORY),
            subcategory,
            pl.lit(None, dtype=pl.List(pl.String)).alias(ENTITY_IDS),
            pl.col("url").fill_null("").alias(URL),
            pl.col(TITLE).fill_null("").alias(TITLE),
            pl.col(ABSTRACT).fill_null("").alias(ABSTRACT),
            pl.col(BODY).fill_null("").alias(BODY),
            pl.col(ENTITIES).fill_null([]).alias(ENTITIES),
        )
        .select(
            *[
                ARTICLE_ID,
                TITLE,
                ABSTRACT,
                BODY,
                CATEGORY,
                SUBCATEGORY,
                PUBLISHED_TIME,
                URL,
                ENTITIES,
                ENTITY_IDS,
            ],
            *[c for c in _EXTRA_ARTICLE_COLS if c in df.columns],
        )
    )
    return out


def _make_labels(df: pl.DataFrame, clicked: pl.Expr) -> pl.DataFrame:
    ridx = "_r"
    out = (
        df.with_row_index(ridx)
        .with_columns(clicked.alias("_clicked"))
        .select(ridx, INVIEW, "_clicked")
        .explode(INVIEW, empty_as_null=True)
        .with_columns(
            pl.col(INVIEW).is_in(pl.col("_clicked")).cast(pl.Int8).alias(LABELS)
        )
        .group_by(ridx)
        .agg(pl.col(LABELS))
        .sort(ridx)
    )
    return df.with_row_index(ridx).join(out, on=ridx).drop(ridx)


def parse_ebnerd_behaviors(
    behaviors_path: Path, history_path: Path, history_size: int = 50
) -> tuple[pl.DataFrame, pl.DataFrame]:
    behaviors = pl.read_parquet(behaviors_path)
    clicked = (
        pl.when(pl.col("article_ids_clicked").is_null())
        .then(pl.lit([], dtype=pl.List(pl.String)))
        .otherwise(pl.col("article_ids_clicked").cast(pl.List(pl.String)))
    )
    impressions = (
        behaviors.with_columns(
            pl.col(IMPRESSION_ID).cast(pl.String),
            pl.col(USER_ID).cast(pl.String),
            pl.col("article_ids_inview").cast(pl.List(pl.String)).alias(INVIEW),
        )
        .select(
            pl.col(IMPRESSION_ID),
            pl.col(USER_ID),
            pl.col(IMPRESSION_TIME),
            pl.col(INVIEW),
            "article_ids_clicked",
            "device_type",
            "is_sso_user",
            "is_subscriber",
            "gender",
            "age",
        )
    )
    impressions = _make_labels(impressions, clicked).drop("article_ids_clicked")

    history = pl.read_parquet(history_path)
    if history.height == 0:
        size = impressions.height
        impressions = impressions.with_columns(
            pl.Series(HISTORY, [[] for _ in range(size)], dtype=pl.List(pl.String))
        )
        return impressions, pl.DataFrame(
            schema={
                USER_ID: pl.String,
                IMPRESSION_ID: pl.String,
                IMPRESSION_TIME: pl.Datetime("us"),
                ARTICLE_ID: pl.String,
                CLICK_TIME: pl.Datetime("us"),
                READ_TIME: pl.Float32,
                RECENCY: pl.Float64,
            }
        )
    return _compute_per_impression_history(
        impressions, history, history_size=history_size
    )


def _compute_per_impression_history(
    impressions: pl.DataFrame, history: pl.DataFrame, history_size: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    imp_t = impressions[IMPRESSION_TIME].cast(pl.Int64).to_numpy()
    imp_u = impressions[USER_ID].to_numpy()
    iid = impressions[IMPRESSION_ID].to_list()
    imp_dt = impressions[IMPRESSION_TIME].to_list()
    n = len(imp_t)

    history_col: list[list[str]] = [[] for _ in range(n)]
    long_u: list[str] = []
    long_i: list[str] = []
    long_im: list[object] = []
    long_a: list[str] = []
    long_t: list[int] = []
    long_r: list[float] = []
    long_rec: list[float] = []

    for part in history.partition_by(USER_ID, as_dict=False):
        u_str = str(part[USER_ID][0])
        ex = (
            part.explode(
                ["impression_time_fixed", "article_id_fixed", "read_time_fixed"],
                empty_as_null=True,
            )
            .drop_nulls("article_id_fixed")
        )
        ct = ex["impression_time_fixed"].cast(pl.Int64).to_numpy()
        ca = ex["article_id_fixed"].cast(pl.String).to_list()
        cr = ex["read_time_fixed"].cast(pl.Float32).to_numpy()
        idx = np.flatnonzero(imp_u == u_str)
        if idx.size == 0:
            continue
        tt = imp_t[idx]
        for pos in np.argsort(tt, kind="stable"):
            t = tt[pos]
            k = int(np.searchsorted(ct, t, side="left"))
            if k == 0:
                continue
            take = min(k, history_size)
            s = k - take
            ai = int(idx[pos])
            history_col[ai] = ca[s:k]
            long_u.extend([u_str] * take)
            long_i.extend([iid[ai]] * take)
            long_im.extend([imp_dt[ai]] * take)
            long_a.extend(ca[s:k])
            long_t.extend(ct[s:k].tolist())
            long_r.extend(cr[s:k].tolist())
            long_rec.extend((t - ct[s:k]).tolist())

    impressions = impressions.with_columns(
        pl.Series(HISTORY, history_col, dtype=pl.List(pl.String))
    )
    history_long = pl.DataFrame(
        {
            USER_ID: pl.Series(long_u, dtype=pl.String),
            IMPRESSION_ID: pl.Series(long_i, dtype=pl.String),
            IMPRESSION_TIME: pl.Series(long_im, dtype=pl.Datetime("us")),
            ARTICLE_ID: pl.Series(long_a, dtype=pl.String),
            CLICK_TIME: pl.Series(long_t, dtype=pl.Int64).cast(pl.Datetime("us")),
            READ_TIME: pl.Series(long_r, dtype=pl.Float32),
            RECENCY: pl.Series(long_rec, dtype=pl.Float64) / 1_000_000.0,
        }
    )
    return impressions, history_long


def consolidate_embedding_parquet(src: Path, dst_dir: Path, name: str) -> dict:
    df = pl.read_parquet(src)
    emb_col = "document_vector" if "document_vector" in df.columns else df.columns[-1]
    df = df.filter(
        pl.col(emb_col).is_not_null(), pl.col(ARTICLE_ID).is_not_null()
    )
    vec = np.asarray(df[emb_col].to_list(), dtype=np.float32)
    dst_dir.mkdir(parents=True, exist_ok=True)
    np.save(dst_dir / f"{name}.npy", vec)
    pl.DataFrame({ARTICLE_ID: pl.Series(df[ARTICLE_ID].cast(pl.String))}).write_parquet(
        dst_dir / f"{name}_ids.parquet"
    )
    return {"dim": int(vec.shape[1]), "n_articles": int(vec.shape[0])}