from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import polars as pl

from .dataio import SPLIT

log = logging.getLogger(__name__)


def add_temporal_split(
    df: pl.DataFrame,
    time_col: str = "impression_time",
    val_days: int = 2,
    test_days: int = 2,
    fallback_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> tuple[pl.DataFrame, dict[str, Any]]:
    df = df.sort(time_col)
    tmin = df[time_col].min()
    tmax = df[time_col].max()
    b_test = tmax - dt.timedelta(days=test_days)
    b_val = b_test - dt.timedelta(days=val_days)

    split = (
        pl.when(pl.col(time_col) >= b_test)
        .then(pl.lit("test"))
        .when(pl.col(time_col) >= b_val)
        .then(pl.lit("val"))
        .otherwise(pl.lit("train"))
    )
    out = df.with_columns(split.alias(SPLIT))
    counts = {s: out.filter(pl.col(SPLIT) == s).height for s in ("train", "val", "test")}

    if counts["val"] == 0 or counts["test"] == 0:
        log.warning(
            "day-based split produced empty sets (train=%d val=%d test=%d); "
            "falling back to proportional-by-time split",
            counts["train"],
            counts["val"],
            counts["test"],
        )
        # Proportional fallback: rows are sorted by time and split by row
        # fraction, but identical timestamps must NOT be divided between
        # splits (same-time impressions belong to the same split).  Each
        # timestamp group is assigned to a split by the position of its LAST
        # row, so a group never straddles a boundary.  Deterministic.
        sorted_df = df.sort(time_col)
        n = sorted_df.height
        out = (
            sorted_df.with_row_index("_i")
            .with_columns(pl.col("_i").max().over(time_col).alias("_g"))
            .with_columns(
                pl.when(pl.col("_g") / n < fallback_ratios[0]).then(pl.lit("train"))
                .when(pl.col("_g") / n < fallback_ratios[0] + fallback_ratios[1])
                .then(pl.lit("val"))
                .otherwise(pl.lit("test"))
                .alias(SPLIT)
            )
            .drop("_i", "_g")
        )
        counts = {s: out.filter(pl.col(SPLIT) == s).height for s in ("train", "val", "test")}
        method = "fallback_proportional"
    else:
        method = "days"

    boundaries = {
        "method": method,
        "min_time": tmin,
        "b_val_start": b_val if method == "days" else None,
        "b_test_start": b_test if method == "days" else None,
        "max_time": tmax,
        "counts": counts,
    }
    return out, boundaries