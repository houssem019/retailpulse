"""Forecast accuracy metrics, following the M5 competition definitions (bottom level only).

RMSSE  = sqrt( mean((y - ŷ)²) over the horizon  /  mean((y_t - y_{t-1})²) over training history )
         i.e. the error relative to a naive "same as yesterday" forecast, so series of very different
         volumes are comparable. History starts at the first non-zero sale (as in M5).
WRMSSE = Σ weight_i · RMSSE_i, weights = each series' share of revenue in the 28 days before the
         horizon, so the score reflects the series that matter to the business.

Note: the official M5 score averages WRMSSE over 12 aggregation levels (total, state, store, ...,
item×store). We report the item×store level, which is the hardest and the one the model predicts.
"""

from __future__ import annotations

from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd

KEYS = ["item_id", "store_id"]


def series_scales_and_weights(
    con: duckdb.DuckDBPyConnection, horizon_start: date, stores: list[str] | None = None
) -> pd.DataFrame:
    """Per-series RMSSE scale (from history before horizon_start) and revenue weight."""
    store_filter, params = "", [horizon_start]
    if stores:
        store_filter = "AND store_id IN (SELECT unnest(?::VARCHAR[]))"
        params.append(stores)
    params.append(horizon_start - timedelta(days=28))
    return con.execute(
        f"""
        WITH hist AS (
            SELECT item_id, store_id, date, units, revenue,
                   units - lag(units) OVER w AS diff,
                   sum(units) OVER (w ROWS UNBOUNDED PRECEDING) AS cum_units
            FROM sales
            WHERE date < ? {store_filter}
            WINDOW w AS (PARTITION BY item_id, store_id ORDER BY date)
        )
        SELECT item_id, store_id,
               avg(diff * diff) FILTER (WHERE cum_units > 0) AS scale,
               sum(revenue) FILTER (WHERE date >= ?) AS recent_revenue
        FROM hist GROUP BY 1, 2
    """,
        params,
    ).df()


def wrmsse(actuals: pd.DataFrame, pred_col: str, scales: pd.DataFrame) -> float:
    """`actuals` has KEYS + 'units' + pred_col for each day of the horizon."""
    df = actuals.astype({k: str for k in KEYS})
    per_series = (
        df.assign(sq_err=(df["units"] - df[pred_col]) ** 2)
        .groupby(KEYS, observed=True)["sq_err"]
        .mean()
        .rename("mse")
        .reset_index()
        .merge(scales.astype({k: str for k in KEYS}), on=KEYS)
    )
    per_series = per_series[per_series["scale"] > 0]
    rmsse = np.sqrt(per_series["mse"] / per_series["scale"])
    weights = per_series["recent_revenue"].fillna(0)
    return float((rmsse * weights).sum() / weights.sum())


def point_metrics(y: pd.Series, yhat: pd.Series) -> dict[str, float]:
    err = yhat - y
    return {
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "bias": float(err.sum() / max(y.sum(), 1e-9)),  # +0.05 = over-forecast by 5% of volume
    }
