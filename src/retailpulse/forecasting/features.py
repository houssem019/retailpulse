"""Feature engineering for the demand model — one SQL query used by BOTH training and serving.

Forecasting strategy: **direct, 28 days ahead**. Every lag/rolling feature looks at least 28 days
back, so a single model can score any day in the next 28 without feeding its own predictions back
in (the "recursive" approach, which compounds errors and is slow to serve).

All lags use `RANGE ... INTERVAL n DAYS` windows over dates (not row offsets), so gaps in a series
can never silently shift a lag onto the wrong day.
"""

from __future__ import annotations

from datetime import date, timedelta

import duckdb
import pandas as pd

HORIZON = 28
LOOKBACK_DAYS = 120  # history needed before the first scored day (longest window = 28 + 56 days)

CATEGORICAL = ["item_id", "store_id", "dept_id", "cat_id", "event_type_1"]
NUMERIC = [
    "sell_price",
    "price_ratio",
    "price_change_wow",
    "wday",
    "month",
    "mday",
    "week",
    "snap",
    "age_days",
    "lag_28",
    "lag_35",
    "lag_42",
    "lag_49",
    "lag_56",
    "rmean_7",
    "rmean_28",
    "rmean_56",
    "rstd_28",
    "zero_frac_28",
]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "units"


def _lag(n: int) -> str:
    return f"max(units) OVER (w RANGE BETWEEN INTERVAL {n} DAYS PRECEDING AND INTERVAL {n} DAYS PRECEDING)"


def _window(agg: str, days: int) -> str:
    """Aggregate over the `days`-long window that ends HORIZON days before the scored date."""
    start = HORIZON + days - 1
    return (
        f"{agg} OVER (w RANGE BETWEEN INTERVAL {start} DAYS PRECEDING AND INTERVAL {HORIZON} DAYS PRECEDING)"
    )


def build_features(
    con: duckdb.DuckDBPyConnection,
    start: date,
    end: date,
    stores: list[str] | None = None,
    items: list[str] | None = None,
) -> pd.DataFrame:
    """Feature rows for every listed (item, store, date) with start <= date <= end.

    Dates after the last observed sale are included (units = NULL) as long as a planned price
    exists, which is what makes future forecasts and what-if scenarios possible.
    """
    filters, params = [], [start - timedelta(days=LOOKBACK_DAYS), end]
    if stores:
        filters.append("p.store_id IN (SELECT unnest(?::VARCHAR[]))")
        params.append(stores)
    if items:
        filters.append("p.item_id IN (SELECT unnest(?::VARCHAR[]))")
        params.append(items)
    where = "".join(f" AND {f}" for f in filters)
    params += [start, end]

    sql = f"""
    WITH week_start AS (SELECT wm_yr_wk, min(date) AS date FROM calendar GROUP BY 1),
    series_stats AS (  -- computed over the full price history so train and serve agree exactly
        SELECT p.store_id, p.item_id, min(ws.date) AS launch_date, avg(p.sell_price) AS mean_price
        FROM prices p JOIN week_start ws USING (wm_yr_wk)
        GROUP BY 1, 2
    ),
    spine AS (
        SELECT p.item_id, p.store_id, c.date, p.sell_price, c.wday, c.month, c.event_type_1,
               CASE st.state_id WHEN 'CA' THEN c.snap_ca WHEN 'TX' THEN c.snap_tx ELSE c.snap_wi END AS snap,
               s.units
        FROM prices p
        JOIN calendar c ON c.wm_yr_wk = p.wm_yr_wk
        JOIN stores st ON st.store_id = p.store_id
        LEFT JOIN sales s ON s.item_id = p.item_id AND s.store_id = p.store_id AND s.date = c.date
        WHERE c.date BETWEEN ? AND ? {where}
    ),
    feats AS (
        SELECT *,
            {_lag(28)} AS lag_28, {_lag(35)} AS lag_35, {_lag(42)} AS lag_42,
            {_lag(49)} AS lag_49, {_lag(56)} AS lag_56,
            {_window("avg(units)", 7)} AS rmean_7,
            {_window("avg(units)", 28)} AS rmean_28,
            {_window("avg(units)", 56)} AS rmean_56,
            {_window("stddev_samp(units)", 28)} AS rstd_28,
            {_window("avg(CAST(units = 0 AS INTEGER))", 28)} AS zero_frac_28,
            sell_price / max(sell_price) OVER (
                w RANGE BETWEEN INTERVAL 7 DAYS PRECEDING AND INTERVAL 7 DAYS PRECEDING) - 1 AS price_change_wow
        FROM spine
        WINDOW w AS (PARTITION BY item_id, store_id ORDER BY date)
    )
    SELECT f.item_id, f.store_id, i.dept_id, i.cat_id, f.date, f.units,
           f.sell_price, f.sell_price / ss.mean_price AS price_ratio, f.price_change_wow,
           f.wday, f.month, dayofmonth(f.date) AS mday, weekofyear(f.date) AS week,
           f.event_type_1, CAST(f.snap AS INTEGER) AS snap,
           date_diff('day', ss.launch_date, f.date) AS age_days,
           f.lag_28, f.lag_35, f.lag_42, f.lag_49, f.lag_56,
           f.rmean_7, f.rmean_28, f.rmean_56, f.rstd_28, f.zero_frac_28
    FROM feats f
    JOIN items i USING (item_id)
    JOIN series_stats ss USING (item_id, store_id)
    WHERE f.date BETWEEN ? AND ?
    ORDER BY f.store_id, f.item_id, f.date
    """
    df = con.execute(sql, params).df()
    for col in CATEGORICAL:
        df[col] = df[col].astype("category")
    df[NUMERIC] = df[NUMERIC].astype("float32")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def last_actual_date(con: duckdb.DuckDBPyConnection) -> date:
    return con.execute("SELECT max(date) FROM sales").fetchone()[0]
