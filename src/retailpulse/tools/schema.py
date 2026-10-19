"""Semantic schema description given to the LLM.

Text-to-SQL accuracy depends far more on *what the model is told about the data* than on the model.
Raw column names are not enough: the model must know units vs revenue, what a "listed" day is,
which values exist for categorical columns, and the date range. This text is built once at agent
start (it is stable, so it sits in the cached part of the prompt).
"""

from __future__ import annotations

import duckdb

TABLE_NOTES = {
    "sales": "Fact table: one row per item × store × day while the item is listed (has a price). "
    "units = units sold that day, sell_price = shelf price (USD), revenue = units * sell_price.",
    "items": "Product hierarchy: item_id -> dept_id -> cat_id.",
    "stores": "Store -> state hierarchy.",
    "calendar": "One row per day. wm_yr_wk = Walmart week id (join key to prices). weekday is a name "
    "('Monday'...). event_name_1/event_type_1 = holiday/event (e.g. Sporting, Cultural, National, "
    "Religious) or NULL. snap_ca/snap_tx/snap_wi = SNAP food-stamp days per state.",
    "prices": "Weekly shelf price per item × store × wm_yr_wk.",
    "forecasts": "Demand-model output. kind='backtest' = predictions for the last 28 actual days "
    "(compare with sales); kind='future' = forecast for the 28 days after the last actual date. "
    "Only covers the stores the model was trained on.",
    "forecast_accuracy": "View: backtest forecasts joined with actual units (actual, error = forecast - "
    "actual, abs_error). Use WAPE = sum(abs_error)/sum(actual) and bias = sum(forecast)/sum(actual) - 1.",
    "inventory_signals": "One row per item × store, as of the last actual date. M5 has NO on-hand "
    "inventory, so stock health is inferred from sales: signal in ('ok', 'likely_out_of_stock', "
    "'possibly_discontinued', 'not_listed'); days_since_last_sale; avg_daily_units (before the zero run); "
    "prob_zero_run_by_chance; est_lost_units = estimated units lost during the zero run.",
}

ENUM_COLUMNS = {
    "items": ["cat_id", "dept_id"],
    "stores": ["store_id", "state_id"],
    "calendar": ["event_type_1"],
}


def describe_schema(con: duckdb.DuckDBPyConnection) -> str:
    tables = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    lines = []
    for table, note in TABLE_NOTES.items():
        if table not in tables:
            continue
        cols = con.execute(
            "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = ? "
            "ORDER BY ordinal_position",
            [table],
        ).fetchall()
        lines.append(f"### {table}\n{note}\nColumns: " + ", ".join(f"{c} {t}" for c, t in cols))
        for col in ENUM_COLUMNS.get(table, []):
            values = [
                r[0]
                for r in con.execute(
                    f"SELECT DISTINCT {col} FROM {table} WHERE {col} IS NOT NULL ORDER BY 1"
                ).fetchall()
            ]
            lines.append(f"Values of {col}: {', '.join(map(str, values))}")
    lo, hi = con.execute("SELECT min(date), max(date) FROM sales").fetchone()
    lines.append(f"\nActual sales cover {lo} to {hi} (the last actual date is {hi}).")
    if "forecasts" in tables:
        stores = [r[0] for r in con.execute("SELECT DISTINCT store_id FROM forecasts ORDER BY 1").fetchall()]
        lines.append(f"Forecasts exist for stores: {', '.join(stores)}.")
    return "\n".join(lines)
