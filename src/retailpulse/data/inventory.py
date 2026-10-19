"""Inventory-health signals derived from sales (M5 has no on-hand stock data).

A run of zero-sales days is only suspicious if the item normally sells. If an item sells λ units/day
on average (Poisson), the chance of k consecutive zero days by pure chance is exp(-λ·k). When that
probability drops below 1% while the item is still listed (has a price), the shelf is very likely
empty: an out-of-stock or "phantom inventory" signal, the same idea retailers use to spot
shelf gaps from point-of-sale data alone.

Zero runs longer than 8 weeks are labelled `possibly_discontinued` instead: M5 keeps publishing a
price for some items long after they stop selling, so very long runs are usually delistings.
"""

from __future__ import annotations

import duckdb

OOS_PROBABILITY = 0.01
RATE_WINDOW_DAYS = 90
DISCONTINUED_AFTER_DAYS = 56


def build_inventory_signals(con: duckdb.DuckDBPyConnection) -> int:
    con.execute(f"""
        CREATE OR REPLACE TABLE inventory_signals AS
        WITH as_of AS (SELECT max(date) AS d FROM sales),
        series AS (
            SELECT item_id, store_id,
                   max(date) FILTER (WHERE units > 0) AS last_sale_date,
                   max(date) AS last_listed_date
            FROM sales GROUP BY 1, 2
        ),
        rate AS (  -- demand rate in the 90 days up to the last sale (the zero run itself excluded)
            SELECT s.item_id, s.store_id, avg(x.units) AS avg_daily_units
            FROM series s
            JOIN sales x ON x.item_id = s.item_id AND x.store_id = s.store_id
                        AND x.date BETWEEN s.last_sale_date - INTERVAL {RATE_WINDOW_DAYS - 1} DAY AND s.last_sale_date
            GROUP BY 1, 2
        )
        SELECT s.item_id, s.store_id, i.dept_id, i.cat_id,
               s.last_sale_date,
               date_diff('day', s.last_sale_date, a.d) AS days_since_last_sale,
               round(r.avg_daily_units, 3) AS avg_daily_units,
               exp(-r.avg_daily_units * date_diff('day', s.last_sale_date, a.d)) AS prob_zero_run_by_chance,
               CASE
                   WHEN s.last_listed_date < a.d THEN 'not_listed'
                   WHEN exp(-r.avg_daily_units * date_diff('day', s.last_sale_date, a.d)) >= {OOS_PROBABILITY} THEN 'ok'
                   -- months of silence on a priced item is more likely a delisting the price feed missed
                   WHEN date_diff('day', s.last_sale_date, a.d) > {DISCONTINUED_AFTER_DAYS} THEN 'possibly_discontinued'
                   ELSE 'likely_out_of_stock'
               END AS signal,
               round(r.avg_daily_units * date_diff('day', s.last_sale_date, a.d), 1) AS est_lost_units,
               a.d AS as_of_date
        FROM series s
        CROSS JOIN as_of a
        JOIN rate r USING (item_id, store_id)
        JOIN items i USING (item_id)
    """)
    return con.execute("SELECT count(*) FROM inventory_signals").fetchone()[0]
