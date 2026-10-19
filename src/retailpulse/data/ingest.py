"""Build the RetailPulse DuckDB warehouse from the raw M5 CSVs.

Raw M5 is "wide": one row per item×store with 1,941 day columns (d_1..d_1941).
We reshape it into a small star schema that is easy for both humans and an LLM to query:

    items(item_id, dept_id, cat_id)                 -- product hierarchy
    stores(store_id, state_id)                      -- location hierarchy
    calendar(date, d, wm_yr_wk, weekday, ..., snap_ca, snap_tx, snap_wi)
    prices(store_id, item_id, wm_yr_wk, sell_price) -- weekly shelf price
    sales(item_id, store_id, date, units, sell_price, revenue)  -- the fact table

Design choices (see docs/BUILD_LOG.md):
* Everything runs as SQL inside DuckDB (UNPIVOT + joins), never in pandas, so the full
  59M-row reshape fits comfortably in memory on a laptop.
* `sales` is INNER-joined to `prices`: before an item is launched in a store there is no
  price, and the zeros in the raw file are structural, not real "zero demand" days.
  Dropping them shrinks the table and stops averages being dragged towards zero.
* `revenue` is pre-computed in the fact table so text-to-SQL questions about dollars do not
  need a three-way join the model could get wrong.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb

from retailpulse.config import settings

SALES_FILE = "sales_train_evaluation.csv"
CALENDAR_FILE = "calendar.csv"
PRICES_FILE = "sell_prices.csv"


def _csv(raw_dir: Path, name: str) -> str:
    path = raw_dir / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — download M5 first (see README).")
    return path.as_posix()


def build_database(raw_dir: Path, db_path: Path, stores: list[str] | None = None) -> dict[str, int]:
    """(Re)build the warehouse. Optionally restrict to a list of store_ids to keep it small.

    Returns row counts per table.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    con = duckdb.connect(str(db_path))

    sales_csv = _csv(raw_dir, SALES_FILE)
    store_filter = ""
    params: list = []
    if stores:
        store_filter = "WHERE store_id IN (SELECT unnest(?::VARCHAR[]))"
        params = [stores]

    con.execute(
        f"CREATE TEMP TABLE raw_sales AS SELECT * FROM read_csv_auto('{sales_csv}') {store_filter}", params
    )

    con.execute(f"""
        CREATE TABLE calendar AS
        SELECT CAST(date AS DATE) AS date, d, wm_yr_wk, weekday, wday, month, year,
               event_name_1, event_type_1, event_name_2, event_type_2,
               CAST(snap_CA AS BOOLEAN) AS snap_ca, CAST(snap_TX AS BOOLEAN) AS snap_tx,
               CAST(snap_WI AS BOOLEAN) AS snap_wi
        FROM read_csv_auto('{_csv(raw_dir, CALENDAR_FILE)}')
    """)
    con.execute("CREATE TABLE items AS SELECT DISTINCT item_id, dept_id, cat_id FROM raw_sales ORDER BY 1")
    con.execute("CREATE TABLE stores AS SELECT DISTINCT store_id, state_id FROM raw_sales ORDER BY 1")
    con.execute(f"""
        CREATE TABLE prices AS
        SELECT p.store_id, p.item_id, p.wm_yr_wk, p.sell_price
        FROM read_csv_auto('{_csv(raw_dir, PRICES_FILE)}') p
        SEMI JOIN raw_sales s ON s.item_id = p.item_id AND s.store_id = p.store_id
    """)
    con.execute("""
        CREATE TABLE sales AS
        WITH long AS (
            UNPIVOT (SELECT item_id, store_id, COLUMNS('^d_\\d+$') FROM raw_sales)
            ON COLUMNS('^d_\\d+$') INTO NAME d VALUE units
        )
        SELECT l.item_id, l.store_id, c.date,
               CAST(l.units AS INTEGER) AS units,
               p.sell_price,
               ROUND(l.units * p.sell_price, 2) AS revenue
        FROM long l
        JOIN calendar c ON c.d = l.d
        JOIN prices p ON p.store_id = l.store_id AND p.item_id = l.item_id AND p.wm_yr_wk = c.wm_yr_wk
        ORDER BY l.store_id, l.item_id, c.date
    """)
    con.execute("DROP TABLE raw_sales")

    counts = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
              for t in ("items", "stores", "calendar", "prices", "sales")}
    con.close()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=settings.raw_dir)
    parser.add_argument("--db", type=Path, default=settings.db_path)
    parser.add_argument("--stores", nargs="*", help="Only ingest these store_ids, e.g. CA_1 TX_1")
    args = parser.parse_args()

    start = time.perf_counter()
    counts = build_database(args.raw_dir, args.db, args.stores)
    print(f"Built {args.db} in {time.perf_counter() - start:.1f}s")
    for table, n in counts.items():
        print(f"  {table:<9} {n:>12,}")


if __name__ == "__main__":
    main()
