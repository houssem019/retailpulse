"""Cut a tiny, committed sample of raw M5 so tests and CI can run the real ingest path without Kaggle.

Sample = 3 stores (one per state) × 5 items from each of the 7 departments = 105 series,
over the last ~2 years of history, in the exact raw M5 file formats.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from retailpulse.config import PROJECT_ROOT, settings

STORES = ["CA_1", "TX_1", "WI_1"]
ITEMS_PER_DEPT = 5
FIRST_DAY = 1941 - 730 + 1  # keep the last 730 days


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=settings.raw_dir)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "tests" / "fixtures" / "m5_sample")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    raw = args.raw_dir.as_posix()
    out = args.out.as_posix()
    con = duckdb.connect()

    day_cols = ", ".join(f"d_{i}" for i in range(FIRST_DAY, 1942))
    con.execute(f"""
        CREATE TABLE s AS
        WITH ranked AS (
            SELECT *, dense_rank() OVER (PARTITION BY dept_id ORDER BY item_id) AS rk
            FROM read_csv_auto('{raw}/sales_train_evaluation.csv')
            WHERE store_id IN ({", ".join(f"'{s}'" for s in STORES)})
        )
        SELECT id, item_id, dept_id, cat_id, store_id, state_id, {day_cols}
        FROM ranked WHERE rk <= {ITEMS_PER_DEPT} ORDER BY id
    """)
    con.execute(f"COPY s TO '{out}/sales_train_evaluation.csv' (HEADER)")
    con.execute(f"""
        COPY (SELECT * FROM read_csv_auto('{raw}/calendar.csv', all_varchar=true)
              WHERE CAST(substr(d, 3) AS INTEGER) >= {FIRST_DAY})
        TO '{out}/calendar.csv' (HEADER)
    """)
    con.execute(f"""
        COPY (SELECT p.* FROM read_csv_auto('{raw}/sell_prices.csv') p
              SEMI JOIN s ON s.item_id = p.item_id AND s.store_id = p.store_id
              ORDER BY p.store_id, p.item_id, p.wm_yr_wk)
        TO '{out}/sell_prices.csv' (HEADER)
    """)
    n = con.execute("SELECT count(*) FROM s").fetchone()[0]
    print(f"Wrote {n} series to {args.out}")


if __name__ == "__main__":
    main()
