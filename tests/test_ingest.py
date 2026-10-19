from pathlib import Path

import duckdb
import pytest

from retailpulse.data.ingest import build_database

FIXTURE = Path(__file__).parent / "fixtures" / "m5_sample"


@pytest.fixture(scope="module")
def db(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("db") / "test.duckdb"
    build_database(FIXTURE, path)
    return path


def test_star_schema_tables_exist(db):
    con = duckdb.connect(str(db), read_only=True)
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert {"items", "stores", "calendar", "prices", "sales"} <= tables


def test_sales_is_long_and_unique_per_series_day(db):
    con = duckdb.connect(str(db), read_only=True)
    total, distinct = con.execute(
        "SELECT count(*), count(DISTINCT (item_id, store_id, date)) FROM sales"
    ).fetchone()
    assert total == distinct > 0


def test_prelaunch_days_are_dropped(db):
    """Every sales row must have a price: days before an item was listed are not demand."""
    con = duckdb.connect(str(db), read_only=True)
    assert con.execute("SELECT count(*) FROM sales WHERE sell_price IS NULL").fetchone()[0] == 0


def test_revenue_matches_units_times_price(db):
    con = duckdb.connect(str(db), read_only=True)
    bad = con.execute("SELECT count(*) FROM sales WHERE abs(revenue - units * sell_price) > 0.01").fetchone()[0]
    assert bad == 0


def test_store_filter(tmp_path):
    counts = build_database(FIXTURE, tmp_path / "ca.duckdb", stores=["CA_1"])
    con = duckdb.connect(str(tmp_path / "ca.duckdb"), read_only=True)
    assert con.execute("SELECT list(DISTINCT store_id) FROM sales").fetchone()[0] == ["CA_1"]
    assert counts["stores"] == 1
