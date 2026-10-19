import json

import pytest

from retailpulse.tools.sql import SqlRejected

MALICIOUS = [
    "DROP TABLE sales",
    "DELETE FROM sales",
    "INSERT INTO items VALUES ('x', 'y', 'z')",
    "SELECT 1; DROP TABLE sales",
    "SELECT * FROM read_csv_auto('data/raw/calendar.csv')",
    "SELECT * FROM 'data/raw/calendar.csv'",
    "ATTACH 'other.db'",
    "COPY sales TO 'leak.csv'",
    "PRAGMA database_list",
    "SELECT getenv('ANTHROPIC_API_KEY')",
    "SELECT * FROM duckdb_secrets()",
    "SELECT * FROM information_schema.tables",
]


@pytest.mark.parametrize("sql", MALICIOUS)
def test_sandbox_rejects_unsafe_sql(toolbox, sql):
    with pytest.raises(SqlRejected):
        toolbox.sandbox.run(sql)


def test_sandbox_allows_ctes_joins_and_injects_limit(toolbox):
    res = toolbox.sandbox.run(
        "WITH r AS (SELECT store_id, sum(revenue) rev FROM sales GROUP BY 1) "
        "SELECT s.state_id, sum(rev) FROM r JOIN stores s USING (store_id) GROUP BY 1 ORDER BY 1"
    )
    assert res.columns[0] == "state_id" and res.row_count == 3
    assert res.sql.endswith("LIMIT 201")


def test_sandbox_caps_rows(toolbox):
    res = toolbox.sandbox.run("SELECT * FROM sales")
    assert res.row_count == 200 and res.truncated
    assert json.loads(res.to_model_payload())["truncated"] is True


def test_expensive_query_requires_approval(toolbox):
    sql = "SELECT count(*) FROM sales a JOIN sales b USING (item_id, store_id, date)"
    assert toolbox.approval_needed("run_sql", {"query": sql}) is not None
    with pytest.raises(SqlRejected, match="approval"):
        toolbox.sandbox.run(sql)
    assert toolbox.sandbox.run(sql, approved=True).rows[0][0] > 0


def test_timeout_interrupts_long_queries(toolbox, monkeypatch):
    monkeypatch.setattr(toolbox.sandbox, "timeout_s", 0.2)
    with pytest.raises(SqlRejected, match="time limit"):
        toolbox.sandbox.run("SELECT count(*) FROM sales a, sales b, sales c", approved=True)


def test_tool_errors_are_returned_not_raised(toolbox):
    out = toolbox.execute("run_sql", {"query": "SELECT nope FROM sales", "purpose": "x"})
    assert out.is_error and "SQL error" in out.content
    out = toolbox.execute("forecast_item", {"item_id": "NOPE", "store_id": "CA_1", "horizon": 7})
    assert out.is_error


def test_forecast_and_what_if_tools(toolbox):
    out = toolbox.execute("forecast_item", {"item_id": "FOODS_1_001", "store_id": "CA_1", "horizon": 7})
    assert not out.is_error and len(json.loads(out.content)["daily"]) == 7
    out = toolbox.execute(
        "price_what_if",
        {"item_id": "FOODS_1_001", "store_id": "CA_1", "price_change_pct": -0.2, "horizon": 28},
    )
    body = json.loads(out.content)
    assert body["scenario_total"] >= body["baseline_total"]


def test_chart_tool_writes_html(toolbox):
    out = toolbox.execute(
        "create_chart",
        {
            "query": "SELECT date_trunc('month', date) m, sum(units) u FROM sales GROUP BY 1 ORDER BY 1",
            "chart_type": "line",
            "x": "m",
            "y": "u",
            "color": "",
            "title": "Monthly units",
        },
    )
    assert not out.is_error
    assert out.meta["chart_path"].endswith(".html")


def test_inventory_signals_table(toolbox):
    res = toolbox.sandbox.run("SELECT signal, count(*) FROM inventory_signals GROUP BY 1")
    signals = dict(res.rows)
    assert set(signals) <= {"ok", "likely_out_of_stock", "possibly_discontinued", "not_listed"}
    assert sum(signals.values()) == 105
