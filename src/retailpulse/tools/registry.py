"""The agent's tool surface: JSON-schema definitions + one executor.

The same Toolbox backs both the LangGraph agent and the MCP server, so a tool behaves identically
whichever client calls it. Tool *descriptions* are part of the prompt; they say when to use each tool,
not only what it does, because tool-selection accuracy is one of the things we evaluate.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from retailpulse.config import settings
from retailpulse.tools.chart import render_chart
from retailpulse.tools.forecast import ForecastClient, ForecastServiceError
from retailpulse.tools.sql import SqlRejected, SqlSandbox


def _tool(name: str, description: str, properties: dict) -> dict:
    return {
        "name": name,
        "description": description,
        "strict": True,  # Claude's tool input is guaranteed to match the schema
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    }


_ITEM = {"type": "string", "description": "Item id, e.g. FOODS_3_090"}
_STORE = {"type": "string", "description": "Store id, e.g. CA_1"}
_HORIZON = {"type": "integer", "description": "Days ahead, 1-28"}

TOOL_SPECS = [
    _tool(
        "run_sql",
        "Run one read-only DuckDB SELECT query against the retail warehouse and get rows back. Use it for any "
        "question about historical sales, revenue, prices, calendar effects, inventory signals, stored "
        "forecasts, and forecast accuracy at any aggregation level. Aggregate in SQL; results are capped at "
        "200 rows. Queries that would scan very large amounts of data pause for human approval.",
        {
            "query": {"type": "string", "description": "A single DuckDB SELECT statement."},
            "purpose": {"type": "string", "description": "One short sentence: what this query answers."},
        },
    ),
    _tool(
        "forecast_item",
        "Get a fresh daily demand forecast for ONE item at ONE store from the deployed forecasting model "
        "(next 1-28 days after the last actual date). Use for single-item forecast questions. For "
        "aggregated forecasts (department, category, store totals) query the `forecasts` table with run_sql.",
        {"item_id": _ITEM, "store_id": _STORE, "horizon": _HORIZON},
    ),
    _tool(
        "price_what_if",
        "Estimate how demand for ONE item at ONE store would change over the next 1-28 days if its price "
        "changed by price_change_pct (e.g. -0.1 = 10% cut). Returns baseline vs scenario totals. Use for "
        "pricing / promotion what-if questions.",
        {
            "item_id": _ITEM,
            "store_id": _STORE,
            "price_change_pct": {"type": "number", "description": "Relative change: -0.2 = 20% cut."},
            "horizon": _HORIZON,
        },
    ),
    _tool(
        "create_chart",
        "Render a chart from a read-only SELECT query and save it as an HTML file. Use only when the user "
        "asks for a chart/plot/visualisation, or a trend is much clearer as a picture.",
        {
            "query": {"type": "string", "description": "SELECT returning the columns to plot."},
            "chart_type": {"type": "string", "enum": ["line", "bar", "scatter"]},
            "x": {"type": "string", "description": "Column for the x axis."},
            "y": {"type": "string", "description": "Column for the y axis."},
            "color": {"type": "string", "description": 'Column to split series by, or "" for one series.'},
            "title": {"type": "string"},
        },
    ),
]
TOOL_NAMES = [t["name"] for t in TOOL_SPECS]
SQL_TOOLS = {"run_sql", "create_chart"}


@dataclass
class ToolOutcome:
    content: str  # what the model sees
    is_error: bool = False
    meta: dict = field(default_factory=dict)  # for logs/evals: sql, rows, chart path, timings


@dataclass
class ApprovalRequest:
    tool: str
    sql: str
    estimated_rows_scanned: int


class Toolbox:
    def __init__(self, sandbox: SqlSandbox, forecasts: ForecastClient, charts_dir: Path):
        self.sandbox = sandbox
        self.forecasts = forecasts
        self.charts_dir = charts_dir

    @classmethod
    def from_settings(cls) -> Toolbox:
        return cls(
            SqlSandbox(
                settings.db_path,
                max_rows=settings.sql_max_rows,
                timeout_s=settings.sql_timeout_s,
                approval_rows=settings.sql_approval_rows,
            ),
            ForecastClient.from_url(settings.forecast_api_url),
            settings.charts_dir,
        )

    def approval_needed(self, name: str, args: dict) -> ApprovalRequest | None:
        """Check before executing: does this call need a human to approve it?"""
        if name not in SQL_TOOLS:
            return None
        try:
            checked = self.sandbox.check(args.get("query", ""))
        except SqlRejected:
            return None  # invalid SQL is rejected (and explained to the model) at execution time
        if checked.needs_approval:
            return ApprovalRequest(name, checked.sql, checked.estimated_rows_scanned)
        return None

    def execute(self, name: str, args: dict, *, approved: bool = False) -> ToolOutcome:
        start = time.perf_counter()
        try:
            outcome = self._dispatch(name, args, approved)
        except (SqlRejected, ForecastServiceError, KeyError, ValueError) as e:
            outcome = ToolOutcome(f"Error: {e}", is_error=True)
        outcome.meta.setdefault("elapsed_ms", round((time.perf_counter() - start) * 1000, 1))
        return outcome

    def _dispatch(self, name: str, args: dict, approved: bool) -> ToolOutcome:
        if name == "run_sql":
            res = self.sandbox.run(args["query"], approved=approved)
            return ToolOutcome(
                res.to_model_payload(),
                meta={
                    "sql": res.sql,
                    "columns": res.columns,
                    "rows": res.rows,
                    "row_count": res.row_count,
                    "estimated_rows_scanned": res.estimated_rows_scanned,
                },
            )
        if name == "forecast_item":
            out = self.forecasts.forecast(args["item_id"], args["store_id"], int(args.get("horizon") or 28))
            return ToolOutcome(json.dumps(out), meta=out)
        if name == "price_what_if":
            out = self.forecasts.what_if(
                args["item_id"],
                args["store_id"],
                float(args["price_change_pct"]),
                int(args.get("horizon") or 28),
            )
            return ToolOutcome(json.dumps(out), meta=out)
        if name == "create_chart":
            res = self.sandbox.run(args["query"], approved=approved)
            out = render_chart(
                res,
                chart_type=args["chart_type"],
                x=args["x"],
                y=args["y"],
                color=args.get("color") or None,
                title=args["title"],
                out_dir=self.charts_dir,
            )
            return ToolOutcome(json.dumps(out), meta=out | {"sql": res.sql})
        raise ValueError(f"Unknown tool '{name}'. Available: {', '.join(TOOL_NAMES)}")
