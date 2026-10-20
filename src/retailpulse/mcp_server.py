"""RetailPulse as a Model Context Protocol (MCP) server.

The same Toolbox the LangGraph agent uses is exposed over MCP, so any MCP client (Claude Code, Claude
Desktop, IDE assistants, other agents) can query the warehouse, call the forecast service and draw
charts with the same guardrails. In that setup the *client's* model is the agent; this server only
provides tools, the schema and usage instructions.

Run (stdio, for local clients):     uv run python -m retailpulse.mcp_server
Run (HTTP, for remote clients):     uv run python -m retailpulse.mcp_server --http --port 8002
Register with Claude Code:          claude mcp add retailpulse -- uv --directory <repo> run python -m retailpulse.mcp_server

Expensive queries: if the client supports MCP elicitation, the server asks the user directly; otherwise
the tool returns an error asking the model to get the user's OK and retry with confirm_expensive=true.
"""

from __future__ import annotations

import argparse
from typing import Annotated

import anyio
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import (
    ElicitRequest,
    ElicitRequestFormParams,
    ElicitResult,
    InputRequiredResult,
    ToolAnnotations,
)
from pydantic import BaseModel, Field

from retailpulse.tools.registry import ApprovalRequest, Toolbox
from retailpulse.tools.schema import describe_schema

INSTRUCTIONS = """RetailPulse: retail analytics over the M5 Walmart dataset (10 stores in CA/TX/WI, daily \
item-level sales 2011-01-29 to 2016-05-22). Call get_schema first to learn the tables. Use run_sql for \
history, revenue, prices, events, inventory signals, stored forecasts and forecast accuracy (aggregate in \
SQL). Use forecast_item / price_what_if for one item at one store. There is no on-hand inventory data: \
stock health is inferred (inventory_signals). Treat the last actual date as "today". Only cite numbers \
returned by these tools."""

READ_ONLY = ToolAnnotations(
    read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
)


class ApproveQuery(BaseModel):
    approve: bool = Field(description="Run this query?")


APPROVAL_KEY = "approve_query"
MRTR_PROTOCOL = "2026-07-28"  # from this protocol version, servers ask for input via InputRequiredResult


def _approval_message(req: ApprovalRequest) -> str:
    return f"RetailPulse wants to run a query that scans ~{req.estimated_rows_scanned:,} rows:\n\n{req.sql}\n\nRun it?"


async def _approve(ctx: Context, req: ApprovalRequest, confirmed: bool) -> bool | InputRequiredResult:
    """Ask the human, using whatever the connected client supports.

    * 2026-07-28+ clients: return InputRequiredResult with an elicitation form; the client asks the user
      and retries the same call carrying the answer in ctx.input_responses.
    * Older clients with elicitation: server-initiated elicitation/create request (ctx.elicit).
    * Clients without elicitation: fail with instructions so the *model* asks the user in chat and
      retries with confirm_expensive=true.
    """
    if confirmed:
        return True
    answer = (ctx.input_responses or {}).get(APPROVAL_KEY)
    if answer is not None:
        return (
            isinstance(answer, ElicitResult)
            and answer.action == "accept"
            and bool((answer.content or {}).get("approve"))
        )

    caps = ctx.client_capabilities
    if caps is None or caps.elicitation is None:
        raise ToolError(
            f"This query would scan ~{req.estimated_rows_scanned:,} rows and needs the user's approval. Ask the "
            "user; if they agree, call again with confirm_expensive=true, otherwise narrow the query."
        )
    if (ctx.protocol_version or "") >= MRTR_PROTOCOL:
        form = ElicitRequestFormParams(
            message=_approval_message(req), requested_schema=ApproveQuery.model_json_schema()
        )
        return InputRequiredResult(input_requests={APPROVAL_KEY: ElicitRequest(params=form)})
    result = await ctx.elicit(_approval_message(req), ApproveQuery)
    return result.action == "accept" and result.data.approve


def build_server(toolbox: Toolbox) -> MCPServer:
    mcp = MCPServer(
        "retailpulse", title="RetailPulse retail analytics", instructions=INSTRUCTIONS, version="0.1.0"
    )
    schema_text = describe_schema(toolbox.sandbox._con.cursor())

    async def _run(name: str, args: dict, approved: bool = False) -> str:
        outcome = await anyio.to_thread.run_sync(lambda: toolbox.execute(name, args, approved=approved))
        if outcome.is_error:
            raise ToolError(outcome.content.removeprefix("Error: "))
        return outcome.content

    async def _run_sql_tool(name: str, args: dict, ctx: Context, confirm: bool) -> str | InputRequiredResult:
        req = toolbox.approval_needed(name, args)
        approved = False
        if req is not None:
            approved = await _approve(ctx, req, confirm)
            if isinstance(approved, InputRequiredResult):
                return approved
            if not approved:
                raise ToolError(
                    "The user declined this query. Narrow it (store, date range, category) instead."
                )
        return await _run(name, args, approved)

    @mcp.tool(annotations=READ_ONLY, structured_output=False)
    def get_schema() -> str:
        """Describe the warehouse tables, columns, valid category values and date range. Call this first."""
        return schema_text

    @mcp.resource("retailpulse://schema", name="schema", mime_type="text/markdown")
    def schema_resource() -> str:
        """Warehouse schema with business definitions."""
        return schema_text

    @mcp.tool(annotations=READ_ONLY, structured_output=False)
    async def run_sql(
        query: Annotated[str, Field(description="A single read-only DuckDB SELECT statement.")],
        purpose: Annotated[str, Field(description="One short sentence: what this query answers.")],
        ctx: Context,
        confirm_expensive: Annotated[bool, Field(description="Only true after the user approved.")] = False,
    ) -> str | InputRequiredResult:
        """Run one read-only SQL query on the retail warehouse (sales, items, stores, calendar, prices,
        forecasts, forecast_accuracy, inventory_signals). Aggregate in SQL; at most 200 rows come back.
        Very large scans need the user's approval."""
        return await _run_sql_tool("run_sql", {"query": query, "purpose": purpose}, ctx, confirm_expensive)

    @mcp.tool(annotations=READ_ONLY, structured_output=False)
    async def forecast_item(
        item_id: Annotated[str, Field(description="Item id, e.g. FOODS_3_090")],
        store_id: Annotated[str, Field(description="Store id, e.g. CA_1")],
        horizon: Annotated[int, Field(ge=1, le=28, description="Days ahead, 1-28")] = 28,
    ) -> str:
        """Daily demand forecast for ONE item at ONE store from the deployed LightGBM model, for the days after
        the last actual date. For aggregated forecasts query the `forecasts` table with run_sql."""
        return await _run("forecast_item", {"item_id": item_id, "store_id": store_id, "horizon": horizon})

    @mcp.tool(annotations=READ_ONLY, structured_output=False)
    async def price_what_if(
        item_id: Annotated[str, Field(description="Item id, e.g. FOODS_3_090")],
        store_id: Annotated[str, Field(description="Store id, e.g. CA_1")],
        price_change_pct: Annotated[float, Field(ge=-0.9, le=2.0, description="-0.2 = 20% price cut")],
        horizon: Annotated[int, Field(ge=1, le=28, description="Days ahead, 1-28")] = 28,
    ) -> str:
        """Estimate how one item's demand at one store changes if its price changes (baseline vs scenario).
        The model is constrained so a price cut never lowers demand; the estimate is correlational."""
        return await _run(
            "price_what_if",
            {
                "item_id": item_id,
                "store_id": store_id,
                "price_change_pct": price_change_pct,
                "horizon": horizon,
            },
        )

    @mcp.tool(
        annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False),
        structured_output=False,
    )
    async def create_chart(
        query: Annotated[str, Field(description="SELECT returning the columns to plot.")],
        chart_type: Annotated[str, Field(pattern="^(line|bar|scatter)$", description="line | bar | scatter")],
        x: Annotated[str, Field(description="Column for the x axis.")],
        y: Annotated[str, Field(description="Column for the y axis.")],
        title: str,
        ctx: Context,
        color: Annotated[str, Field(description="Column to split series by, or empty.")] = "",
        confirm_expensive: Annotated[bool, Field(description="Only true after the user approved.")] = False,
    ) -> str | InputRequiredResult:
        """Render a Plotly chart from a read-only query and save it as an HTML file; returns the file path.
        Use when the user asks for a chart or plot."""
        args = {"query": query, "chart_type": chart_type, "x": x, "y": y, "color": color, "title": title}
        return await _run_sql_tool("create_chart", args, ctx, confirm_expensive)

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(description="RetailPulse MCP server")
    parser.add_argument("--http", action="store_true", help="Serve Streamable HTTP instead of stdio")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 inside a container")
    args = parser.parse_args()
    server = build_server(Toolbox.from_settings())
    if args.http:
        server.run("streamable-http", host=args.host, port=args.port)
    else:
        server.run("stdio")


if __name__ == "__main__":
    main()
