"""The MCP server, exercised through a real MCP client connected in-process."""

import json

import anyio
import pytest
from mcp import Client
from mcp.types import ElicitResult

from retailpulse.mcp_server import build_server

EXPENSIVE_SQL = "SELECT count(*) AS n FROM sales a JOIN sales b USING (item_id, store_id, date)"


def _text(result) -> str:
    return result.content[0].text


def _call(server, tool: str, args: dict, elicit=None):
    async def go():
        async with Client(server, elicitation_callback=elicit) as client:
            return await client.call_tool(tool, args)

    return anyio.run(go)


@pytest.fixture(scope="module")
def server(toolbox):
    return build_server(toolbox)


def test_lists_tools_with_read_only_annotations(server):
    async def go():
        async with Client(server) as client:
            return (await client.list_tools()).tools

    tools = {t.name: t for t in anyio.run(go)}
    assert set(tools) == {"get_schema", "run_sql", "forecast_item", "price_what_if", "create_chart"}
    assert tools["run_sql"].annotations.read_only_hint is True
    assert "query" in tools["run_sql"].input_schema["properties"]


def test_schema_and_sql(server):
    assert "inventory_signals" in _text(_call(server, "get_schema", {}))
    res = _call(server, "run_sql", {"query": "SELECT count(*) AS n FROM stores", "purpose": "count"})
    assert not res.is_error and json.loads(_text(res))["rows"] == [[3]]


def test_guardrail_errors_reach_the_model(server):
    res = _call(server, "run_sql", {"query": "DROP TABLE sales", "purpose": "x"})
    assert res.is_error and "read-only" in _text(res)


def test_forecast_tools(server):
    res = _call(server, "forecast_item", {"item_id": "FOODS_1_001", "store_id": "CA_1", "horizon": 7})
    assert not res.is_error and len(json.loads(_text(res))["daily"]) == 7
    res = _call(
        server, "price_what_if", {"item_id": "FOODS_1_001", "store_id": "CA_1", "price_change_pct": -0.2}
    )
    assert json.loads(_text(res))["delta_pct"] >= 0


def test_expensive_query_without_elicitation_asks_model_to_confirm(server):
    res = _call(server, "run_sql", {"query": EXPENSIVE_SQL, "purpose": "x"})
    assert res.is_error and "confirm_expensive=true" in _text(res)
    res = _call(server, "run_sql", {"query": EXPENSIVE_SQL, "purpose": "x", "confirm_expensive": True})
    assert not res.is_error


@pytest.mark.parametrize("approve", [True, False])
def test_expensive_query_uses_elicitation_when_supported(server, approve):
    prompts = []

    async def elicit(context, params):
        prompts.append(params.message)
        return ElicitResult(action="accept", content={"approve": approve})

    res = _call(server, "run_sql", {"query": EXPENSIVE_SQL, "purpose": "x"}, elicit=elicit)
    assert len(prompts) == 1 and "scans" in prompts[0]
    assert res.is_error is (not approve)
