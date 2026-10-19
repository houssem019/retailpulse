"""Launch the MCP server over stdio (exactly as Claude Code does) and call a few tools on the real warehouse."""

from __future__ import annotations

import sys

import anyio
from mcp import Client, StdioServerParameters


async def main() -> None:
    params = StdioServerParameters(command=sys.executable, args=["-m", "retailpulse.mcp_server"])
    async with Client(params) as client:
        tools = await client.list_tools()
        print("tools:", [t.name for t in tools.tools])
        calls = [
            (
                "run_sql",
                {
                    "query": "SELECT signal, count(*) AS n FROM inventory_signals GROUP BY 1 ORDER BY 2 DESC",
                    "purpose": "inventory health",
                },
            ),
            ("run_sql", {"query": "DROP TABLE sales", "purpose": "attack"}),
            ("forecast_item", {"item_id": "FOODS_3_090", "store_id": "CA_1", "horizon": 7}),
        ]
        for name, args in calls:
            res = await client.call_tool(name, args)
            print(f"\n{name} (is_error={res.is_error}):\n  {res.content[0].text[:300]}")


if __name__ == "__main__":
    anyio.run(main)
