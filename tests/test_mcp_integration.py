"""End-to-end MCP + Text-to-SQL integration test."""
import sys, os, sqlite3, tempfile, json, asyncio

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_full_mcp_flow():
    """End-to-end: MCPClient -> SQLite MCP Server -> tool calls -> result"""
    from mcp.client import MCPClient
    from mcp.transports.stdio import StdioTransport

    async def _run():
        db_path = tempfile.mktemp(suffix=".db")
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE financials (company TEXT, year INT, net_profit REAL, revenue REAL)"
        )
        conn.execute("INSERT INTO financials VALUES ('maotai', 2024, 747.3, 1500.0)")
        conn.execute("INSERT INTO financials VALUES ('icbc', 2024, 698.2, 1200.0)")
        conn.execute("INSERT INTO financials VALUES ('petrochina', 2024, 500.5, 3000.0)")
        conn.commit()
        conn.close()

        cmd = [sys.executable, "-m", "mcp.servers.sqlite_server", "--db", db_path]
        transport = StdioTransport(command=cmd)
        await transport.start()
        client = MCPClient("sqlite_default", transport)
        await client.connect()

        try:
            # Tool discovery
            assert "list_tables" in client.tools
            assert "sql_query" in client.tools
            assert "describe_table" in client.tools
            assert "get_sample_rows" in client.tools

            # list_tables
            result = await client.call_tool("list_tables", {})
            tables = [t["name"] for t in result["content"]]
            assert "financials" in tables

            # describe_table
            result = await client.call_tool("describe_table", {"table": "financials"})
            cols = {c["name"]: c["type"] for c in result["content"]}
            assert cols["company"] == "TEXT"
            assert cols["net_profit"] == "REAL"
            assert cols["year"] == "INT"

            # sql_query with ORDER BY and LIMIT
            result = await client.call_tool("sql_query", {
                "sql": "SELECT company, net_profit FROM financials WHERE year=2024 ORDER BY net_profit DESC LIMIT 2"
            })
            assert len(result["content"]) == 2
            assert result["content"][0]["company"] == "maotai"
            assert result["content"][0]["net_profit"] == 747.3

            # sql_query reject INSERT
            with pytest.raises(Exception):
                await client.call_tool("sql_query", {
                    "sql": "INSERT INTO financials VALUES ('hack', 2024, 0, 0)"
                })

            # get_sample_rows
            result = await client.call_tool("get_sample_rows", {
                "table": "financials", "limit": 2
            })
            assert len(result["content"]) == 2
        finally:
            await client.close()
            os.unlink(db_path)

    asyncio.run(_run())
