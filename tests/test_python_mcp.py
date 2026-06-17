"""Python MCP Server integration tests."""
import sys, os, asyncio, json

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_python_mcp_execute_basic():
    """python_default MCP server: execute_python with basic code."""
    from mcp.client import MCPClient
    from mcp.transports.stdio import StdioTransport

    async def _run():
        cmd = [sys.executable, "-m", "mcp.servers.python_server"]
        transport = StdioTransport(command=cmd)
        await transport.start()
        client = MCPClient("python_default", transport)
        await client.connect()

        try:
            assert "execute_python" in client.tools
            tool = client.tools["execute_python"]
            assert tool["name"] == "execute_python"
            assert "code" in tool["inputSchema"]["required"]

            result = await client.call_tool("execute_python", {
                "code": "x = 2 + 3\nprint(f'result={x}')",
            })
            content = result["content"]
            assert content["success"] is True
            assert "result=5" in content["stdout"]
            assert content["returncode"] == 0
        finally:
            await client.close()

    asyncio.run(_run())


def test_python_mcp_execute_with_context():
    """execute_python with context variables."""
    from mcp.client import MCPClient
    from mcp.transports.stdio import StdioTransport

    async def _run():
        cmd = [sys.executable, "-m", "mcp.servers.python_server"]
        transport = StdioTransport(command=cmd)
        await transport.start()
        client = MCPClient("python_default", transport)
        await client.connect()

        try:
            result = await client.call_tool("execute_python", {
                "code": "npm = net_profit / revenue\nprint(f'NPM={npm:.4f}')",
                "context": {"net_profit": 36.0, "revenue": 292.0},
            })
            content = result["content"]
            assert content["success"] is True
            assert "NPM=0.1233" in content["stdout"]
        finally:
            await client.close()

    asyncio.run(_run())


def test_python_mcp_timeout():
    """execute_python timeout kills infinite loops."""
    from mcp.client import MCPClient
    from mcp.transports.stdio import StdioTransport

    async def _run():
        cmd = [sys.executable, "-m", "mcp.servers.python_server"]
        transport = StdioTransport(command=cmd)
        await transport.start()
        client = MCPClient("python_default", transport)
        await client.connect()

        try:
            result = await client.call_tool("execute_python", {
                "code": "while True: pass",
                "timeout": 2,
            })
            content = result["content"]
            assert content["success"] is False
            assert "timed out" in content["stderr"].lower()
        finally:
            await client.close()

    asyncio.run(_run())


def test_python_mcp_restricted_import():
    """execute_python blocks non-whitelisted imports."""
    from mcp.client import MCPClient
    from mcp.transports.stdio import StdioTransport

    async def _run():
        cmd = [sys.executable, "-m", "mcp.servers.python_server"]
        transport = StdioTransport(command=cmd)
        await transport.start()
        client = MCPClient("python_default", transport)
        await client.connect()

        try:
            result = await client.call_tool("execute_python", {
                "code": "import os\nprint(os.getcwd())",
            })
            content = result["content"]
            assert content["success"] is False
            assert "os" in content["stderr"].lower() or "not allowed" in content["stderr"].lower()
        finally:
            await client.close()

    asyncio.run(_run())


def test_python_mcp_syntax_error():
    """execute_python returns stderr for syntax errors."""
    from mcp.client import MCPClient
    from mcp.transports.stdio import StdioTransport

    async def _run():
        cmd = [sys.executable, "-m", "mcp.servers.python_server"]
        transport = StdioTransport(command=cmd)
        await transport.start()
        client = MCPClient("python_default", transport)
        await client.connect()

        try:
            result = await client.call_tool("execute_python", {
                "code": "x = ",
            })
            content = result["content"]
            assert content["success"] is False
            assert content["stderr"] != ""
        finally:
            await client.close()

    asyncio.run(_run())
