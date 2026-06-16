"""MCP Client — connect to an MCP Server, discover and call tools."""
import json
import logging

from mcp.transports.base import BaseTransport

logger = logging.getLogger(__name__)


class MCPClient:
    """Connects to a single MCP Server via a transport.

    Each MCPClient manages one server connection, discovers its tools,
    and routes tool calls.
    """

    def __init__(self, name: str, transport: BaseTransport):
        self.name = name
        self.transport = transport
        self.tools: dict[str, dict] = {}
        self._server_info: dict = {}
        self._connected = False

    async def connect(self) -> None:
        init_req = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "0.1.0",
                "clientInfo": {"name": "agenticrag", "version": "0.1.0"},
                "capabilities": {},
            },
        }
        init_resp = await self.transport.send(init_req)
        self._server_info = init_resp.get("result", {})

        list_req = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
        }
        list_resp = await self.transport.send(list_req)
        tools = list_resp.get("result", {}).get("tools", [])
        for tool in tools:
            self.tools[tool["name"]] = tool

        self._connected = True
        logger.info("MCP server %r connected: %d tools discovered",
                     self.name, len(self.tools))

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        if not self._connected:
            raise RuntimeError(f"MCP server {self.name!r} not connected.")
        req = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }
        resp = await self.transport.send(req)
        if "error" in resp:
            raise RuntimeError(f"MCP tool error: {resp['error']}")
        return resp.get("result", {})

    async def close(self) -> None:
        self._connected = False
        self.tools.clear()
        await self.transport.close()
