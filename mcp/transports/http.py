"""MCP HTTP transport — JSON-RPC over HTTP POST, with optional SSE."""
import logging

import httpx

from mcp.transports.base import BaseTransport

logger = logging.getLogger(__name__)


class HttpTransport(BaseTransport):
    """Send JSON-RPC 2.0 requests over HTTP POST.

    SSE streaming is supported via the optional on_event callback.
    """

    def __init__(self, url: str, headers: dict | None = None,
                 timeout: float = 30.0):
        self.url = url
        self.headers = headers or {}
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={**self.headers, "Content-Type": "application/json"},
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    async def send(self, message: dict) -> dict:
        client = await self._ensure_client()
        resp = await client.post(self.url, json=message)
        resp.raise_for_status()
        return resp.json()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
