"""MCP stdio transport — subprocess with JSON-RPC over stdin/stdout."""
import asyncio
import json
import logging

from mcp.transports.base import BaseTransport

logger = logging.getLogger(__name__)


class StdioTransport(BaseTransport):
    """Launch a subprocess and communicate via JSON-RPC over stdin/stdout.

    Each message is a single line of JSON (JSON-RPC 2.0).
    """

    def __init__(self, command: list[str], cwd: str | None = None,
                 env: dict | None = None):
        self.command = command
        self.cwd = cwd
        self.env = env
        self.process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._request_id = 0

    async def start(self) -> None:
        self.process = await asyncio.create_subprocess_exec(
            *self.command,
            cwd=self.cwd,
            env=self.env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def send(self, message: dict) -> dict:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("Transport not started. Call start() first.")
        self._request_id += 1
        message["id"] = self._request_id
        payload = json.dumps(message, ensure_ascii=False) + "\n"

        async with self._lock:
            self.process.stdin.write(payload.encode())
            await self.process.stdin.drain()
            line = await asyncio.wait_for(
                self.process.stdout.readline(), timeout=30
            )
            if not line:
                raise ConnectionError("MCP subprocess closed stdout")
        return json.loads(line.decode())

    async def close(self) -> None:
        if self.process is None:
            return
        try:
            self.process.stdin.close()
        except Exception:
            pass
        try:
            await asyncio.wait_for(self.process.wait(), timeout=5)
        except asyncio.TimeoutError:
            self.process.kill()
            await self.process.wait()
        self.process = None
