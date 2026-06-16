"""Base transport abstraction for MCP JSON-RPC communication."""
from abc import ABC, abstractmethod


class BaseTransport(ABC):
    """Abstract transport for MCP JSON-RPC 2.0 messages."""

    @abstractmethod
    async def send(self, message: dict) -> dict:
        """Send a JSON-RPC request and return the response."""

    @abstractmethod
    async def close(self) -> None:
        """Close the transport and release resources."""
