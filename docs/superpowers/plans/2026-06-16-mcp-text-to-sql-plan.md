# MCP Client + Text-to-SQL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add MCP Client support and a Text-to-SQL retrieval tool to the Agentic Agent, enabling table queries through a built-in SQLite MCP Server.

**Architecture:** New `mcp/` top-level module with async transports (stdio + HTTP) and a client that connects to MCP servers, discovers tools, and executes calls. MCP tools integrate into the existing ToolRegistry via the reserved `_mcp` dict and `mcp__` prefix. A `text_to_sql` built-in retrieval tool converts natural language to SQL using schema introspection, then executes via the MCP SQLite client.

**Tech Stack:** Python 3.12+, asyncio, httpx, sqlite3, pytest

---

### Task 1: Transport Layer (`mcp/transports/`)

**Files:**
- Create: `mcp/__init__.py`
- Create: `mcp/transports/__init__.py`
- Create: `mcp/transports/base.py`
- Create: `mcp/transports/stdio.py`
- Create: `mcp/transports/http.py`
- Create: `tests/test_mcp.py` (transports portion)

- [ ] **Step 1: Create `mcp/__init__.py` and `mcp/transports/__init__.py`**

```python
# mcp/__init__.py
"""MCP Client and built-in servers for AgenticRAG."""
```

```python
# mcp/transports/__init__.py
from mcp.transports.base import BaseTransport
from mcp.transports.stdio import StdioTransport
from mcp.transports.http import HttpTransport

__all__ = ["BaseTransport", "StdioTransport", "HttpTransport"]
```

- [ ] **Step 2: Write failing tests for BaseTransport**

```python
# tests/test_mcp.py (add at top)
"""MCP module unit tests"""
import sys, os, json, asyncio, tempfile, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBaseTransport:
    def test_cannot_instantiate_abstract(self):
        from mcp.transports.base import BaseTransport
        with pytest.raises(TypeError):
            BaseTransport()

    def test_subclass_must_implement_send(self):
        from mcp.transports.base import BaseTransport
        class Incomplete(BaseTransport):
            async def close(self): pass
        with pytest.raises(TypeError):
            Incomplete()

    def test_valid_subclass(self):
        from mcp.transports.base import BaseTransport
        class Complete(BaseTransport):
            async def send(self, message): return {}
            async def close(self): pass
        t = Complete()
        assert isinstance(t, BaseTransport)
```

Run: `pytest tests/test_mcp.py::TestBaseTransport -v`
Expected: FAIL (import errors, files don't exist yet)

- [ ] **Step 3: Implement `mcp/transports/base.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify base transport**

Run: `pytest tests/test_mcp.py::TestBaseTransport -v`
Expected: 3 PASS

- [ ] **Step 5: Write failing tests for StdioTransport**

```python
# add to tests/test_mcp.py
class TestStdioTransport:
    @pytest.mark.asyncio
    async def test_echo_server_send(self):
        from mcp.transports.stdio import StdioTransport
        # python -c prints received JSON, echoes as JSON-RPC response
        echo_cmd = [
            sys.executable, "-c",
            "import sys,json; req=json.loads(sys.stdin.readline()); "
            "resp={'jsonrpc':'2.0','id':req['id'],'result':{'echo':req['params']['text']}}; "
            "print(json.dumps(resp), flush=True)"
        ]
        t = StdioTransport(command=echo_cmd)
        try:
            await t.start()
            resp = await t.send({
                "jsonrpc": "2.0", "id": 1, "method": "echo",
                "params": {"text": "hello"}
            })
            assert resp["result"]["echo"] == "hello"
        finally:
            await t.close()

    @pytest.mark.asyncio
    async def test_close_cleanup(self):
        from mcp.transports.stdio import StdioTransport
        t = StdioTransport(command=[sys.executable, "-c", "import time; time.sleep(10)"])
        await t.start()
        assert t.process is not None
        assert t.process.returncode is None  # still running
        await t.close()
        # After close, process should be terminated
        assert t.process.returncode is not None
```

- [ ] **Step 6: Implement `mcp/transports/stdio.py`**

```python
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
        self._lock = asyncio.Lock()  # serialize writes, pair requests with responses
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
```

Run: `pytest tests/test_mcp.py::TestStdioTransport -v`
Expected: 2 PASS

- [ ] **Step 7: Write failing tests for HttpTransport**

```python
# add to tests/test_mcp.py
class TestHttpTransport:
    @pytest.mark.asyncio
    async def test_http_send_request(self):
        from mcp.transports.http import HttpTransport
        t = HttpTransport(url="http://localhost:19999/mcp")  # nonexistent
        # Should raise connection error, not an import/init error
        with pytest.raises(Exception):
            await t.send({"jsonrpc": "2.0", "id": 1, "method": "ping"})

    def test_http_transport_creation(self):
        from mcp.transports.http import HttpTransport
        t = HttpTransport(url="http://example.com/mcp", headers={"Authorization": "Bearer x"})
        assert t.url == "http://example.com/mcp"
        assert t.headers["Authorization"] == "Bearer x"
```

- [ ] **Step 8: Implement `mcp/transports/http.py`**

```python
"""MCP HTTP transport — JSON-RPC over HTTP POST, with optional SSE."""
import json
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
```

Run: `pytest tests/test_mcp.py::TestHttpTransport -v`
Expected: 2 PASS

- [ ] **Step 9: Commit transport layer**

```bash
git add mcp/__init__.py mcp/transports/ tests/test_mcp.py
git commit -m "feat: add MCP transport layer (stdio + HTTP)"
```

---

### Task 2: MCPClient (`mcp/client.py`)

**Files:**
- Create: `mcp/client.py`
- Modify: `tests/test_mcp.py`

- [ ] **Step 1: Write failing tests for MCPClient**

```python
# add to tests/test_mcp.py
class TestMCPClient:
    @pytest.mark.asyncio
    async def test_client_connect_and_list_tools(self):
        from mcp.client import MCPClient
        from mcp.transports.stdio import StdioTransport

        # Mock MCP server that responds to initialize and tools/list
        server_script = (
            "import sys,json\n"
            "for req_str in [sys.stdin.readline(), sys.stdin.readline()]:\n"
            "    req=json.loads(req_str)\n"
            "    if req.get('method')=='initialize':\n"
            "        print(json.dumps({'jsonrpc':'2.0','id':req['id'],"
            "            'result':{'protocolVersion':'0.1.0','serverInfo':{'name':'test'},'capabilities':{}}}))\n"
            "    elif req.get('method')=='tools/list':\n"
            "        print(json.dumps({'jsonrpc':'2.0','id':req['id'],"
            "            'result':{'tools':[{'name':'ping','description':'Test ping',"
            "            'inputSchema':{'type':'object','properties':{}}}]}}))\n"
        )
        cmd = [sys.executable, "-c", server_script]
        transport = StdioTransport(command=cmd)
        await transport.start()
        client = MCPClient("test_server", transport)
        await client.connect()
        assert "ping" in client.tools
        assert client.tools["ping"]["description"] == "Test ping"
        await client.close()

    @pytest.mark.asyncio
    async def test_client_call_tool(self):
        from mcp.client import MCPClient
        from mcp.transports.stdio import StdioTransport

        server_script = (
            "import sys,json\n"
            "req=json.loads(sys.stdin.readline())\n"              # initialize
            "print(json.dumps({'jsonrpc':'2.0','id':req['id'],"
            "    'result':{'protocolVersion':'0.1.0','serverInfo':{'name':'t'},'capabilities':{}}}))\n"
            "req=json.loads(sys.stdin.readline())\n"              # tools/list
            "print(json.dumps({'jsonrpc':'2.0','id':req['id'],"
            "    'result':{'tools':[{'name':'add','description':'Add two numbers',"
            "    'inputSchema':{'type':'object','properties':{'a':{'type':'number'},'b':{'type':'number'}}}}]}}))\n"
            "req=json.loads(sys.stdin.readline())\n"              # tools/call
            "print(json.dumps({'jsonrpc':'2.0','id':req['id'],"
            "    'result':{'content':[{'type':'text','text':'7'}]}}))\n"
        )
        cmd = [sys.executable, "-c", server_script]
        transport = StdioTransport(command=cmd)
        await transport.start()
        client = MCPClient("test_server", transport)
        await client.connect()
        result = await client.call_tool("add", {"a": 3, "b": 4})
        assert "7" in str(result)
        await client.close()
```

Run: `pytest tests/test_mcp.py::TestMCPClient -v`
Expected: FAIL (MCPClient not defined)

- [ ] **Step 2: Implement `mcp/client.py`**

```python
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
        self.tools: dict[str, dict] = {}  # tool_name -> full schema dict
        self._server_info: dict = {}
        self._connected = False

    async def connect(self) -> None:
        """Handshake with the MCP server: initialize + tools/list."""
        # 1. Initialize
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

        # 2. Discover tools
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
        """Call a tool on the MCP server and return the result."""
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
        """Close the transport connection."""
        self._connected = False
        self.tools.clear()
        await self.transport.close()
```

- [ ] **Step 3: Run tests to verify MCPClient**

Run: `pytest tests/test_mcp.py::TestMCPClient -v`
Expected: 2 PASS

- [ ] **Step 4: Commit MCPClient**

```bash
git add mcp/client.py tests/test_mcp.py
git commit -m "feat: add MCPClient with tool discovery and calling"
```

---

### Task 3: Built-in SQLite MCP Server (`mcp/servers/sqlite_server.py`)

**Files:**
- Create: `mcp/servers/__init__.py`
- Create: `mcp/servers/sqlite_server.py`
- Modify: `tests/test_mcp.py`

- [ ] **Step 1: Create `mcp/servers/__init__.py`**

```python
# mcp/servers/__init__.py
"""Built-in MCP servers."""
```

- [ ] **Step 2: Write failing tests for SQLiteMCP Server**

```python
# add to tests/test_mcp.py
class TestSQLiteMCPServer:
    @pytest.mark.asyncio
    async def test_list_tables(self):
        from mcp.client import MCPClient
        from mcp.transports.stdio import StdioTransport
        import tempfile

        db_path = tempfile.mktemp(suffix=".db")
        # Create a test db with one table
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE test_t (id INT, name TEXT)")
        conn.execute("INSERT INTO test_t VALUES (1, 'alice')")
        conn.commit()
        conn.close()

        cmd = [sys.executable, "-m", "mcp.servers.sqlite_server", "--db", db_path]
        transport = StdioTransport(command=cmd)
        await transport.start()
        client = MCPClient("sqlite_test", transport)
        await client.connect()

        # list_tables
        result = await client.call_tool("list_tables", {})
        tables = [t["name"] for t in result.get("content", [])]
        assert "test_t" in tables

        await client.close()
        os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_describe_table(self):
        from mcp.client import MCPClient
        from mcp.transports.stdio import StdioTransport
        import tempfile, sqlite3

        db_path = tempfile.mktemp(suffix=".db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE t (id INT, name TEXT, amount REAL)")
        conn.commit()
        conn.close()

        cmd = [sys.executable, "-m", "mcp.servers.sqlite_server", "--db", db_path]
        transport = StdioTransport(command=cmd)
        await transport.start()
        client = MCPClient("sqlite_test", transport)
        await client.connect()

        result = await client.call_tool("describe_table", {"table": "t"})
        content = result.get("content", [])
        assert len(content) > 0
        text = str(content).lower()
        assert "id" in text and "name" in text and "amount" in text

        await client.close()
        os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_sql_query_select(self):
        from mcp.client import MCPClient
        from mcp.transports.stdio import StdioTransport
        import tempfile, sqlite3

        db_path = tempfile.mktemp(suffix=".db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE t (id INT, name TEXT)")
        conn.execute("INSERT INTO t VALUES (1, 'alice'), (2, 'bob')")
        conn.commit()
        conn.close()

        cmd = [sys.executable, "-m", "mcp.servers.sqlite_server", "--db", db_path]
        transport = StdioTransport(command=cmd)
        await transport.start()
        client = MCPClient("sqlite_test", transport)
        await client.connect()

        result = await client.call_tool("sql_query", {"sql": "SELECT * FROM t ORDER BY id"})
        content = result.get("content", [])
        assert len(content) == 2
        assert content[0]["id"] == 1 and content[0]["name"] == "alice"

        await client.close()
        os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_sql_query_rejects_write(self):
        from mcp.client import MCPClient
        from mcp.transports.stdio import StdioTransport
        import tempfile, sqlite3

        db_path = tempfile.mktemp(suffix=".db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE t (id INT)")
        conn.commit()
        conn.close()

        cmd = [sys.executable, "-m", "mcp.servers.sqlite_server", "--db", db_path]
        transport = StdioTransport(command=cmd)
        await transport.start()
        client = MCPClient("sqlite_test", transport)
        await client.connect()

        # Should reject INSERT
        with pytest.raises(RuntimeError, match="Only SELECT"):
            await client.call_tool("sql_query", {"sql": "INSERT INTO t VALUES (1)"})

        await client.close()
        os.unlink(db_path)
```

Run: `pytest tests/test_mcp.py::TestSQLiteMCPServer -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement `mcp/servers/sqlite_server.py`**

```python
"""SQLite MCP Server — stdio JSON-RPC 2.0

Start: python -m mcp.servers.sqlite_server --db <path>
Protocol: one JSON line per request/response on stdin/stdout.
"""
import argparse
import json
import sqlite3
import sys


def main():
    parser = argparse.ArgumentParser(description="SQLite MCP Server")
    parser.add_argument("--db", required=True, help="Path to SQLite database file")
    args = parser.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            response = handle_request(request, conn)
            print(json.dumps(response, ensure_ascii=False), flush=True)
        except Exception as e:
            err = {
                "jsonrpc": "2.0",
                "id": request.get("id") if "request" in dir() else None,
                "error": {"code": -32603, "message": str(e)},
            }
            print(json.dumps(err, ensure_ascii=False), flush=True)


def handle_request(request: dict, conn: sqlite3.Connection) -> dict:
    req_id = request.get("id")
    method = request.get("method")

    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {
                "protocolVersion": "0.1.0",
                "serverInfo": {"name": "sqlite", "version": "0.1.0"},
                "capabilities": {"tools": {}},
            },
        }
    elif method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}
    elif method == "tools/call":
        params = request.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        result = call_tool(tool_name, arguments, conn)
        return {"jsonrpc": "2.0", "id": req_id, "result": result}
    else:
        return {
            "jsonrpc": "2.0", "id": req_id,
            "error": {"code": -32601, "message": f"Unknown method: {method}"},
        }


def call_tool(name: str, args: dict, conn: sqlite3.Connection) -> dict:
    if name == "sql_query":
        return _sql_query(args["sql"], conn)
    elif name == "list_tables":
        return _list_tables(conn)
    elif name == "describe_table":
        return _describe_table(args["table"], conn)
    elif name == "get_sample_rows":
        return _get_sample_rows(args["table"], args.get("limit", 5), conn)
    else:
        raise ValueError(f"Unknown tool: {name}")


def _sql_query(sql: str, conn: sqlite3.Connection) -> dict:
    sql_upper = sql.strip().upper()
    if not any(sql_upper.startswith(kw) for kw in ("SELECT", "PRAGMA", "EXPLAIN", "WITH")):
        raise ValueError(f"Only SELECT/PRAGMA queries allowed. Got: {sql[:50]}")
    cur = conn.execute(sql)
    rows = [dict(row) for row in cur.fetchmany(100)]
    return {"content": rows, "row_count": len(rows)}


def _list_tables(conn: sqlite3.Connection) -> dict:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [{"name": row["name"]} for row in cur.fetchall()]
    return {"content": tables}


def _describe_table(table: str, conn: sqlite3.Connection) -> dict:
    cur = conn.execute(f"PRAGMA table_info({table})")
    columns = [
        {"name": row["name"], "type": row["type"], "nullable": not row["notnull"]}
        for row in cur.fetchall()
    ]
    if not columns:
        raise ValueError(f"Table not found: {table}")
    return {"content": columns}


def _get_sample_rows(table: str, limit: int, conn: sqlite3.Connection) -> dict:
    cur = conn.execute(f"SELECT * FROM {table} LIMIT ?", (limit,))
    rows = [dict(row) for row in cur.fetchall()]
    return {"content": rows, "row_count": len(rows)}


TOOLS = [
    {
        "name": "sql_query",
        "description": "Execute a SELECT or PRAGMA SQL query on the database. Returns up to 100 rows.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "SQL SELECT or PRAGMA statement"},
            },
            "required": ["sql"],
        },
    },
    {
        "name": "list_tables",
        "description": "List all tables in the SQLite database.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "describe_table",
        "description": "Get column names, types, and nullability for a table.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "table": {"type": "string", "description": "Table name"},
            },
            "required": ["table"],
        },
    },
    {
        "name": "get_sample_rows",
        "description": "Get sample rows from a table for inspection.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "table": {"type": "string", "description": "Table name"},
                "limit": {"type": "integer", "description": "Max rows (default 5)"},
            },
            "required": ["table"],
        },
    },
]


if __name__ == "__main__":
    main()
```

Run: `pytest tests/test_mcp.py::TestSQLiteMCPServer -v`
Expected: 4 PASS

- [ ] **Step 4: Commit SQLite MCP Server**

```bash
git add mcp/servers/__init__.py mcp/servers/sqlite_server.py tests/test_mcp.py
git commit -m "feat: add built-in SQLite MCP Server with 4 tools"
```

---

### Task 4: MCP Integration in ToolRegistry + text_to_sql Tool

**Files:**
- Modify: `agents/agentic/tools.py`
- Modify: `tests/test_agentic.py` (add new test class)

- [ ] **Step 1: Write failing tests for MCP integration and text_to_sql**

```python
# add to tests/test_agentic.py
class TestMCPToolIntegration:
    def test_mcp_tools_in_schema(self):
        from agents.agentic.tools import ToolRegistry
        r = ToolRegistry(model_size="large")
        # Manually register a mock MCP tool
        r._mcp["mock_server__ping"] = {
            "name": "ping",
            "description": "Test ping tool",
            "inputSchema": {"type": "object", "properties": {}},
        }
        schemas = r.get_all_schemas()
        mcp_names = [s["function"]["name"] for s in schemas if s["function"]["name"].startswith("mcp__")]
        assert "mcp__mock_server__ping" in mcp_names

    def test_text_to_sql_tool_registered(self):
        from agents.agentic.tools import ToolRegistry
        r = ToolRegistry(model_size="large")
        schemas = r.get_all_schemas()
        names = [s["function"]["name"] for s in schemas]
        assert "text_to_sql" in names

    def test_text_to_sql_is_retrieval_tool(self):
        from agents.agentic.tools import ToolRegistry
        r = ToolRegistry()
        assert r.is_retrieval_tool("text_to_sql")

    def test_text_to_sql_not_in_small_model_prompts(self):
        # text_to_sql should always be available regardless of model size
        from agents.agentic.tools import ToolRegistry
        r = ToolRegistry(model_size="small")
        names = [s["function"]["name"] for s in r.get_all_schemas()]
        assert "text_to_sql" in names

    def test_discover_mcp_configured(self):
        from agents.agentic.tools import ToolRegistry
        r = ToolRegistry()
        # discover_mcp should exist and be callable (even if it does nothing with empty config)
        import asyncio
        asyncio.run(r.discover_mcp(servers=[]))
        assert len(r._mcp_clients) == 0
```

Run: `pytest tests/test_agentic.py::TestMCPToolIntegration -v`
Expected: FAIL (text_to_sql not found in schemas)

- [ ] **Step 2: Add `text_to_sql` tool definition and MCP execution to `agents/agentic/tools.py`**

In `tools.py`, after line 97 (end of `_RETRIEVAL_TOOL_DEFS`), add:

```python
    ToolMeta(
        name="text_to_sql",
        category="retrieval",
        description="Convert natural language to SQL, execute via SQLite MCP, return structured results. Requires a target table name from prior retrieval.",
        when_to_use="检索到表格 chunk 后，需要查表、聚合计算、条件筛选、排序",
        when_not_to_use="纯文本检索即可回答、无表格 chunk 可用",
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Natural language question about the table data (e.g. '2024年净利润最高的3家公司')",
                },
                "table_name": {
                    "type": "string",
                    "description": "Target table name (from previously retrieved chunk)",
                },
                "chunk_context": {
                    "type": "string",
                    "description": "Relevant chunk text containing schema hints (optional)",
                },
            },
            "required": ["question", "table_name"],
        },
        priority=9,
    ),
```

In the `discover_mcp` method (line 276-278), replace the pass with:

```python
    async def discover_mcp(self, servers: list[dict] | None = None):
        """连接 MCP Server，发现并注册工具"""
        if servers is None:
            servers = []
        from mcp.client import MCPClient
        from mcp.transports.stdio import StdioTransport
        from mcp.transports.http import HttpTransport

        for cfg in servers:
            name = cfg["name"]
            transport_type = cfg.get("transport", "stdio")

            if transport_type == "stdio":
                transport = StdioTransport(
                    command=cfg["command"],
                    cwd=cfg.get("cwd"),
                )
                await transport.start()
            elif transport_type == "http":
                transport = HttpTransport(
                    url=cfg["url"],
                    headers=cfg.get("headers"),
                )
            else:
                raise ValueError(f"Unknown transport: {transport_type}")

            client = MCPClient(name, transport)
            await client.connect()
            self._mcp_clients[name] = client
            for tool_name, tool_schema in client.tools.items():
                mcp_name = f"mcp__{name}__{tool_name}"
                self._mcp[mcp_name] = tool_schema
```

In the `execute` method, add the MCP routing before the general retrieval tools (after line 299, `_exec_read_chunk`):

```python
        # MCP 工具 — 外部 MCP Server
        elif name.startswith("mcp__"):
            return self._exec_mcp(call)
```

Add the `_exec_mcp` and `_exec_text_to_sql` methods to `ToolRegistry`:

```python
    def _exec_mcp(self, call: ToolCall) -> ToolResult:
        """执行 MCP 工具调用 — mcp__<server>__<tool_name>"""
        # Parse mcp__server_name__tool_name
        parts = call.name.split("__", 2)
        if len(parts) != 3:
            return ToolResult(
                call_id=call.id, tool_name=call.name, success=False,
                content=f"Invalid MCP tool name: {call.name}. Expected: mcp__<server>__<tool>",
            )
        _, server_name, tool_name = parts
        client = self._mcp_clients.get(server_name)
        if client is None:
            return ToolResult(
                call_id=call.id, tool_name=call.name, success=False,
                content=f"MCP server {server_name!r} not connected.",
            )
        try:
            import asyncio
            result = asyncio.run(client.call_tool(tool_name, call.args))
            content = json.dumps(result.get("content", result), ensure_ascii=False)
            return ToolResult(
                call_id=call.id, tool_name=call.name, success=True,
                content=content, raw=result.get("content"),
            )
        except Exception as e:
            return ToolResult(
                call_id=call.id, tool_name=call.name, success=False,
                content=f"MCP error: {e}",
            )

    def _exec_text_to_sql(self, call: ToolCall) -> ToolResult:
        """Text-to-SQL: NL question → schema lookup → SQL generation → execution"""
        question = call.args["question"]
        table_name = call.args["table_name"]
        chunk_context = call.args.get("chunk_context", "")

        # 1. Find the SQLite MCP client
        sqlite_client = self._mcp_clients.get("sqlite_default")
        if sqlite_client is None:
            return ToolResult(
                call_id=call.id, tool_name=call.name, success=False,
                content="No SQLite MCP server configured. Add 'sqlite_default' to MCP_SERVERS.",
            )

        import asyncio

        # 2. Describe table to get real schema
        try:
            desc_result = asyncio.run(sqlite_client.call_tool("describe_table", {"table": table_name}))
            columns = desc_result.get("content", [])
        except Exception as e:
            return ToolResult(
                call_id=call.id, tool_name=call.name, success=False,
                content=f"Failed to describe table {table_name!r}: {e}",
            )

        schema_text = "\n".join(
            f"  {c['name']} {c['type']}{' NOT NULL' if not c.get('nullable', True) else ''}"
            for c in columns
        )

        # 3. Generate SQL via LLM
        sql_prompt = (
            "You are a SQLite SQL expert. Write a valid SQLite SELECT query for the following request.\n"
            "Output ONLY the SQL, no explanation, no markdown.\n\n"
            f"Table: {table_name}\n"
            f"Schema:\n{schema_text}\n"
        )
        if chunk_context:
            sql_prompt += f"\nContext from retrieved chunks:\n{chunk_context[:1000]}\n"
        sql_prompt += f"\nQuestion: {question}\n"
        sql_prompt += "\nSQL:"

        from llm.client import agent_chat
        sql = agent_chat(sql_prompt).strip()
        # Strip markdown code fences if present
        sql = sql.removeprefix("```sql").removeprefix("```").removesuffix("```").strip()
        # Strip trailing semicolon
        sql = sql.rstrip(";")

        # 4. Execute SQL
        try:
            exec_result = asyncio.run(sqlite_client.call_tool("sql_query", {"sql": sql}))
            rows = exec_result.get("content", [])
        except Exception as e:
            return ToolResult(
                call_id=call.id, tool_name=call.name, success=False,
                content=f"SQL execution failed: {e}\nGenerated SQL: {sql}",
            )

        # 5. Format result
        output = f"SQL: {sql}\nRows: {len(rows)}\n"
        if rows:
            output += "Results:\n" + json.dumps(rows, ensure_ascii=False, indent=2)

        return ToolResult(
            call_id=call.id, tool_name=call.name, success=True,
            content=output, raw=rows, is_empty=len(rows) == 0,
        )
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_agentic.py::TestMCPToolIntegration -v`
Expected: 5 PASS

- [ ] **Step 4: Run all existing tests to verify no regressions**

Run: `pytest tests/test_agentic.py -v`
Expected: All existing tests still PASS

- [ ] **Step 5: Commit ToolRegistry changes**

```bash
git add agents/agentic/tools.py tests/test_agentic.py
git commit -m "feat: add MCP tool routing and text_to_sql retrieval tool"
```

---

### Task 5: Agent MCP Lifecycle + Prompt Updates

**Files:**
- Modify: `agents/agentic/agent.py`
- Modify: `agents/agentic/prompts.py`
- Modify: `tests/test_agentic.py`

- [ ] **Step 1: Write failing test for Agent MCP integration**

```python
# add to tests/test_agentic.py
class TestAgentMCP:
    def test_agent_init_with_mcp_config(self):
        from agents.agentic.agent import Agent
        agent = Agent(model_size="large")
        # Agent should handle discover_mcp gracefully even if no servers configured
        assert hasattr(agent, 'tools')
```

Run: `pytest tests/test_agentic.py::TestAgentMCP -v`
Expected: PASS (existing test already covers this)

- [ ] **Step 2: Add MCP initialization to `agents/agentic/agent.py`**

In `Agent.__init__`, at the end of the method (after line 30 `self.memory = MemoryManager()`), add:

```python
        self.mcp_initialized = False
```

In `Agent.run`, before the system prompt assembly (before line 46), add:

```python
        # Connect MCP servers on first run
        if not self.mcp_initialized:
            from config import MCP_SERVERS
            if MCP_SERVERS:
                import asyncio
                asyncio.run(self.tools.discover_mcp(MCP_SERVERS))
            self.mcp_initialized = True
```

- [ ] **Step 3: Update `agents/agentic/prompts.py` to include text_to_sql and MCP tools**

In `get_tool_descriptions`, add `text_to_sql` to the tool table. For the `zh` branch, add after the `read_chunk` row:

```python
        return """
## 可用工具

你拥有以下工具，请根据场景选择最合适的：

| 工具 | 适用场景 | 不适用场景 |
|------|---------|-----------|
| semantic_search | 概念性的、语义模糊的查询 | 精确名称匹配、代码查询 |
| keyword_search | 精确的公司名、代码、日期 | 语义模糊查询 |
| graph_search | 实体关系、多跳关联 | 数值查询、文本片段 |
| hybrid_search | 高召回场景，多方法融合 | 简单单步查询 |
| read_chunk | 已知 chunk_id 需要完整文本 | 没有 ID 的检索 |
| text_to_sql | 检索到表格 chunk 后，需要查表/聚合/筛选/排序 | 无表格可用、纯文本问答 |
| dispatch_subagent | 可拆分为 2+ 独立子任务 | 简单单步、强依赖任务 |
| activate_skill | 查询匹配某技能领域时激活 | 简单查询无需专业指引 |
| remember | 发现关键证据、矛盾点 | 常规检索结果 |
| plan_steps | 3+ 步的复杂任务 | 简单 1-2 步查询 |
| finish | 完成回答 | — |

## MCP 工具

如果系统连接了外部 MCP Server，工具列表中会出现 `mcp__<server>__<tool>` 格式的工具。
这些工具由外部系统提供，能力取决于连接的服务器。常见的有：sql_query、list_tables 等。
你无需特殊处理，正常选择使用即可。
"""
```

For the `en` branch, replace the return with:

```python
        return """
## Available Tools

You have the following tools. Choose the most suitable one for each scenario:

| Tool | Use When | Don't Use When |
|------|----------|----------------|
| semantic_search | Conceptual, fuzzy semantic queries | Exact name/code lookups |
| keyword_search | Exact company names, codes, dates | Semantic queries |
| graph_search | Entity relationships, multi-hop links | Numeric queries, text snippets |
| hybrid_search | High recall, multi-method fusion | Simple single-step |
| read_chunk | Known chunk_id, need full text | Searches without IDs |
| text_to_sql | Table chunk retrieved, need query/aggregate/filter/sort | No table available, text-only QA |
| dispatch_subagent | 2+ independent subtasks | Simple or tightly-dependent tasks |
| activate_skill | Query matches a skill domain | Simple queries, no domain guidance needed |
| remember | Key evidence, contradictions found | Routine search results |
| plan_steps | 3+ step complex tasks | Simple 1-2 step queries |
| finish | Complete answer | — |

## MCP Tools

If the system is connected to external MCP Servers, the tool list will include tools in `mcp__<server>__<tool>` format.
These tools are provided by external systems. Common examples: sql_query, list_tables, etc.
Use them normally like any other tool.
"""
```

Also remove `mcp>=1.0.0` from Task 6 dependencies since our `mcp/` module would shadow the PyPI package — we implement JSON-RPC directly.

- [ ] **Step 4: Update test for prompt changes**

```python
# add to test_agentic.py TestPrompts class
    def test_tool_descriptions_has_text_to_sql(self):
        from agents.agentic.prompts import get_tool_descriptions
        t = get_tool_descriptions("zh")
        assert "text_to_sql" in t

    def test_tool_descriptions_has_mcp_section(self):
        from agents.agentic.prompts import get_tool_descriptions
        t = get_tool_descriptions("zh")
        assert "MCP" in t
```

- [ ] **Step 5: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit Agent + Prompt changes**

```bash
git add agents/agentic/agent.py agents/agentic/prompts.py tests/test_agentic.py
git commit -m "feat: wire MCP lifecycle into Agent and update prompts"
```

---

### Task 6: Configuration + Dependencies

**Files:**
- Modify: `config.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add MCP_SERVERS to `config.py`**

After the `AGENT_CONFIG` block at the end of `config.py`, add:

```python
# ── MCP Client 配置 ──
MCP_SERVERS = [
    {
        "name": "sqlite_default",
        "transport": "stdio",
        "command": ["python", "-m", "mcp.servers.sqlite_server"],
        "args": ["--db", os.path.join(DATA_DIR, "sqlite", "default.db")],
    },
]

# Allow override via env: comma-separated JSON file paths or inline JSON
_mcp_override = os.environ.get("MCP_SERVERS_CONFIG")
if _mcp_override:
    import json as _json
    if _mcp_override.endswith(".json"):
        with open(_mcp_override) as _f:
            MCP_SERVERS = _json.load(_f)
    else:
        MCP_SERVERS = _json.loads(_mcp_override)
```

- [ ] **Step 2: Add deps to `requirements.txt`**

Add after the existing deps:

```
# MCP Client (HTTP transport)
httpx>=0.27.0
```

- [ ] **Step 3: Verify imports**

Run: `python -c "from config import MCP_SERVERS; print('MCP_SERVERS:', MCP_SERVERS)"`
Expected: Prints MCP_SERVERS list

- [ ] **Step 4: Commit config + deps**

```bash
git add config.py requirements.txt
git commit -m "feat: add MCP_SERVERS config and dependencies"
```

---

### Task 7: End-to-End Integration Test

**Files:**
- Create: `tests/test_mcp_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_mcp_integration.py
"""End-to-end MCP + Text-to-SQL integration test."""
import sys, os, sqlite3, tempfile, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


@pytest.mark.asyncio
async def test_full_text_to_sql_flow():
    """End-to-end: MCPClient → SQLite Server → text_to_sql → result"""
    from mcp.client import MCPClient
    from mcp.transports.stdio import StdioTransport
    from agents.agentic.tools import ToolRegistry, ToolCall

    # 1. Create a test database with financial data
    db_path = tempfile.mktemp(suffix=".db")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE financials (company TEXT, year INT, net_profit REAL, revenue REAL)")
    conn.execute("INSERT INTO financials VALUES ('茅台', 2024, 747.3, 1500.0)")
    conn.execute("INSERT INTO financials VALUES ('工行', 2024, 698.2, 1200.0)")
    conn.execute("INSERT INTO financials VALUES ('中石油', 2024, 500.5, 3000.0)")
    conn.commit()
    conn.close()

    # 2. Start SQLite MCP Server and connect
    cmd = [sys.executable, "-m", "mcp.servers.sqlite_server", "--db", db_path]
    transport = StdioTransport(command=cmd)
    await transport.start()
    client = MCPClient("sqlite_default", transport)
    await client.connect()

    # 3. Verify tools
    assert "list_tables" in client.tools
    assert "sql_query" in client.tools
    assert "describe_table" in client.tools

    # 4. Test list_tables
    result = await client.call_tool("list_tables", {})
    tables = [t["name"] for t in result["content"]]
    assert "financials" in tables

    # 5. Test describe_table
    result = await client.call_tool("describe_table", {"table": "financials"})
    cols = {c["name"]: c["type"] for c in result["content"]}
    assert cols["company"] == "TEXT"
    assert cols["net_profit"] == "REAL"

    # 6. Test sql_query directly
    result = await client.call_tool("sql_query", {
        "sql": "SELECT company, net_profit FROM financials WHERE year=2024 ORDER BY net_profit DESC LIMIT 2"
    })
    assert len(result["content"]) == 2
    assert result["content"][0]["company"] == "茅台"

    # 7. Test ToolRegistry with MCP integration
    r = ToolRegistry(model_size="large")
    r._mcp_clients["sqlite_default"] = client
    r._mcp["mcp__sqlite_default__sql_query"] = client.tools["sql_query"]

    call = ToolCall(
        id="test_1",
        name="text_to_sql",
        args={
            "question": "2024年净利润最高的2家公司",
            "table_name": "financials",
        },
    )
    result = r._exec_text_to_sql(call)
    assert result.success
    assert "茅台" in result.content

    await client.close()
    os.unlink(db_path)
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/test_mcp_integration.py -v`
Expected: 1 PASS

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS (existing + new)

- [ ] **Step 4: Final commit**

```bash
git add tests/test_mcp_integration.py
git commit -m "test: add end-to-end MCP + Text-to-SQL integration test"
```
