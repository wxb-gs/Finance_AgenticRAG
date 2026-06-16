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


class TestStdioTransport:
    @pytest.mark.asyncio
    async def test_echo_server_send(self):
        from mcp.transports.stdio import StdioTransport
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
        proc = t.process  # capture before close sets it to None
        await t.close()
        assert proc.returncode is not None  # process was terminated


class TestHttpTransport:
    @pytest.mark.asyncio
    async def test_http_send_request(self):
        from mcp.transports.http import HttpTransport
        t = HttpTransport(url="http://localhost:19999/mcp")
        with pytest.raises(Exception):
            await t.send({"jsonrpc": "2.0", "id": 1, "method": "ping"})

    def test_http_transport_creation(self):
        from mcp.transports.http import HttpTransport
        t = HttpTransport(url="http://example.com/mcp", headers={"Authorization": "Bearer x"})
        assert t.url == "http://example.com/mcp"
        assert t.headers["Authorization"] == "Bearer x"


def _write_mock_server(code: str) -> str:
    """Write a mock MCP server script to a temp file. Returns the file path."""
    import tempfile
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    tmp.write(code)
    tmp.close()
    return tmp.name


class TestMCPClient:
    @pytest.mark.asyncio
    async def test_client_connect_and_list_tools(self):
        from mcp.client import MCPClient
        from mcp.transports.stdio import StdioTransport

        server_path = _write_mock_server("""import sys, json
for _ in range(2):
    req = json.loads(sys.stdin.readline())
    if req.get("method") == "initialize":
        resp = {"jsonrpc": "2.0", "id": req["id"], "result": {"protocolVersion": "0.1.0", "serverInfo": {"name": "test"}, "capabilities": {}}}
    elif req.get("method") == "tools/list":
        resp = {"jsonrpc": "2.0", "id": req["id"], "result": {"tools": [{"name": "ping", "description": "Test ping", "inputSchema": {"type": "object", "properties": {}}}]}}
    print(json.dumps(resp), flush=True)
""")
        try:
            cmd = [sys.executable, server_path]
            transport = StdioTransport(command=cmd)
            await transport.start()
            client = MCPClient("test_server", transport)
            await client.connect()
            assert "ping" in client.tools
            assert client.tools["ping"]["description"] == "Test ping"
            await client.close()
        finally:
            os.unlink(server_path)

    @pytest.mark.asyncio
    async def test_client_call_tool(self):
        from mcp.client import MCPClient
        from mcp.transports.stdio import StdioTransport

        server_path = _write_mock_server("""import sys, json
req = json.loads(sys.stdin.readline())
resp = {"jsonrpc": "2.0", "id": req["id"], "result": {"protocolVersion": "0.1.0", "serverInfo": {"name": "t"}, "capabilities": {}}}
print(json.dumps(resp), flush=True)

req = json.loads(sys.stdin.readline())
resp = {"jsonrpc": "2.0", "id": req["id"], "result": {"tools": [{"name": "add", "description": "Add two numbers", "inputSchema": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}}}]}}
print(json.dumps(resp), flush=True)

req = json.loads(sys.stdin.readline())
resp = {"jsonrpc": "2.0", "id": req["id"], "result": {"content": [{"type": "text", "text": "7"}]}}
print(json.dumps(resp), flush=True)
""")
        try:
            cmd = [sys.executable, server_path]
            transport = StdioTransport(command=cmd)
            await transport.start()
            client = MCPClient("test_server", transport)
            await client.connect()
            result = await client.call_tool("add", {"a": 3, "b": 4})
            assert "7" in str(result)
            await client.close()
        finally:
            os.unlink(server_path)


class TestSQLiteMCPServer:
    @pytest.mark.asyncio
    async def test_list_tables(self):
        from mcp.client import MCPClient
        from mcp.transports.stdio import StdioTransport
        import sqlite3

        db_path = tempfile.mktemp(suffix=".db")
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

        result = await client.call_tool("list_tables", {})
        tables = [t["name"] for t in result.get("content", [])]
        assert "test_t" in tables

        await client.close()
        os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_describe_table(self):
        from mcp.client import MCPClient
        from mcp.transports.stdio import StdioTransport
        import sqlite3

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
        import sqlite3

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
        import sqlite3

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

        with pytest.raises(RuntimeError, match="Only SELECT"):
            await client.call_tool("sql_query", {"sql": "INSERT INTO t VALUES (1)"})

        await client.close()
        os.unlink(db_path)
