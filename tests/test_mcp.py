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
