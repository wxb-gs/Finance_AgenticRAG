"""Python MCP Server — stdio JSON-RPC 2.0 sandbox execution.

Start: python -m mcp.servers.python_server
Protocol: one JSON line per request/response on stdin/stdout.
"""
import json
import os
import subprocess
import sys
import textwrap

WHITELIST = frozenset({
    "pandas", "numpy", "scipy", "math", "statistics",
    "json", "datetime", "collections", "itertools", "functools",
})

TOOLS = [
    {
        "name": "execute_python",
        "description": (
            "Execute Python code in a sandbox and return stdout/stderr. "
            "Use for precise financial calculations, statistical analysis, data processing. "
            "Available: pandas, numpy, scipy.stats, math, statistics, json. "
            "Use print() for output or assign to __result__ dict for structured return data."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute.",
                },
                "context": {
                    "type": "object",
                    "description": (
                        "Optional context dict. Keys become variables available in code. "
                        'Example: {"revenue": 292.0, "cost": 218.0}'
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 30, max 60).",
                },
            },
            "required": ["code"],
        },
    },
]


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            response = _handle_request(request)
            print(json.dumps(response, ensure_ascii=False), flush=True)
        except Exception as exc:
            req_id = request.get("id") if "request" in dir() else None
            err = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32603, "message": str(exc)},
            }
            print(json.dumps(err, ensure_ascii=False), flush=True)


def _handle_request(request):
    req_id = request.get("id")
    method = request.get("method")

    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {
                "protocolVersion": "0.1.0",
                "serverInfo": {"name": "python", "version": "0.1.0"},
                "capabilities": {"tools": {}},
            },
        }
    elif method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}
    elif method == "tools/call":
        params = request.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        result = _call_tool(tool_name, arguments)
        return {"jsonrpc": "2.0", "id": req_id, "result": result}
    else:
        return {
            "jsonrpc": "2.0", "id": req_id,
            "error": {"code": -32601, "message": f"Unknown method: {method}"},
        }


def _call_tool(name, args):
    if name == "execute_python":
        return _execute_python(
            code=args.get("code", ""),
            context=args.get("context", {}),
            timeout=args.get("timeout", 30),
        )
    else:
        raise ValueError(f"Unknown tool: {name}")


def _execute_python(code, context, timeout):
    timeout = min(int(timeout or 30), 60)

    context_lines = "\n".join(
        f"{k} = {repr(v)}"
        for k, v in (context or {}).items()
    )

    import_guard = textwrap.dedent(f"""
    import builtins as __builtins
    _WHITELIST = {json.dumps(sorted(WHITELIST))}
    _orig_import = __builtins.__import__
    def _restricted_import(name, *args, **kwargs):
        top = name.split(".")[0]
        if top not in _WHITELIST:
            raise ImportError(
                f"Module '{{top}}' is not allowed. "
                f"Whitelist: {{sorted(_WHITELIST)}}"
            )
        return _orig_import(name, *args, **kwargs)
    __builtins.__import__ = _restricted_import
    _orig_open = __builtins.open
    def _restricted_open(file, mode='r', *args, **kwargs):
        if mode not in ('r', 'rb', 'rt'):
            raise PermissionError("File writes are not allowed in sandbox.")
        return _orig_open(file, mode, *args, **kwargs)
    __builtins.open = _restricted_open
    """).strip()

    full_code = import_guard + "\n" + context_lines + "\n" + code

    try:
        result = subprocess.run(
            [sys.executable, "-c", full_code],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={
                "PATH": os.environ.get("PATH", ""),
                "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
            },
        )
        return {
            "content": {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "success": result.returncode == 0,
            },
        }
    except subprocess.TimeoutExpired:
        return {
            "content": {
                "stdout": "",
                "stderr": f"Execution timed out after {timeout}s",
                "returncode": -1,
                "success": False,
            },
        }


if __name__ == "__main__":
    main()
