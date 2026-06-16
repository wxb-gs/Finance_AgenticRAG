"""SQLite MCP Server -- stdio JSON-RPC 2.0

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
            req_id = request.get("id") if "request" in dir() else None
            err = {
                "jsonrpc": "2.0",
                "id": req_id,
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
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
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
