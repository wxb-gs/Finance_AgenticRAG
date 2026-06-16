# MCP Client + Text-to-SQL 设计文档

## 概述

为 AgenticRAG 引入 MCP Client 支持，让 Agent 能连接外部 MCP Server 获取工具能力。同时内置 SQLite MCP Server 和 Text-to-SQL 工具，支持"检索表格 chunk → 自然语言查表 → SQL 执行"的完整链路。

## 新增文件

```
mcp/
├── __init__.py
├── client.py                 ← MCPClient：连接 MCP Server，发现+调用工具
├── transports/
│   ├── __init__.py
│   ├── base.py               ← BaseTransport 抽象
│   ├── stdio.py              ← StdioTransport（子进程 stdin/stdout JSON-RPC）
│   └── http.py               ← HttpTransport（HTTP POST JSON-RPC + SSE）
└── servers/
    ├── __init__.py
    └── sqlite_server.py      ← 内置 SQLite MCP Server（stdio 模式）
```

## 修改文件

| 文件 | 改动 |
|------|------|
| `agents/agentic/tools.py` | 实现 `discover_mcp()`，增加 MCP 工具执行路由，新增 `text_to_sql` 检索工具 |
| `agents/agentic/agent.py` | 启动时调用 `discover_mcp()`，`run()` 改为 async |
| `agents/agentic/prompts.py` | system prompt 增加 MCP 工具和 text_to_sql 的使用说明 |
| `config.py` | 增加 `MCP_SERVERS` 配置列表 |
| `requirements.txt` | 增加 `mcp>=1.0.0`、`httpx>=0.27.0` |

## 架构

### MCPClient（`mcp/client.py`）

每个 MCP Server 一个 MCPClient 实例，负责连接、工具发现、工具调用。

```
MCPClient(name, transport)
  ├─ connect()       → initialize + tools/list
  ├─ call_tool()     → tools/call
  ├─ close()         → 断开连接
  └─ tools: dict     → {tool_name: schema}
```

### 传输层（`mcp/transports/`）

- **BaseTransport**：`send(message) → dict`、`close()`
- **StdioTransport**：`asyncio.create_subprocess_exec`，stdin 写 JSON-RPC，stdout 读 JSON-RPC（一行一个 JSON）
- **HttpTransport**：`httpx.AsyncClient` POST JSON-RPC，SSE 用于服务端推送

### ToolRegistry 集成

- `discover_mcp(servers)`：按配置创建 MCPClient → connect → 发现工具 → 注册到 `_mcp` dict
- `execute()`：新增 `mcp__` 前缀路由，解析 `mcp__<server>__<tool>` 分发到对应 MCPClient
- MCP 工具 schema 在 `get_all_schemas()` 中以 `mcp__` 前缀暴露给 LLM

### 配置（`config.py`）

```python
MCP_SERVERS = [
    {
        "name": "sqlite_default",
        "transport": "stdio",
        "command": ["python", "-m", "mcp.servers.sqlite_server"],
        "args": ["--db", "data/sqlite/default.db"],
    },
]
```

## 内置 SQLite MCP Server（`mcp/servers/sqlite_server.py`）

### 协议

标准 MCP JSON-RPC 2.0 over stdio（一行一个 JSON）。

### 工具

| 工具 | 参数 | 功能 |
|------|------|------|
| `sql_query` | `sql: str` | 执行 SELECT/PRAGMA 查询 |
| `list_tables` | 无 | 列出所有表名 |
| `describe_table` | `table: str` | 返回列名、类型、是否可空 |
| `get_sample_rows` | `table: str, limit: int` | 前 N 行样本 |

### 安全约束

- `sql_query` 只允许 `SELECT` / `PRAGMA`，拒绝写入/修改语句
- 返回行数上限 100 行
- 连接以只读模式打开

## Text-to-SQL 工具

### 注册

作为第 6 个内置检索工具 `text_to_sql` 加入 `_RETRIEVAL_TOOL_DEFS`，优先级 9。

### 参数

- `question`（必填）：自然语言查表问题
- `table_name`（必填）：目标表名
- `chunk_context`（可选）：检索到的 chunk 文本（含 schema 提示）

### 执行流程

```
1. 调用 MCP describe_table(table) → 获取真实 schema（列名+类型）
2. 组装 prompt：问题 + schema + chunk_context → LLM 生成 SQL
3. 调用 MCP sql_query(sql) → 执行 SQL
4. 格式化结果（表格文本 + 执行的 SQL）
```

### Schema 感知

- 执行前必须调 `describe_table` 获取真实 schema，不信任 chunk 文本中的假设信息
- 列名和类型显式写入 LLM prompt，降低生成错误率

## 数据流

```
用户查询
  → Agent (ReAct)
    → 检索工具 (semantic/keyword/graph/hybrid) → 发现表格 chunk
    → text_to_sql(question, table_name, chunk_context)
      → MCPClient → SQLite MCP Server
        → describe_table → schema
        → sql_query → 结果
    → MCP 外部工具 (mcp__<server>__<tool>)
      → MCPClient → 外部 MCP Server
    → finish
```

## 兼容性

- PEV pipeline 不变，MCP 功能仅在 Agentic 模式下可用
- 如果没有配置 MCP_SERVERS，Agent 行为完全不变（向后兼容）
- 现有测试不受影响

## 依赖

- `mcp>=1.0.0`：MCP Python SDK（提供 JSON-RPC 工具类型定义）
- `httpx>=0.27.0`：HTTP 传输层异步客户端
