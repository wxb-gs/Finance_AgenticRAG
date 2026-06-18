# Planning + Python MCP Server 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为财报分析 Agent 新增结构化规划 (Plan)、Python 沙箱执行 (MCP Server)、子 Agent 类型重构三个子系统。

**Architecture:** Python MCP Server 遵循现有 JSON-RPC 2.0 stdio 模式，与 SQLite MCP Server 一致。Plan 采用 Claude Code 风格——作为追踪工具嵌入 AgentState，模型自主调度但不强制约束。子 Agent 按能力层次（retrieval/analysis/general）替代按业务领域分类，execute_python 作为通用工具而非独立 Agent 类型。

**Tech Stack:** Python stdlib (subprocess/json/argparse), asyncio, pytest, 复用现有 MCPClient/StdioTransport

**依赖顺序:**
```
Task 1 (Python MCP Server) ──┐
                              ├──> Task 3 (SubAgent refactoring)
                              │
Task 2 (Plan types) ──────────┼──> Task 4 (Plan tools) ──> Task 5 (Agent loop) ──> Task 7 (Integration)
                              │
Task 6 (Prompts) ─────────────┘
```

---

### Task 1: Python MCP Server

**Files:**
- Create: `mcp/servers/python_server.py`
- Create: `tests/test_python_mcp.py`

- [ ] **Step 1: Write integration test for Python MCP Server**

```python
# tests/test_python_mcp.py
"""Python MCP Server integration tests."""
import sys, os, asyncio, json

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_python_mcp_execute_basic():
    """python_default MCP server: execute_python with basic code."""
    from mcp.client import MCPClient
    from mcp.transports.stdio import StdioTransport

    async def _run():
        cmd = [sys.executable, "-m", "mcp.servers.python_server"]
        transport = StdioTransport(command=cmd)
        await transport.start()
        client = MCPClient("python_default", transport)
        await client.connect()

        try:
            assert "execute_python" in client.tools
            tool = client.tools["execute_python"]
            assert tool["name"] == "execute_python"
            assert "code" in tool["inputSchema"]["required"]

            result = await client.call_tool("execute_python", {
                "code": "x = 2 + 3\nprint(f'result={x}')",
            })
            content = result["content"]
            assert content["success"] is True
            assert "result=5" in content["stdout"]
            assert content["returncode"] == 0
        finally:
            await client.close()

    asyncio.run(_run())


def test_python_mcp_execute_with_context():
    """execute_python with context variables."""
    from mcp.client import MCPClient
    from mcp.transports.stdio import StdioTransport

    async def _run():
        cmd = [sys.executable, "-m", "mcp.servers.python_server"]
        transport = StdioTransport(command=cmd)
        await transport.start()
        client = MCPClient("python_default", transport)
        await client.connect()

        try:
            result = await client.call_tool("execute_python", {
                "code": "npm = net_profit / revenue\nprint(f'NPM={npm:.4f}')",
                "context": {"net_profit": 36.0, "revenue": 292.0},
            })
            content = result["content"]
            assert content["success"] is True
            assert "NPM=0.1233" in content["stdout"]
        finally:
            await client.close()

    asyncio.run(_run())


def test_python_mcp_timeout():
    """execute_python timeout kills infinite loops."""
    from mcp.client import MCPClient
    from mcp.transports.stdio import StdioTransport

    async def _run():
        cmd = [sys.executable, "-m", "mcp.servers.python_server"]
        transport = StdioTransport(command=cmd)
        await transport.start()
        client = MCPClient("python_default", transport)
        await client.connect()

        try:
            result = await client.call_tool("execute_python", {
                "code": "while True: pass",
                "timeout": 2,
            })
            content = result["content"]
            assert content["success"] is False
            assert "timed out" in content["stderr"].lower()
        finally:
            await client.close()

    asyncio.run(_run())


def test_python_mcp_restricted_import():
    """execute_python blocks non-whitelisted imports."""
    from mcp.client import MCPClient
    from mcp.transports.stdio import StdioTransport

    async def _run():
        cmd = [sys.executable, "-m", "mcp.servers.python_server"]
        transport = StdioTransport(command=cmd)
        await transport.start()
        client = MCPClient("python_default", transport)
        await client.connect()

        try:
            result = await client.call_tool("execute_python", {
                "code": "import os\nprint(os.getcwd())",
            })
            content = result["content"]
            assert content["success"] is False
            assert "os" in content["stderr"].lower() or "not allowed" in content["stderr"].lower()
        finally:
            await client.close()

    asyncio.run(_run())


def test_python_mcp_syntax_error():
    """execute_python returns stderr for syntax errors."""
    from mcp.client import MCPClient
    from mcp.transports.stdio import StdioTransport

    async def _run():
        cmd = [sys.executable, "-m", "mcp.servers.python_server"]
        transport = StdioTransport(command=cmd)
        await transport.start()
        client = MCPClient("python_default", transport)
        await client.connect()

        try:
            result = await client.call_tool("execute_python", {
                "code": "x = ",
            })
            content = result["content"]
            assert content["success"] is False
            assert content["stderr"] != ""
        finally:
            await client.close()

    asyncio.run(_run())
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_python_mcp.py -v
```
Expected: all FAIL with `ModuleNotFoundError: No module named 'mcp.servers.python_server'`

- [ ] **Step 3: Implement Python MCP Server**

```python
# mcp/servers/python_server.py
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
        f"{k} = {json.dumps(v, ensure_ascii=False)}"
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
    """).strip()

    full_code = import_guard + "\n" + context_lines + "\n" + code

    try:
        result = subprocess.run(
            [sys.executable, "-c", full_code],
            capture_output=True,
            text=True,
            timeout=timeout,
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_python_mcp.py -v
```
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add mcp/servers/python_server.py tests/test_python_mcp.py
git commit -m "feat: add Python MCP Server with execute_python sandbox tool"
```

---

### Task 2: Plan + SubAgentConfig types

**Files:**
- Modify: `agents/agentic/types.py`

- [ ] **Step 1: Run existing type tests to establish baseline**

```bash
pytest tests/test_agentic.py::TestAgentTypes -v
```
Expected: all PASS

- [ ] **Step 2: Add PlanStep and Plan dataclasses to types.py**

Add after the `SubAgentConfig` dataclass (line 137):

```python
@dataclass
class PlanStep:
    """计划中的单个步骤"""
    id: str
    description: str
    depends_on: list[str] = field(default_factory=list)
    agent_type: Literal["retrieval", "analysis", "general"] = "retrieval"
    status: Literal["pending", "in_progress", "completed", "failed"] = "pending"
    result_summary: str = ""


@dataclass
class Plan:
    """多跳查询的结构化执行计划"""
    query: str
    steps: list[PlanStep] = field(default_factory=list)
    version: int = 1
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def ready_steps(self) -> list[PlanStep]:
        """返回所有依赖已满足且状态为 pending 的步骤"""
        completed = {s.id for s in self.steps if s.status == "completed"}
        return [
            s for s in self.steps
            if s.status == "pending"
            and set(s.depends_on).issubset(completed)
        ]

    def mark_step(self, step_id: str, status: str,
                  result_summary: str = ""):
        """更新步骤状态"""
        for s in self.steps:
            if s.id == step_id:
                s.status = status
                if result_summary:
                    s.result_summary = result_summary
                self.updated_at = time.time()
                return

    def all_done(self) -> bool:
        return all(
            s.status in ("completed", "failed") for s in self.steps
        )

    def format_status(self) -> str:
        """生成 Plan 状态的文本摘要，注入 System Prompt"""
        if not self.steps:
            return ""
        lines = [f"[当前计划] (版本 {self.version})"]
        for s in self.steps:
            status_mark = {
                "pending": "   ", "in_progress": "⏳",
                "completed": "✓", "failed": "✗",
            }.get(s.status, "?")
            deps = f" ← 依赖: {s.depends_on}" if s.depends_on else ""
            summary = f" → {s.result_summary[:80]}" if s.result_summary else ""
            lines.append(
                f"  {s.id} [{status_mark} {s.status}] {s.description}{deps}{summary}"
            )
        ready = self.ready_steps()
        if ready:
            lines.append(f"可并行派发: {[s.id for s in ready]}")
        blocked = [
            s.id for s in self.steps
            if s.status == "pending" and s.id not in {r.id for r in ready}
        ]
        if blocked:
            lines.append(f"等待依赖: {blocked}")
        return "\n".join(lines)
```

- [ ] **Step 3: Add Plan to AgentState**

Add `plan` field to `AgentState` dataclass (after `finished` field):

```python
plan: Plan | None = None
```

- [ ] **Step 4: Run type tests to verify no breakage**

```bash
pytest tests/test_agentic.py::TestAgentTypes -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add agents/agentic/types.py
git commit -m "feat: add Plan, PlanStep types and Plan field to AgentState"
```

---

### Task 3: SubAgent 类型重构

**Files:**
- Modify: `agents/agentic/sub_agent.py`
- Modify: `tests/test_agentic.py` (TestSubAgent)

- [ ] **Step 1: Update tests for new sub-agent types**

In `tests/test_agentic.py`, replace the `TestSubAgent` class:

```python
class TestSubAgent:
    def test_configs_exist(self):
        from agents.agentic.sub_agent import SUBAGENT_TYPES
        assert "retrieval" in SUBAGENT_TYPES
        assert "analysis" in SUBAGENT_TYPES
        assert "general" in SUBAGENT_TYPES
        assert "computation" not in SUBAGENT_TYPES
        assert "comparison" not in SUBAGENT_TYPES

    def test_retrieval_config(self):
        from agents.agentic.sub_agent import SUBAGENT_TYPES
        r = SUBAGENT_TYPES["retrieval"]
        assert r.max_iterations == 5
        assert "semantic_search" in r.tools
        assert r.model_hint == "small"

    def test_analysis_has_execute_python(self):
        from agents.agentic.sub_agent import SUBAGENT_TYPES
        a = SUBAGENT_TYPES["analysis"]
        assert a.max_iterations == 8
        assert a.model_hint == "large"
        assert "mcp__python_default__execute_python" in a.tools
        assert "semantic_search" in a.tools
        assert "finish" in a.tools

    def test_general_has_all_tools(self):
        from agents.agentic.sub_agent import SUBAGENT_TYPES
        g = SUBAGENT_TYPES["general"]
        assert g.max_iterations == 10
        assert g.model_hint == "mid"
        assert "mcp__python_default__execute_python" in g.tools
        assert "semantic_search" in g.tools
        assert "graph_search" in g.tools
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_agentic.py::TestSubAgent -v
```
Expected: FAIL on `assert "computation" not in SUBAGENT_TYPES` and `assert "comparison" not in SUBAGENT_TYPES`

- [ ] **Step 3: Rewrite SUBAGENT_TYPES in sub_agent.py**

Replace the entire `SUBAGENT_TYPES` dict (lines 12-34):

```python
SUBAGENT_TYPES: dict[str, SubAgentConfig] = {
    "retrieval": SubAgentConfig(
        description="聚焦的信息检索：搜索、读取、返回结构化结果",
        tools=["semantic_search", "keyword_search", "graph_search",
               "read_chunk", "finish"],
        max_iterations=5,
        system_prompt_override=(
            "你是信息检索专家。快速定位相关信息，返回结构化结果。"
            "不做深度分析推理。"
        ),
        model_hint="small",
    ),
    "analysis": SubAgentConfig(
        description=(
            "深度财务分析：搜索证据、用 Python 精确计算、"
            "多源对比、标注矛盾"
        ),
        tools=["semantic_search", "keyword_search", "graph_search",
               "read_chunk",
               "mcp__python_default__execute_python", "finish"],
        max_iterations=8,
        system_prompt_override=(
            "你是财务分析专家。搜索相关数据，用 execute_python "
            "执行精确计算，对比多源信息并标注矛盾点。"
            "输出结构化表格。"
        ),
        model_hint="large",
    ),
    "general": SubAgentConfig(
        description=(
            "通用子代理：搜+算一体，"
            "处理需要多工具组合的复杂子任务"
        ),
        tools=["semantic_search", "keyword_search", "graph_search",
               "read_chunk",
               "mcp__python_default__execute_python", "finish"],
        max_iterations=10,
        system_prompt_override=(
            "你是财务分析通用代理。根据任务需要自由组合搜索和计算工具，"
            "独立完成子任务并返回完整结果。"
        ),
        model_hint="mid",
    ),
}
```

Update `SubAgentManager.get_config` docstring (no code change needed, but the Literal type in `dispatch` method signature should remain flexible since it uses `SUBAGENT_TYPES.get(agent_type, ...)` which handles unknown types gracefully).

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_agentic.py::TestSubAgent -v
```
Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add agents/agentic/sub_agent.py tests/test_agentic.py
git commit -m "refactor: replace computation/comparison with analysis/general sub-agent types"
```

---

### Task 4: Plan tools — plan_query + plan_update

**Files:**
- Modify: `agents/agentic/tools.py`
- Modify: `tests/test_agentic.py` (add test class)

- [ ] **Step 1: Add tests for new plan tools**

Add to `tests/test_agentic.py`:

```python
class TestPlanTools:
    def test_plan_query_registered(self):
        from agents.agentic.tools import ToolRegistry
        r = ToolRegistry(model_size="large")
        names = [s["function"]["name"] for s in r.get_all_schemas()]
        assert "plan_query" in names
        assert "plan_steps" not in names

    def test_plan_update_registered(self):
        from agents.agentic.tools import ToolRegistry
        r = ToolRegistry(model_size="large")
        names = [s["function"]["name"] for s in r.get_all_schemas()]
        assert "plan_update" in names

    def test_plan_query_is_meta(self):
        from agents.agentic.tools import ToolRegistry
        r = ToolRegistry()
        assert r.is_meta_tool("plan_query")
        assert r.is_meta_tool("plan_update")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_agentic.py::TestPlanTools -v
```
Expected: FAIL on `assert "plan_query" in names`

- [ ] **Step 3: Replace plan_steps tool definition with plan_query + plan_update**

In `tools.py`, replace the `plan_steps` ToolMeta definition (lines 189-213 in `_META_TOOL_DEFS`):

```python
    ToolMeta(
        name="plan_query",
        category="meta",
        description="Generate a structured execution plan for complex multi-hop queries.",
        when_to_use="复杂查询 3 跳以上，需要预先分解为有序步骤",
        when_not_to_use="简单 1-2 步查询",
        parameters={
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "步骤 ID，如 step_1"},
                            "description": {"type": "string", "description": "步骤描述"},
                            "depends_on": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "此步骤依赖的步骤 ID 列表",
                            },
                            "agent_type": {
                                "type": "string",
                                "enum": ["retrieval", "analysis", "general"],
                                "description": "推荐子 Agent 类型",
                            },
                        },
                        "required": ["id", "description"],
                    },
                },
            },
            "required": ["steps"],
        },
    ),
    ToolMeta(
        name="plan_update",
        category="meta",
        description="Update plan step status or append new steps during execution.",
        when_to_use="步骤完成/失败时标记，或发现新信息需追加步骤",
        when_not_to_use="dispatch_subagent 会自动标记关联步骤的完成状态",
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["complete", "fail", "append", "revise"],
                    "description": "complete=标记完成, fail=标记失败, append=追加新步骤, revise=修订计划",
                },
                "step_id": {
                    "type": "string",
                    "description": "操作的步骤 ID（complete/fail 时必填）",
                },
                "result_summary": {
                    "type": "string",
                    "description": "步骤结果摘要（complete 时使用）",
                },
                "new_steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "description": {"type": "string"},
                            "depends_on": {"type": "array", "items": {"type": "string"}},
                            "agent_type": {"type": "string", "enum": ["retrieval", "analysis", "general"]},
                        },
                        "required": ["id", "description"],
                    },
                    "description": "追加的步骤列表（append 时必填）",
                },
            },
            "required": ["action"],
        },
    ),
```

- [ ] **Step 4: Add plan_query and plan_update handling to ToolRegistry.execute()**

In `tools.py`, update the meta tool sentinel check (line 360):

```python
        elif name in ("dispatch_subagent", "activate_skill", "remember",
                      "plan_query", "plan_update"):
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_agentic.py::TestPlanTools -v
pytest tests/test_agentic.py::TestToolRegistry -v
```
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add agents/agentic/tools.py tests/test_agentic.py
git commit -m "feat: add plan_query and plan_update tools, replace plan_steps"
```

---

### Task 5: Agent loop — Plan 生成/追踪/System Prompt 注入

**Files:**
- Modify: `agents/agentic/agent.py`

- [ ] **Step 1: Run existing tests to establish baseline**

```bash
pytest tests/test_agentic.py::TestAgent -v
```
Expected: all existing tests PASS

- [ ] **Step 2: Add Plan handling in agent.py run() method**

In `agent.py`, after memory recall (line 52) and before the while loop (line 70), add Plan injection logic. Replace the `while` loop preamble:

At line 68 (before `# 4. ReAct 循环`), insert Plan status injection:

```python
        # 3.5 注入 Plan 状态到 System Prompt
        def _build_messages_with_plan(base_messages, state):
            """每轮重建消息，动态注入 Plan 状态"""
            if state.plan and state.plan.steps:
                plan_text = state.plan.format_status()
                msgs = list(base_messages)
                msgs.insert(1, {"role": "system", "content": plan_text})
                return msgs
            return base_messages
```

Replace the while loop with Plan-aware version. At line 70, the while loop stays; inside the loop, before `# LLM 调用` (line 79), add:

```python
            # 每次 LLM 调用前注入最新 Plan 状态
            current_messages = _build_messages_with_plan(messages, state)
            response = self._chat(current_messages, tool_schemas)
```

And replace the existing `response = self._chat(messages, tool_schemas)` line.

- [ ] **Step 3: Add plan_query tool handling in the ReAct loop**

In `agent.py`, after the `plan_steps` handling block (lines 169-181), replace with plan_query + plan_update handling:

```python
                    elif call.name == "plan_query":
                        from agents.agentic.types import Plan, PlanStep
                        steps_data = call.args.get("steps", [])
                        plan_steps = [
                            PlanStep(
                                id=s["id"],
                                description=s["description"],
                                depends_on=s.get("depends_on", []),
                                agent_type=s.get("agent_type", "retrieval"),
                            )
                            for s in steps_data
                        ]
                        state.plan = Plan(
                            query=query,
                            steps=plan_steps,
                        )
                        result = ToolResult(
                            call_id=call.id, tool_name=call.name,
                            success=True,
                            content=f"Plan created with {len(plan_steps)} steps:\n{state.plan.format_status()}",
                        )
                        state.add_tool_call(call, result)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": result.content,
                        })

                    elif call.name == "plan_update":
                        action = call.args.get("action", "")
                        step_id = call.args.get("step_id", "")
                        if state.plan:
                            if action == "complete":
                                state.plan.mark_step(
                                    step_id, "completed",
                                    call.args.get("result_summary", ""),
                                )
                            elif action == "fail":
                                state.plan.mark_step(step_id, "failed")
                            elif action == "append":
                                from agents.agentic.types import PlanStep
                                new_steps = call.args.get("new_steps", [])
                                for s in new_steps:
                                    state.plan.steps.append(PlanStep(
                                        id=s["id"],
                                        description=s["description"],
                                        depends_on=s.get("depends_on", []),
                                        agent_type=s.get("agent_type", "retrieval"),
                                    ))
                                state.plan.version += 1
                                state.plan.updated_at = time.time()
                            elif action == "revise":
                                state.plan.version += 1
                                state.plan.updated_at = time.time()
                        result = ToolResult(
                            call_id=call.id, tool_name=call.name,
                            success=True,
                            content=f"Plan updated: action={action}",
                        )
                        state.add_tool_call(call, result)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": result.content,
                        })
```

- [ ] **Step 4: Add step_id tracking to dispatch_subagent handling**

In the `dispatch_subagent` handling block (lines 116-132), after `state.subagent_count += 1`, add Plan step auto-update:

```python
                        # Auto-update Plan step on completion
                        step_id = call.args.get("step_id")
                        if step_id and state.plan:
                            state.plan.mark_step(
                                step_id, "completed",
                                sub_result.get("findings", "")[:200],
                            )
```

And before `_run_subagent_sync`, mark the step as in_progress:

```python
                    elif call.name == "dispatch_subagent" and self.enable_subagents:
                        step_id = call.args.get("step_id")
                        if step_id and state.plan:
                            state.plan.mark_step(step_id, "in_progress")

                        sub_result = self._run_subagent_sync(...)
```

- [ ] **Step 5: Add step_id to dispatch_subagent tool schema**

In `tools.py`, update the `dispatch_subagent` ToolMeta parameters to include step_id:

Inside the `dispatch_subagent` ToolMeta definition (around line 130-151), add to the `properties` dict:

```python
                "step_id": {
                    "type": "string",
                    "description": "Optional. Associate this dispatch with a plan step for automatic status tracking.",
                },
```

- [ ] **Step 6: Run tests to verify no breakage**

```bash
pytest tests/test_agentic.py::TestAgent -v
pytest tests/test_agentic.py::TestToolRegistry -v
```
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add agents/agentic/agent.py agents/agentic/tools.py
git commit -m "feat: integrate Plan generation, tracking, and auto-update into Agent loop"
```

---

### Task 6: Prompt 更新

**Files:**
- Modify: `agents/agentic/prompts.py`

- [ ] **Step 1: Run existing prompt tests to establish baseline**

```bash
pytest tests/test_agentic.py::TestPrompts -v
```
Expected: all PASS

- [ ] **Step 2: Update tool description table**

In `prompts.py`, update the `_ZH_BASE` tool selection rules section to reflect new tools. After the existing tool list, replace the behavior rules section:

Replace lines 82-88 (the tool selection rules in `_ZH_BASE`):

```python
## 工具选择规则

- 精确匹配（公司全称、代码、日期、金额）→ `keyword_search`
- 模糊语义查询（概念解释、趋势分析）→ `semantic_search`
- 实体关系查询（股东、子公司、关联方）→ `graph_search`
- 需要读取完整文本块 → `read_chunk`
- 需要跨多个检索方法高召回 → `hybrid_search`
- 检索到表格后需查表/聚合/筛选 → `text_to_sql`
- 需要精确数值计算 → `execute_python`（通过 Python 沙箱执行）
- 复杂多跳查询 → 先用 `plan_query` 生成执行计划
```

Update the `_ZH_LARGE_EXTRA` section to describe new sub-agent types. Replace the sub-agent decomposition paragraph (lines 123-127):

```python
## 子代理类型

你可以使用 `dispatch_subagent` 并行拆分独立子任务，支持三种类型：
- `retrieval` — 纯信息检索，快速返回结构化结果（小模型）
- `analysis` — 深度财务分析：搜索 + execute_python 精确计算 + 多源对比（大模型）
- `general` — 通用子代理：自由组合搜索和计算工具处理复杂子任务（中模型）

派发时可选 `step_id` 关联计划步骤，系统会自动追踪步骤状态。
```

- [ ] **Step 3: Update get_tool_descriptions() table**

In `prompts.py`, update the tool description table (lines 176-188 for zh). Replace:

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
| plan_query | 3+ 跳复杂查询，需预先分解 | 简单 1-2 步查询 |
| plan_update | 执行中标记步骤完成/失败，或追加步骤 | 步骤自动追踪时无需手动调用 |
| execute_python | 精确数值计算、统计分析 | 纯文本推理、不需要计算 |
| finish | 完成回答 | — |

## MCP 工具

如果系统连接了外部 MCP Server，工具列表中会出现 `mcp__<server>__<tool>` 格式的工具。
这些工具由外部系统提供，能力取决于连接的服务器。常见的有：execute_python、sql_query、list_tables 等。
你无需特殊处理，正常选择使用即可。
"""
```

- [ ] **Step 4: Run prompt tests**

```bash
pytest tests/test_agentic.py::TestPrompts -v
```
Expected: some tests may fail if they assert on old tool names — update assertions to match new tool names (add `plan_query`, `plan_update`, `execute_python` to expected lists, remove `plan_steps`).

- [ ] **Step 5: Commit**

```bash
git add agents/agentic/prompts.py tests/test_agentic.py
git commit -m "feat: update prompts for plan tools, execute_python, and new sub-agent types"
```

---

### Task 7: 配置更新 + 集成测试

**Files:**
- Modify: `config.py`
- Modify: `tests/test_agentic.py` (add integration-focused tests)

- [ ] **Step 1: Add PYTHON_MCP_SERVER to config.py**

In `config.py`, add after the sqlite_default MCP_SERVERS entry (line 63):

```python
    {
        "name": "python_default",
        "transport": "stdio",
        "command": ["python", "-m", "mcp.servers.python_server"],
    },
```

- [ ] **Step 2: Add Plan integration test**

Add to `tests/test_agentic.py`:

```python
class TestPlanIntegration:
    def test_plan_creation_and_status(self):
        from agents.agentic.types import Plan, PlanStep

        plan = Plan(
            query="测试查询",
            steps=[
                PlanStep(id="s1", description="步骤1"),
                PlanStep(id="s2", description="步骤2", depends_on=["s1"]),
                PlanStep(id="s3", description="步骤3"),
            ],
        )

        ready = plan.ready_steps()
        assert len(ready) == 2
        assert {s.id for s in ready} == {"s1", "s3"}

        plan.mark_step("s1", "completed", "完成步骤1")
        ready = plan.ready_steps()
        assert len(ready) == 2
        assert {s.id for s in ready} == {"s2", "s3"}

        plan.mark_step("s2", "completed")
        plan.mark_step("s3", "completed")
        assert plan.all_done()

    def test_plan_format_status(self):
        from agents.agentic.types import Plan, PlanStep

        plan = Plan(
            query="测试",
            steps=[
                PlanStep(id="s1", description="检索数据",
                         status="completed", result_summary="找到3条"),
                PlanStep(id="s2", description="计算ROE",
                         depends_on=["s1"]),
            ],
        )
        text = plan.format_status()
        assert "[当前计划]" in text
        assert "s1" in text
        assert "s2" in text
        assert "completed" in text
        assert "依赖" in text

    def test_agent_state_has_plan(self):
        from agents.agentic.types import AgentState, Plan, PlanStep

        state = AgentState(query="测试")
        state.plan = Plan(
            query="测试",
            steps=[PlanStep(id="s1", description="步骤1")],
        )
        assert state.plan is not None
        assert len(state.plan.steps) == 1

    def test_plan_version_increment(self):
        from agents.agentic.types import Plan, PlanStep

        plan = Plan(query="测试", steps=[PlanStep(id="s1", description="s1")])
        v1 = plan.version
        plan.version += 1
        assert plan.version == v1 + 1
```

- [ ] **Step 3: Run all tests**

```bash
pytest tests/test_agentic.py -v
pytest tests/test_python_mcp.py -v
```
Expected: all PASS

- [ ] **Step 4: Run full test suite**

```bash
pytest tests/ -v
```
Expected: all PASS (no regressions)

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_agentic.py
git commit -m "feat: add PYTHON_MCP_SERVER config and Plan integration tests"
```

---

## 改动文件汇总

| 文件 | 操作 | 行数估计 |
|------|------|---------|
| `mcp/servers/python_server.py` | 新建 | ~100 |
| `tests/test_python_mcp.py` | 新建 | ~130 |
| `agents/agentic/types.py` | 修改 (+Plan/PlanStep/Plan on AgentState) | +65 |
| `agents/agentic/sub_agent.py` | 修改 (3 种新类型) | ~25 |
| `agents/agentic/tools.py` | 修改 (plan_query/plan_update/step_id) | +60, -20 |
| `agents/agentic/agent.py` | 修改 (Plan 注入/追踪/自动更新) | +60 |
| `agents/agentic/prompts.py` | 修改 (工具表/子Agent描述) | ~30 |
| `config.py` | 修改 (+python_default MCP) | +4 |
| `tests/test_agentic.py` | 修改 (更新 SubAgent 测试 + Plan 测试) | +80 |
| **总计** | | **~530 行** |
