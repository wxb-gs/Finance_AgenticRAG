# Planning + Python MCP Server 设计

日期: 2026-06-17 | 状态: approved

## 概述

为财报分析 Agent 新增三个能力：结构化规划（Plan）、Python 脚本执行（MCP Server）、依赖感知的子 Agent 调度。设计全程参照 Claude Code 模式——规划是追踪工具而非强制约束，子 Agent 按能力层次而非业务领域分类。

## 1. Python MCP Server

### 1.1 架构

独立 stdio 进程，与现有 SQLite MCP Server 完全一致的 MCP 模式。Agent 通过 `ToolRegistry.discover_mcp()` 自动发现并注册工具。

```
Agent → ToolRegistry._exec_mcp() → MCPClient → stdio transport → Python MCP Server (子进程)
```

### 1.2 工具定义

**`execute_python`** — 沙箱执行 Python 代码并返回结果。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `code` | string | 是 | Python 代码。最后一行如果是表达式会自动打印。用 `print()` 输出，或赋值给 `__result__` 字典返回结构化数据 |
| `context` | object | 否 | 上下文字典，代码中可直接作为变量访问。如 `{"revenue": 292.0, "cost": 218.0}` |
| `timeout` | integer | 否 | 超时秒数，默认 30，最大 60 |

返回结构：
```json
{
  "success": true,
  "stdout": "ROE=0.1820\n",
  "stderr": "",
  "result": {"roe": 0.182, "npm": 0.1233},
  "execution_time_ms": 234
}
```

### 1.3 安全沙箱

- **subprocess 隔离**：`subprocess.run([sys.executable, "-c", code])`，独立进程
- **超时强制 kill**：默认 30s，最大 60s
- **无网络**：子进程环境无 HTTP 访问能力
- **无文件写**：不可写文件系统
- **白名单 import**：只允许 pandas, numpy, scipy.stats, math, statistics, json, datetime, collections, itertools, functools。os/sys/subprocess/socket 等被拦截
- **内存限制**：500MB 上限

### 1.4 配置

```python
# config.py
PYTHON_MCP_SERVER = {
    "name": "python_default",
    "transport": "stdio",
    "command": "python",
    "args": ["-m", "mcp_servers.python_server"],
    "cwd": os.path.dirname(os.path.abspath(__file__)),
}

# MCP_SERVERS 环境变量自动合并 PYTHON_MCP_SERVER
```

### 1.5 文件清单

| 文件 | 说明 |
|------|------|
| `mcp_servers/__init__.py` | 新建包声明 |
| `mcp_servers/python_server.py` | MCP Server 实现，~120 行 |
| `config.py` | 新增 PYTHON_MCP_SERVER 配置 |

---

## 2. Plan 规划器

### 2.1 设计原则

Claude Code 风格——Plan 是追踪工具 + 对齐手段，不是硬约束。

- Plan 存储在 AgentState，写入 trace 可评测
- 模型每轮看到 Plan 状态，自主判断下一步
- 执行中可以追加/修改步骤（Plan 版本递增）
- 不是 PEV 式的 DAG 强制编排

### 2.2 数据结构

```python
@dataclass
class PlanStep:
    id: str                                    # "step_1"
    description: str                           # "获取宁德时代利润表"
    depends_on: list[str] = field(default_factory=list)
    agent_type: Literal["retrieval", "analysis", "general"] = "retrieval"
    status: Literal["pending", "in_progress", "completed", "failed"] = "pending"
    result_summary: str = ""

@dataclass
class Plan:
    query: str
    steps: list[PlanStep]
    version: int = 1
    created_at: float = 0.0
    updated_at: float = 0.0

    def ready_steps(self) -> list[PlanStep]:
        """返回所有依赖已满足且未开始的步骤"""
        ...

    def mark_step(self, step_id: str, status: str, result_summary: str = ""):
        """更新步骤状态"""
        ...
```

### 2.3 工具定义

**`plan_query`**（替代现有 `plan_steps`）：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `steps` | array | 是 | 步骤列表：[{id, description, depends_on, agent_type}] |

触发条件：3 跳以上的复杂查询，需要预先分解。

**`plan_update`**（新增）：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `action` | string | 是 | complete / fail / append / revise |
| `step_id` | string | 视 action | 操作的步骤 ID |
| `result_summary` | string | 否 | complete 时的结果摘要 |
| `new_steps` | array | append 时 | 追加的步骤列表 |

### 2.4 System Prompt 注入

每轮 LLM 调用前动态注入 Plan 状态：

```
[当前计划] (版本 1)
  1. [completed] 获取利润表 → 营收 292亿, 净利润 36亿
  2. [completed] 获取资产负债表 → 总资产 320亿, 净资产 198亿
  3. [pending]   DuPont ROE 分解 ← 依赖: [1, 2]

提示: 可并行派发: [3]
```

### 2.5 Agent 循环集成

`agent.py:run()` 中，`dispatch_subagent` 处理逻辑扩展：

- 新增可选 `step_id` 参数——关联 Plan step
- 派发时自动标记 step 为 `in_progress`
- 完成时自动标记 step 为 `completed`，附带 result_summary
- 失败时自动标记 step 为 `failed`
- 模型无需手动调用 plan_update 来标记步骤完成（追加/修订仍需显式调用）

---

## 3. SubAgent 类型重新设计

### 3.1 对照 Claude Code 修正

Claude Code 按**读写边界**分类子 Agent（Explore=只读, general-purpose=全工具），而非按业务领域。当前 computation 类型"只能算不能搜"不符合这个逻辑。

修正：移除独立 computation 类型，execute_python 作为工具提供给有搜索能力的子 Agent。

### 3.2 最终 3 种类型

| 类型 | 工具 | 迭代上限 | 模型 | 对标 CC | 适用场景 |
|------|------|---------|------|---------|---------|
| `retrieval` | semantic_search, keyword_search, graph_search, read_chunk, finish | 5 | small | Explore | 纯搜索：获取数据、定位信息 |
| `analysis` | semantic_search, keyword_search, graph_search, read_chunk, execute_python, finish | 8 | large | Plan | 深度分析：搜+算+对比+判断 |
| `general` | 全部检索 + execute_python + finish | 10 | mid | general-purpose | 开放式子任务：不确定需要什么工具时兜底 |

```python
SUBAGENT_TYPES = {
    "retrieval": SubAgentConfig(
        description="聚焦的信息检索：搜索、读取、返回结构化结果",
        tools=["semantic_search", "keyword_search", "graph_search", "read_chunk", "finish"],
        max_iterations=5,
        system_prompt_override="你是信息检索专家。快速定位相关信息，返回结构化结果。不做深度分析推理。",
        model_hint="small",
    ),
    "analysis": SubAgentConfig(
        description="深度财务分析：搜索证据、用 Python 精确计算、多源对比、标注矛盾",
        tools=["semantic_search", "keyword_search", "graph_search", "read_chunk",
               "mcp__python_default__execute_python", "finish"],
        max_iterations=8,
        system_prompt_override="你是财务分析专家。搜索相关数据，用 execute_python 执行精确计算，对比多源信息并标注矛盾点。输出结构化表格。",
        model_hint="large",
    ),
    "general": SubAgentConfig(
        description="通用子代理：搜+算一体，处理需要多工具组合的复杂子任务",
        tools=["semantic_search", "keyword_search", "graph_search", "read_chunk",
               "mcp__python_default__execute_python", "finish"],
        max_iterations=10,
        system_prompt_override="你是财务分析通用代理。根据任务需要自由组合搜索和计算工具，独立完成子任务并返回完整结果。",
        model_hint="mid",
    ),
}
```

### 3.3 覆盖度验证

| 财报分析任务 | 推荐子 Agent | 说明 |
|-------------|-------------|------|
| "搜索宁德时代 2024 营收" | retrieval | 纯搜索 |
| "计算三家公司的 DuPont ROE 并排名" | analysis | 搜数据 + Python 计算 + 对比，一个 Agent 内完成 |
| "对比宁德时代和比亚迪的研发投入趋势" | analysis | 横向对比 + 趋势判断，需大模型 |
| "找出锂电行业 ROE 最高的公司" | general | 搜行业列表 → 逐个搜财务 → 算 ROE → 排序，多步长链路 |
| "评估恒大退市风险" | Plan → retrieval×2 + analysis | 主 Agent 拆分 Plan，并行搜 + 综合评分 |
| "验证报告中的 ROE 计算是否正确" | analysis | 搜原文数据 → Python 重算 → 对比差异 |

---

## 4. 完整执行时序

```
查询: "计算宁德时代 DuPont ROE 分解并对比行业均值"

主 Agent 循环:
  Round 1 → plan_query: 生成 Plan (step_1~5)
  Round 2 → dispatch_subagent × 3 (step_1,2,3 并行)
              ├─ SubAgent(retrieval, step_1): 利润表 → NPM=12.33%
              ├─ SubAgent(retrieval, step_2): 资产负债表 → assets, equity
              └─ SubAgent(retrieval, step_3): 行业ROE均值 12.7%
  Round 3 → dispatch_subagent(analysis, step_4):
              └─ execute_python(DuPont 分解) → ROE=18.20%
  Round 4 → dispatch_subagent(analysis, step_5):
              └─ 行业对比 → 高于均值 5.5pp
  Round 5 → finish: 最终答案 + Plan trace
```

---

## 5. 改动文件清单

| 文件 | 改动 | 量级 |
|------|------|------|
| `mcp_servers/__init__.py` | 新建 | ~3 行 |
| `mcp_servers/python_server.py` | 新建 Python MCP Server | ~120 行 |
| `config.py` | 新增 PYTHON_MCP_SERVER，自动合并到 MCP_SERVERS | ~10 行 |
| `agents/agentic/types.py` | 新增 Plan, PlanStep dataclass；SubAgentConfig 移除 computation | ~40 行 |
| `agents/agentic/tools.py` | plan_steps → plan_query，新增 plan_update 工具定义 | ~30 行 |
| `agents/agentic/agent.py` | Plan 生成/追踪/System Prompt 注入，dispatch 关联 step_id | ~50 行 |
| `agents/agentic/sub_agent.py` | 子 Agent 类型重构（移除 computation，新增 analysis/general） | ~30 行 |
| `agents/agentic/prompts.py` | 工具描述表更新，子 Agent 能力描述更新 | ~15 行 |
