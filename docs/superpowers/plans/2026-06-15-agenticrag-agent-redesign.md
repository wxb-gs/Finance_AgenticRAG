# AgenticRAG Agent 架构改造实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 PEV AgenticRAG 基础上增量添加 Claude Code 风格的 ReAct Agent，通过工具驱动的 while 循环替代固定图编排。

**Architecture:** Agent = System Prompt + Tool Registry + Simple Loop。System Prompt 是唯一行为来源，工具按三层分层（检索/元/生命周期），元工具 dispatch_subagent 实现递归任务分解。通过 PipelineRouter 与现有 PEV 并行共存，共享 retrieval/ 和 llm/ 层。

**Tech Stack:** Python 3.10+, OpenAI SDK (兼容 vLLM), FAISS, jieba, pytest, LangGraph (仅 PEV 侧复用)

---

## 文件依赖关系

```
types.py ─────────────────────────────────────────────┐
prompts.py ───────────────────────────────────────────┤
tools.py (↗ types, retrieval/) ───────────────────────┤
skills.py (↗ types) ──────────────────────────────────┤
context.py (↗ types, llm/) ───────────────────────────┤
memory.py (↗ types, llm/) ────────────────────────────┤
sub_agent.py (↗ types, agent, tools) ─────────────────┤
agent.py (↗ all above) ───────────────────────────────┤
__init__.py (↗ agent.py) ─────────────────────────────┤
pipeline_router.py (↗ agentic, PEV agents/) ──────────┤
evaluation/compare.py (↗ pipeline_router) ────────────┤
config.py (独立修改) ──────────────────────────────────┤
api/server.py (↗ pipeline_router) ────────────────────┘
```

---

### Task 1: 类型定义 `agents/agentic/types.py`

**Files:**
- Create: `agents/agentic/__init__.py` (空文件)
- Create: `agents/agentic/types.py`

**Purpose:** 定义 Agent 系统的所有共享数据结构，无外部依赖。

- [ ] **Step 1: 创建空 `__init__.py`**

```bash
touch agents/agentic/__init__.py
```

- [ ] **Step 2: 写入 `types.py`**

```python
"""Agentic Agent 共享类型定义"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Any


@dataclass
class ToolCall:
    """一次工具调用"""
    id: str
    name: str
    args: dict[str, Any]
    timestamp: float = 0.0


@dataclass
class ToolResult:
    """一次工具调用的结果"""
    call_id: str
    tool_name: str
    success: bool
    content: str                  # 序列化后的文本结果
    raw: list[dict] | None = None # 原始结构化结果
    confidence: float = 1.0       # 结果置信度 (0-1)
    is_empty: bool = False        # 是否为空结果
    has_contradiction: bool = False


@dataclass
class CompressionEvent:
    """一次上下文压缩事件"""
    before_tokens: int
    after_tokens: int
    strategy: Literal["aggressive", "summarize_old", "preserve_recent"]
    summary: str = ""


@dataclass
class AgentState:
    """Agent 运行时状态"""
    query: str
    iterations: int = 0
    total_tool_calls: int = 0
    tool_calls: list[dict] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    final_answer: str = ""
    skills_activated: list[str] = field(default_factory=list)
    subagents_dispatched: int = 0
    memories_recalled: int = 0
    compression_events: list[CompressionEvent] = field(default_factory=list)
    finished: bool = False
    token_usage: int = 0

    def add_tool_call(self, call: ToolCall, result: ToolResult):
        self.tool_calls.append({
            "call_id": call.id,
            "name": call.name,
            "args": call.args,
            "success": result.success,
            "content": result.content[:2000],
        })
        self.trace.append({
            "iteration": self.iterations,
            "tool_call": call.name,
            "args": call.args,
            "result_summary": result.content[:500],
        })
        self.total_tool_calls += 1

    def to_result(self) -> dict:
        return {
            "final_answer": self.final_answer,
            "iterations": self.iterations,
            "total_tool_calls": self.total_tool_calls,
            "trace": self.trace,
            "skills_activated": self.skills_activated,
            "subagents_dispatched": self.subagents_dispatched,
            "memories_recalled": self.memories_recalled,
            "compression_events": [
                {"strategy": e.strategy, "before": e.before_tokens, "after": e.after_tokens}
                for e in self.compression_events
            ],
        }


@dataclass
class AgentResult:
    """Agent 执行结果"""
    answer: str
    confidence: float
    iterations: int
    total_tool_calls: int
    trace: list[dict]
    skills_used: list[str]
    subagent_count: int
    memories_used: int
    compression_events: list[dict]
    evidence_summary: list[dict] = field(default_factory=list)


@dataclass
class ToolMeta:
    """工具元信息 — 用于 LLM 工具选择"""
    name: str
    category: Literal["retrieval", "meta", "lifecycle"]
    description: str
    when_to_use: str
    when_not_to_use: str
    parameters: dict
    priority: int = 0
    require_confirmation: bool = False

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": f"{self.description}\n\nUse when: {self.when_to_use}\nDo NOT use when: {self.when_not_to_use}",
                "parameters": self.parameters,
            },
        }


@dataclass
class SubAgentConfig:
    """子代理类型配置"""
    description: str
    tools: list[str]
    max_iterations: int
    system_prompt_override: str
    model_hint: Literal["small", "large"]
```

- [ ] **Step 3: 验证导入**

```bash
cd C:/lib/codes/python_projects/core && python -c "from agents.agentic.types import AgentState, ToolMeta, ToolCall, ToolResult; print('types OK')"
```

预期输出: `types OK`

---

### Task 2: System Prompt 模板 `agents/agentic/prompts.py`

**Files:**
- Create: `agents/agentic/prompts.py`

**Purpose:** 按模型规格（small/large）和语言（zh/en）提供 System Prompt 模板。无代码依赖。

- [ ] **Step 1: 写入 `prompts.py`**

```python
"""Agent System Prompt 模板 — 按模型尺寸和语言分层"""

# ══════════════════════════════════════════════════════════════════
# 英文基础模板
# ══════════════════════════════════════════════════════════════════

_EN_BASE = """You are a financial information retrieval Agent. Your task is to answer multi-hop financial queries by searching a knowledge base.

## Core Loop

Think → Act → Observe → Think → ... → Finish

For each step, decide:
- Do I have enough information to answer? → call `finish`
- Do I need to search? → call retrieval tools
- Can I split this into independent sub-tasks? → call `dispatch_subagent`
- Did I find something worth remembering? → call `remember`
- Is this a complex multi-step task? → call `plan_steps`

## Tool Selection Rules

- Use `keyword_search` for exact matches: company names, stock codes, dates, amounts
- Use `semantic_search` for conceptual or fuzzy queries
- Use `graph_search` for entity relationship queries
- Use `read_chunk` to fetch full text by chunk_id
- Use `hybrid_search` when high recall matters across retrieval methods

## Behavior Rules

- Search before answering. Never guess.
- When finding contradictions across sources, mark them, don't force consistency.
- Dispatch independent sub-queries in parallel via `dispatch_subagent`.
- Be thorough but efficient: don't re-search what you already have.

## Stopping Conditions

- Call `finish` when you have sufficient evidence.
- If you cannot answer after exhausting searches, call `finish` with confidence=0.
- Don't loop indefinitely — after 3 rounds with no new evidence, finish.
"""

_EN_SMALL_EXTRA = """
## Model Constraints (Small Model)

You are running on a small model (7B-14B). Follow these extra constraints:
- Call only ONE tool per turn.
- `dispatch_subagent` is NOT available — work sequentially.
- Tool results over 500 characters will be truncated to the first 3 items.
- Keep reasoning concise: 1-2 sentences max per think step.
"""

_EN_LARGE_EXTRA = """
## Model Capabilities (Large Model)

You are running on a large model (32B+). You may:
- Call multiple independent tools in a single turn.
- Use `dispatch_subagent` for parallel decomposition.
- Reason at length when the problem requires it.
"""

# ══════════════════════════════════════════════════════════════════
# 中文基础模板
# ══════════════════════════════════════════════════════════════════

_ZH_BASE = """你是金融信息检索 Agent。你的任务是通过搜索知识库来回答多跳金融查询。

## 核心循环

思考 → 行动 → 观察 → 思考 → ... → 完成

每步决定：
- 信息是否足够回答问题？→ 调用 `finish`
- 是否需要搜索？→ 调用检索工具
- 能否拆分为独立子任务？→ 调用 `dispatch_subagent`
- 是否发现了值得记住的信息？→ 调用 `remember`
- 是否是复杂的多步任务？→ 调用 `plan_steps`

## 工具选择规则

- 精确匹配（公司全称、代码、日期、金额）→ `keyword_search`
- 模糊语义查询（概念解释、趋势分析）→ `semantic_search`
- 实体关系查询（股东、子公司、关联方）→ `graph_search`
- 需要读取完整文本块 → `read_chunk`
- 需要跨多个检索方法高召回 → `hybrid_search`

## 行为规则

- 先搜索再回答，绝不猜测。
- 发现不同来源的数据矛盾时，标注差异而不是强行统一。
- 使用 `dispatch_subagent` 并行获取独立的子任务。
- 彻底但高效：不要重复搜索已有的信息。

## 停止条件

- 证据充分 → 调用 `finish`
- 穷尽搜索仍无法回答 → 调用 `finish` 并设置 confidence=0
- 连续 3 轮没有获得新证据 → 强制结束
"""

_ZH_SMALL_EXTRA = """
## 模型限制（小模型）

你运行在 7B-14B 小模型上，请遵循以下额外限制：
- 每轮只调用一个工具。
- `dispatch_subagent` 不可用，请按顺序串行推理。
- 工具结果超过 500 字时仅保留前 3 条。
- 每步推理保持简洁：1-2 句。
"""

_ZH_LARGE_EXTRA = """
## 模型能力（大模型）

你运行在 32B+ 大模型上，你可以：
- 在同一轮中调用多个无依赖的工具。
- 使用 `dispatch_subagent` 进行并行任务分解。
- 复杂问题可以详细推理。
"""


def get_system_prompt(model_size: str, language: str = "zh") -> str:
    """获取基础 System Prompt

    Args:
        model_size: "small" (7B-14B) | "mid" (32B-70B) | "large" (70B+)
        language: "zh" | "en"
    """
    if language == "en":
        base = _EN_BASE
        extra = _EN_LARGE_EXTRA if model_size in ("mid", "large") else _EN_SMALL_EXTRA
    else:
        base = _ZH_BASE
        extra = _ZH_LARGE_EXTRA if model_size in ("mid", "large") else _ZH_SMALL_EXTRA

    return base + extra


def get_tool_descriptions(language: str = "zh") -> str:
    """获取工具描述段落（注入到 System Prompt 的工具选择规则之后）"""
    if language == "zh":
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
| dispatch_subagent | 可拆分为 2+ 独立子任务 | 简单单步、强依赖任务 |
| remember | 发现关键证据、矛盾点 | 常规检索结果 |
| plan_steps | 3+ 步的复杂任务 | 简单 1-2 步查询 |
| finish | 完成回答 | — |
"""
    else:
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
| dispatch_subagent | 2+ independent subtasks | Simple or tightly-dependent tasks |
| remember | Key evidence, contradictions found | Routine search results |
| plan_steps | 3+ step complex tasks | Simple 1-2 step queries |
| finish | Complete answer | — |
"""
```

- [ ] **Step 2: 验证 Prompt 渲染**

```bash
cd C:/lib/codes/python_projects/core && python -c "
from agents.agentic.prompts import get_system_prompt, get_tool_descriptions
p = get_system_prompt('small', 'zh')
assert '小模型' in p
assert 'dispatch_subagent' in p
t = get_tool_descriptions('zh')
assert 'semantic_search' in t
print('prompts OK')
"
```

预期输出: `prompts OK`

---

### Task 3: 工具注册表 `agents/agentic/tools.py`

**Files:**
- Create: `agents/agentic/tools.py`

**Purpose:** 三层工具注册表，统一管理检索工具、元工具、生命周期工具。每个工具附带 `when_to_use` / `when_not_to_use` 指导。复用现有 `retrieval/` 全部工具。

- [ ] **Step 1: 写入 `tools.py`**

```python
"""工具注册表 — 三层分层：检索 / 元 / 生命周期"""
import json
import time
import uuid
from typing import Any

from agents.agentic.types import ToolMeta, ToolCall, ToolResult

# ══════════════════════════════════════════════════════════════════
# 检索工具定义
# ══════════════════════════════════════════════════════════════════

_RETRIEVAL_TOOL_DEFS = [
    ToolMeta(
        name="semantic_search",
        category="retrieval",
        description="FAISS dense vector semantic search with BGE-reranker. Returns top-ranked text chunks.",
        when_to_use="概念性查询、语义模糊查询、需要理解上下文的查询",
        when_not_to_use="精确名称匹配、股票代码查询——优先用 keyword_search",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query text"},
                "top_k": {"type": "integer", "description": "Number of results (default 20)"},
            },
            "required": ["query"],
        },
        priority=6,
    ),
    ToolMeta(
        name="keyword_search",
        category="retrieval",
        description="BM25 keyword-based search with jieba tokenizer. Best for exact term matching.",
        when_to_use="精确字段匹配：公司全称、股票代码、日期、金额、专有名词",
        when_not_to_use="模糊语义查询、概念解释——优先用 semantic_search",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keyword query text"},
                "top_k": {"type": "integer", "description": "Number of results (default 20)"},
            },
            "required": ["query"],
        },
        priority=8,
    ),
    ToolMeta(
        name="graph_search",
        category="retrieval",
        description="Knowledge graph entity traversal. Finds relationships between entities.",
        when_to_use="实体关系查询：A公司股东是谁、X与Y什么关系、供应链上下游",
        when_not_to_use="直接数值查询、文本片段检索——优先用 keyword_search 或 semantic_search",
        parameters={
            "type": "object",
            "properties": {
                "entity": {"type": "string", "description": "Starting entity name"},
                "relation": {"type": "string", "description": "Relation type (optional)"},
            },
            "required": ["entity"],
        },
        priority=5,
    ),
    ToolMeta(
        name="hybrid_search",
        category="retrieval",
        description="Multi-method retrieval with RRF fusion and CrossEncoder reranking. Highest recall.",
        when_to_use="高召回场景、关键信息可能被单一方法遗漏时",
        when_not_to_use="简单单步查询、对延迟敏感的场景",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query text"},
                "tools": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["semantic_search", "keyword_search"]},
                    "description": "Retrieval methods to fuse (default: both)",
                },
            },
            "required": ["query"],
        },
        priority=3,
    ),
    ToolMeta(
        name="read_chunk",
        category="retrieval",
        description="Read full text of a specific chunk by its ID.",
        when_to_use="已知 chunk_id 需要获取完整文本内容时",
        when_not_to_use="没有具体 chunk_id 的搜索——用其他检索工具",
        parameters={
            "type": "object",
            "properties": {
                "chunk_id": {"type": "string", "description": "Chunk ID to retrieve"},
            },
            "required": ["chunk_id"],
        },
        priority=2,
    ),
]

# ══════════════════════════════════════════════════════════════════
# 元工具定义
# ══════════════════════════════════════════════════════════════════

_META_TOOL_DEFS = [
    ToolMeta(
        name="dispatch_subagent",
        category="meta",
        description="Spawn a sub-agent to handle an independent sub-task. Sub-agents run with restricted tools and context.",
        when_to_use="任务可以分解为 2+ 个独立不共享状态的子问题",
        when_not_to_use="简单单步查询、子问题之间有强顺序依赖",
        parameters={
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Sub-task description"},
                "agent_type": {
                    "type": "string",
                    "enum": ["retrieval", "comparison", "computation"],
                    "description": "Type of sub-agent",
                },
                "run_in_background": {
                    "type": "boolean",
                    "description": "Run in background (true for parallel execution)",
                },
            },
            "required": ["task"],
        },
    ),
    ToolMeta(
        name="remember",
        category="meta",
        description="Save key evidence, contradictions, or knowledge gaps to persistent memory for future queries.",
        when_to_use="发现经过验证的关键证据、来源之间的矛盾、确认的信息缺失",
        when_not_to_use="常规检索结果——这些已在消息历史中",
        parameters={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Content to save"},
                "type": {
                    "type": "string",
                    "enum": ["evidence", "contradiction", "gap", "pattern"],
                    "description": "Memory type",
                },
            },
            "required": ["content", "type"],
        },
    ),
    ToolMeta(
        name="plan_steps",
        category="meta",
        description="Create a structured task plan for complex multi-step queries.",
        when_to_use="复杂查询有 3+ 步需要结构化追踪",
        when_not_to_use="简单 1-2 步查询",
        parameters={
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "description": {"type": "string"},
                            "depends_on": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["id", "description"],
                    },
                },
            },
            "required": ["steps"],
        },
    ),
]

# ══════════════════════════════════════════════════════════════════
# 生命周期工具定义
# ══════════════════════════════════════════════════════════════════

_LIFECYCLE_TOOL_DEFS = [
    ToolMeta(
        name="finish",
        category="lifecycle",
        description="Output the final answer with confidence and evidence summary. Call this when you have enough information.",
        when_to_use="所有必要证据已收集完毕，或确认无法回答",
        when_not_to_use="还有未尝试的搜索路径",
        parameters={
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "Final answer text"},
                "evidence_summary": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key evidence points supporting the answer",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Confidence score (0-1)",
                },
            },
            "required": ["answer", "confidence"],
        },
    ),
]


class ToolRegistry:
    """统一工具注册表 — 内置 + MCP (预留)"""

    def __init__(self, model_size: str = "large"):
        self.model_size = model_size
        self._builtin: dict[str, ToolMeta] = {}
        self._mcp: dict[str, Any] = {}
        self._skill_provided: dict[str, ToolMeta] = {}
        self._register_all()

    def _register_all(self):
        for t in _RETRIEVAL_TOOL_DEFS:
            self._builtin[t.name] = t
        for t in _META_TOOL_DEFS:
            if t.name == "dispatch_subagent" and self.model_size == "small":
                continue  # 小模型不可用
            self._builtin[t.name] = t
        for t in _LIFECYCLE_TOOL_DEFS:
            self._builtin[t.name] = t

    def get_all_schemas(self) -> list[dict]:
        """合并所有工具，统一返回 OpenAI function schema 列表"""
        schemas = []
        for name, meta in self._builtin.items():
            schemas.append(meta.to_openai_schema())
        for name, meta in self._skill_provided.items():
            schemas.append(meta.to_openai_schema())
        for name, tool in self._mcp.items():
            schemas.append({
                "type": "function",
                "function": {
                    "name": f"mcp__{name}",
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                },
            })
        return schemas

    def get_meta(self, name: str) -> ToolMeta | None:
        return self._builtin.get(name) or self._skill_provided.get(name)

    def is_meta_tool(self, name: str) -> bool:
        meta = self.get_meta(name)
        return meta is not None and meta.category == "meta"

    def is_lifecycle_tool(self, name: str) -> bool:
        meta = self.get_meta(name)
        return meta is not None and meta.category == "lifecycle"

    def is_retrieval_tool(self, name: str) -> bool:
        meta = self.get_meta(name)
        return meta is not None and meta.category == "retrieval"

    def discover_mcp(self, servers: list[str] | None = None):
        """预留 MCP 工具发现接口（一期不实现）"""
        pass

    # ══════════════════════════════════════════════════════════════
    # 工具执行
    # ══════════════════════════════════════════════════════════════

    def execute(self, call: ToolCall) -> ToolResult:
        """同步执行单个工具调用"""
        name = call.name
        args = call.args

        # 检索工具
        if name == "semantic_search":
            return self._exec_semantic_search(call)
        elif name == "keyword_search":
            return self._exec_keyword_search(call)
        elif name == "graph_search":
            return self._exec_graph_search(call)
        elif name == "hybrid_search":
            return self._exec_hybrid_search(call)
        elif name == "read_chunk":
            return self._exec_read_chunk(call)
        # 元工具 — 返回 sentinel 标记，由 Agent 循环处理
        elif name in ("dispatch_subagent", "remember", "plan_steps"):
            return ToolResult(
                call_id=call.id,
                tool_name=name,
                success=True,
                content=json.dumps(args, ensure_ascii=False),
                confidence=1.0,
            )
        # 生命周期工具
        elif name == "finish":
            return ToolResult(
                call_id=call.id,
                tool_name=name,
                success=True,
                content=json.dumps(args, ensure_ascii=False),
                confidence=args.get("confidence", 1.0),
            )
        else:
            return ToolResult(
                call_id=call.id,
                tool_name=name,
                success=False,
                content=f"Unknown tool: {name}",
            )

    def _exec_semantic_search(self, call: ToolCall) -> ToolResult:
        from retrieval.semantic_search import semantic_search
        results = semantic_search(query=call.args["query"],
                                  top_k=call.args.get("top_k", 20))
        return ToolResult(
            call_id=call.id, tool_name=call.name, success=True,
            content=self._format_results(results),
            raw=results, is_empty=len(results) == 0,
        )

    def _exec_keyword_search(self, call: ToolCall) -> ToolResult:
        from retrieval.keyword_search import keyword_search
        results = keyword_search(query=call.args["query"],
                                  top_k=call.args.get("top_k", 20))
        return ToolResult(
            call_id=call.id, tool_name=call.name, success=True,
            content=self._format_results(results),
            raw=results, is_empty=len(results) == 0,
        )

    def _exec_graph_search(self, call: ToolCall) -> ToolResult:
        from retrieval.graph_search import graph_search
        results = graph_search(entity=call.args["entity"],
                                relation=call.args.get("relation"))
        return ToolResult(
            call_id=call.id, tool_name=call.name, success=True,
            content=self._format_results(results),
            raw=results, is_empty=len(results) == 0,
        )

    def _exec_hybrid_search(self, call: ToolCall) -> ToolResult:
        from retrieval.hybrid_search import multi_tool_search
        tools = call.args.get("tools", ["keyword_search", "semantic_search"])
        results = multi_tool_search(query=call.args["query"], tools=tools)
        return ToolResult(
            call_id=call.id, tool_name=call.name, success=True,
            content=self._format_results(results),
            raw=results, is_empty=len(results) == 0,
        )

    def _exec_read_chunk(self, call: ToolCall) -> ToolResult:
        from retrieval.read_chunk import read_chunk
        results = read_chunk(chunk_id=call.args["chunk_id"])
        return ToolResult(
            call_id=call.id, tool_name=call.name, success=bool(results),
            content=results[0].get("text", "") if results else "Chunk not found",
            raw=results, is_empty=len(results) == 0,
        )

    def _format_results(self, results: list[dict]) -> str:
        """将检索结果序列化为 LLM 可读的文本"""
        if not results:
            return "[No results found]"
        lines = []
        for i, r in enumerate(results[:10]):
            lines.append(
                f"[{i+1}] chunk_id={r.get('chunk_id', '?')} "
                f"score={r.get('score', 0):.4f}\n"
                f"    {r.get('text', '')[:300]}"
            )
        if len(results) > 10:
            lines.append(f"... and {len(results) - 10} more results")
        return "\n".join(lines)
```

- [ ] **Step 2: 验证工具注册和 schema 生成**

```bash
cd C:/lib/codes/python_projects/core && python -c "
from agents.agentic.tools import ToolRegistry
r = ToolRegistry(model_size='large')
schemas = r.get_all_schemas()
names = [s['function']['name'] for s in schemas]
assert 'semantic_search' in names
assert 'keyword_search' in names
assert 'dispatch_subagent' in names
assert 'finish' in names
print(f'{len(schemas)} tools registered: {names}')

# 验证小模型排除 dispatch_subagent
r_small = ToolRegistry(model_size='small')
small_names = [s['function']['name'] for s in r_small.get_all_schemas()]
assert 'dispatch_subagent' not in small_names
print(f'small model: {len(small_names)} tools')
print('tools OK')
"
```

预期输出: `9 tools registered: ...`, `small model: 8 tools`, `tools OK`

- [ ] **Step 3: 验证工具执行（需要索引）**

```bash
cd C:/lib/codes/python_projects/core && python -c "
from agents.agentic.tools import ToolRegistry
from agents.agentic.types import ToolCall
import uuid

r = ToolRegistry()
call = ToolCall(id=str(uuid.uuid4()), name='semantic_search', args={'query': '营收增长率'})
result = r.execute(call)
print(f'success={result.success}, empty={result.is_empty}, content_len={len(result.content)}')
print('execute OK')
"
```

预期输出: `success=True, ...`, `execute OK`

---

### Task 4: Skills 技能系统 `agents/agentic/skills.py`

**Files:**
- Create: `agents/agentic/skills.py`

**Purpose:** 按查询关键词匹配领域 Skills，注入对应行为指令到 System Prompt。

- [ ] **Step 1: 写入 `skills.py`**

```python
"""Skills 技能系统 — 按需加载的领域能力模块"""
from dataclasses import dataclass, field


@dataclass
class Skill:
    """技能：领域知识 + 行为约束的模块"""
    name: str
    description: str
    trigger_keywords: list[str]
    prompt_extension: str
    restricted_tools: list[str] | None = None
    override_max_iterations: int | None = None

    def matches(self, query: str) -> bool:
        return any(kw in query for kw in self.trigger_keywords)


# ══════════════════════════════════════════════════════════════════
# 技能注册表
# ══════════════════════════════════════════════════════════════════

SKILL_REGISTRY: dict[str, Skill] = {
    "financial-statement-analysis": Skill(
        name="financial-statement-analysis",
        description="深度解析财报三大表，计算关键财务比率",
        trigger_keywords=["财报", "营收", "净利润", "ROE", "资产负债", "现金流",
                          "毛利润", "净利率", "流动比率", "速动比率", "应收账款",
                          "存货周转", "营业成本", "归母净利润"],
        prompt_extension="""
## 激活技能：财务报表分析

当前查询涉及财务报表分析，遵循以下工作流：

1. **识别报表范围**：确认需要哪些报表期间、哪些科目
2. **交叉验证**：同一数据如果多个来源有出入，标注差异而非选择其一
3. **比率计算要点**：
   - 盈利能力：毛利率、净利率、ROE、ROA
   - 偿债能力：流动比率、速动比率、资产负债率
   - 运营效率：存货周转率、应收账款周转率
4. **趋势判断**：对比多期数据时给出方向性结论
5. **置信度要求**：财报分析要求更高置信度 (>= 0.8)
""",
    ),
    "risk-assessment": Skill(
        name="risk-assessment",
        description="评估公司的财务风险、经营风险和合规风险",
        trigger_keywords=["风险", "违约", "担保", "诉讼", "ST", "退市",
                          "处罚", "监管", "立案", "警示函", "问询函"],
        prompt_extension="""
## 激活技能：风险评估

当前查询涉及风险评估，遵循：

1. **风险分类**：先将风险归类（财务/经营/合规/市场）
2. **证据收集**：不仅找正面证据，必须主动搜索负面信号
3. **对立分析**：对每个风险点，同时搜索支持和反对的证据
4. **量化评分**：给每类风险打分（1-5），附带依据
""",
    ),
    "multi-hop-comparison": Skill(
        name="multi-hop-comparison",
        description="跨公司、跨时间段的对比分析",
        trigger_keywords=["对比", "比较", "差异", "排名", "优于", "不如",
                          "最高", "最低", "大于", "小于", "超过"],
        prompt_extension="""
## 激活技能：多跳对比

当前查询涉及对比分析，遵循：

1. **并行获取**：所有可并行的信息获取通过 `dispatch_subagent` 并行执行
   - 不同公司 → 每个公司一个 retrieval 子代理
   - 不同指标 → 每个指标一个 computation 子代理
2. **输出格式**：表格优先，每列一家公司/时间段，每行一个指标
3. **差异标注**：关键差异高亮，附带来源引用
""",
    ),
}


class SkillManager:
    """技能管理器 — 匹配与注入"""

    def __init__(self, model_size: str = "large"):
        self.model_size = model_size
        self.loaded: list[Skill] = []

    def match(self, query: str) -> list[Skill]:
        """根据查询匹配应激活的 Skills"""
        matched = []
        for name, skill in SKILL_REGISTRY.items():
            if skill.matches(query):
                matched.append(skill)

        # 小模型限制
        if self.model_size == "small" and len(matched) > 1:
            # 按 trigger_keywords 命中数排序，取最佳
            matched.sort(
                key=lambda s: sum(1 for kw in s.trigger_keywords if kw in query),
                reverse=True,
            )
            matched = matched[:1]

        self.loaded = matched
        return matched

    def build_system_prompt(self, query: str, language: str = "zh") -> str:
        """组装完整 System Prompt：基础 + Skills 扩展"""
        from agents.agentic.prompts import get_system_prompt, get_tool_descriptions

        prompt = get_system_prompt(self.model_size, language)
        prompt += "\n" + get_tool_descriptions(language)

        matched = self.match(query)
        for skill in matched:
            prompt += f"\n\n{skill.prompt_extension}"

        if self.model_size == "small" and len(matched) > 0:
            prompt += "\n\n注意：小模型模式下只激活一个技能。"

        return prompt

    def get_active_skill_names(self) -> list[str]:
        return [s.name for s in self.loaded]
```

- [ ] **Step 2: 验证 Skill 匹配**

```bash
cd C:/lib/codes/python_projects/core && python -c "
from agents.agentic.skills import SkillManager

sm = SkillManager(model_size='large')
# 财报查询
matched = sm.match('分析一下这家公司2024年的营收和净利润增长率')
assert len(matched) == 1
assert matched[0].name == 'financial-statement-analysis'
print(f'FS match: {[s.name for s in matched]}')

# 无匹配
matched2 = sm.match('这家公司什么时候上市的')
assert len(matched2) == 0
print(f'No match: {len(matched2)}')

# 小模型只取 1 个
sm_small = SkillManager(model_size='small')
matched3 = sm_small.match('对比分析A公司和B公司的营收差异和风险指标')
assert len(matched3) == 1
print(f'Small model only 1: {matched3[0].name}')
print('skills OK')
"
```

预期输出: `FS match: ['financial-statement-analysis']`, `No match: 0`, `Small model only 1: ...`, `skills OK`

- [ ] **Step 3: 验证 System Prompt 组装**

```bash
cd C:/lib/codes/python_projects/core && python -c "
from agents.agentic.skills import SkillManager

sm = SkillManager(model_size='large')
prompt = sm.build_system_prompt('分析公司2024年营收和净利润', 'zh')
assert '财务报表分析' in prompt
assert 'semantic_search' in prompt or '工具' in prompt
print(f'Prompt length: {len(prompt)} chars')
print('prompt build OK')
"
```

预期输出: `Prompt length: ... chars`, `prompt build OK`

---

### Task 5: 上下文压缩 `agents/agentic/context.py`

**Files:**
- Create: `agents/agentic/context.py`

**Purpose:** 当 token 使用量超过 80% 水位线时触发压缩，按模型尺寸三种策略。

- [ ] **Step 1: 写入 `context.py`**

```python
"""上下文压缩 — 三层策略按模型规格"""
import tiktoken
from typing import Literal

from agents.agentic.types import CompressionEvent


def count_tokens(messages: list[dict], model: str = "gpt-4") -> int:
    """估算消息列表的 token 数"""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    total = 0
    for msg in messages:
        for key in ("content", "tool_calls", "tool_call_id"):
            if key in msg and msg[key]:
                if isinstance(msg[key], str):
                    total += len(enc.encode(msg[key]))
                elif isinstance(msg[key], list):
                    for item in msg[key]:
                        if isinstance(item, dict) and "function" in item:
                            total += len(enc.encode(str(item["function"])))
        if "role" in msg:
            total += 4
    return total


class ContextManager:
    """三层上下文压缩策略"""

    def __init__(self, model_size: Literal["small", "mid", "large"],
                 max_tokens: int | None = None):
        self.model_size = model_size
        # 默认值
        if max_tokens is not None:
            self.max_tokens = max_tokens
        elif model_size == "small":
            self.max_tokens = 8192
        elif model_size == "mid":
            self.max_tokens = 16384
        else:
            self.max_tokens = 32768

    def should_compress(self, messages: list[dict]) -> bool:
        current = count_tokens(messages)
        return current > self.max_tokens * 0.8

    def compress(self, messages: list[dict]) -> tuple[list[dict], CompressionEvent]:
        """压缩消息列表，返回 (压缩后消息, 压缩事件)"""
        before = count_tokens(messages)

        if self.model_size == "small":
            result, strategy = self._aggressive(messages), "aggressive"
        elif self.model_size == "mid":
            result, strategy = self._summarize_old(messages), "summarize_old"
        else:
            result, strategy = self._preserve_recent(messages), "preserve_recent"

        after = count_tokens(result)
        return result, CompressionEvent(
            before_tokens=before,
            after_tokens=after,
            strategy=strategy,
        )

    def _aggressive(self, messages: list[dict]) -> list[dict]:
        """小模型激进策略：System Prompt + 摘要 + 最近 4 条"""
        if len(messages) <= 5:
            return messages
        system_msgs = [m for m in messages if m.get("role") == "system"]
        rest = [m for m in messages if m.get("role") != "system"]

        # 压缩旧消息为摘要
        old = rest[1:-4]
        summary = self._summarize_tool_results(old)

        compressed = [
            *system_msgs,
            {"role": "system", "content": f"[前期检索摘要]\n{summary}"},
            *rest[-4:],
        ]
        return compressed

    def _summarize_old(self, messages: list[dict]) -> list[dict]:
        """中等模型：保留 System Prompt + 摘要 + 最近 6 条"""
        if len(messages) <= 8:
            return messages
        system_msgs = [m for m in messages if m.get("role") == "system"]
        rest = [m for m in messages if m.get("role") != "system"]

        old = rest[1:-6]
        summary = self._summarize_tool_results(old)

        return [
            *system_msgs,
            {"role": "system", "content": f"[历史检索摘要]\n{summary}"},
            *rest[-6:],
        ]

    def _preserve_recent(self, messages: list[dict]) -> list[dict]:
        """大模型：仅移除最早的 tool 结果，保留推理链"""
        kept = []
        removed_tool = 0
        for i, msg in enumerate(messages):
            is_old_tool = (msg.get("role") == "tool" and
                           i < len(messages) - 12 and
                           removed_tool < 20)
            if is_old_tool:
                removed_tool += 1
                continue
            kept.append(msg)
        return kept

    def _summarize_tool_results(self, messages: list[dict]) -> str:
        """将旧消息中的工具结果压缩为结构化摘要"""
        data_points = []
        seen_chunks = set()
        for msg in messages:
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                for line in content.split("\n"):
                    if line.startswith("[") and "chunk_id=" in line:
                        chunk_id = line.split("chunk_id=")[1].split()[0]
                        if chunk_id not in seen_chunks:
                            seen_chunks.add(chunk_id)
                            text = line.split("\n    ")[-1][:150] if "\n    " in line else ""
                            data_points.append(f"- {text} | source:{chunk_id}")

        if not data_points:
            return "（早期轮次无有效检索结果）"

        return "[数据]\n" + "\n".join(data_points[:20])
```

- [ ] **Step 2: 验证压缩逻辑**

```bash
cd C:/lib/codes/python_projects/core && python -c "
from agents.agentic.context import ContextManager, count_tokens

msgs = [
    {'role': 'system', 'content': 'You are an Agent.'},
    {'role': 'user', 'content': '查询营收数据'},
    {'role': 'assistant', 'content': 'Let me search.'},
    {'role': 'tool', 'content': '[1] chunk_id=001 score=0.9\n    2024年营收为123亿元'},
]
for i in range(10):
    msgs.append({'role': 'assistant', 'content': f'step {i}: continue searching...'})
    msgs.append({'role': 'tool', 'content': f'[results for step {i}] some content here'})

print(f'Before: {count_tokens(msgs)} tokens')

mgr = ContextManager(model_size='small', max_tokens=4000)
should_compress = mgr.should_compress(msgs)
print(f'Should compress: {should_compress}')

compressed, event = mgr.compress(msgs)
print(f'After: {count_tokens(compressed)} tokens, strategy={event.strategy}')
print('context OK')
"
```

预期输出: token counts and strategy, `context OK`

---

### Task 6: Memory 持久化 `agents/agentic/memory.py`

**Files:**
- Create: `agents/agentic/memory.py`

**Purpose:** 文件系统记忆存储，四类记忆（evidence/contradiction/gap/pattern），MEMORY.md 索引。

- [ ] **Step 1: 写入 `memory.py`**

```python
"""Memory 持久化系统 — 跨会话的知识记忆"""
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import jieba

MEMORY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "memory"
)


@dataclass
class Memory:
    name: str
    type: str                      # evidence | contradiction | gap | pattern
    description: str
    content: str
    source_query: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    access_count: int = 0

    def to_markdown(self) -> str:
        return f"""---
name: {self.name}
description: {self.description}
metadata:
  type: {self.type}
  source_query: {self.source_query}
  created_at: {self.created_at}
  access_count: {self.access_count}
---

{self.content}
"""


class MemoryManager:
    def __init__(self, base_dir: str = MEMORY_DIR):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index: dict[str, Memory] = {}
        self._load_index()

    def _load_index(self):
        index_file = self.base_dir / "MEMORY.md"
        if not index_file.exists():
            return
        for line in index_file.read_text(encoding="utf-8").split("\n"):
            match = re.match(r"- \[(.*?)\]\((.*?)\) — (.*)", line)
            if match:
                name = match.group(1)
                desc = match.group(3)
                mem_file = self.base_dir / f"{match.group(2)}"
                if mem_file.exists():
                    self.index[name] = Memory(
                        name=name, type="evidence", description=desc,
                        content="", source_query="",
                    )

    def _update_index_file(self):
        lines = []
        for mem in self.index.values():
            filename = f"{mem.name}.md"
            lines.append(f"- [{mem.name}]({filename}) — {mem.description}")
        (self.base_dir / "MEMORY.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )

    def save(self, content: str, mem_type: str, query: str) -> Memory:
        """保存记忆"""
        name = self._generate_name(content)
        description = content[:150].replace("\n", " ")

        mem = Memory(
            name=name,
            type=mem_type,
            description=description,
            content=content,
            source_query=query,
        )
        self.index[name] = mem
        filepath = self.base_dir / f"{name}.md"
        filepath.write_text(mem.to_markdown(), encoding="utf-8")
        self._update_index_file()
        return mem

    def recall(self, query: str, top_k: int = 5) -> list[Memory]:
        """关键词召回相关记忆"""
        scored = []
        query_tokens = set(jieba.cut(query))
        for mem in self.index.values():
            score = sum(1 for t in query_tokens
                       if t in mem.description or t in mem.content)
            if score > 0:
                scored.append((mem, score))
        scored.sort(key=lambda x: (x[1], x[0].access_count), reverse=True)
        return [m for m, _ in scored[:top_k]]

    def forget(self, name: str):
        filepath = self.base_dir / f"{name}.md"
        if filepath.exists():
            filepath.unlink()
        if name in self.index:
            del self.index[name]
        self._update_index_file()

    def _generate_name(self, content: str) -> str:
        """生成 kebab-case 标识符"""
        name = content[:60].strip().lower()
        name = re.sub(r'[^\\w\\-]', '-', name)
        name = re.sub(r'-+', '-', name).strip('-')
        return name[:50] or "untitled"
```

- [ ] **Step 2: 验证 Memory 存取**

```bash
cd C:/lib/codes/python_projects/core && python -c "
import tempfile, os
from agents.agentic.memory import MemoryManager

with tempfile.TemporaryDirectory() as tmp:
    mgr = MemoryManager(base_dir=tmp)
    
    # Save
    mem = mgr.save('2024年Q1营收为123亿元，Q2营收为156亿元，环比增长26.8%', 
                    'evidence', '分析营收增长趋势')
    print(f'Saved: {mem.name}')
    assert os.path.exists(os.path.join(tmp, f'{mem.name}.md'))
    assert os.path.exists(os.path.join(tmp, 'MEMORY.md'))
    
    # Recall
    recalled = mgr.recall('营收增长')
    print(f'Recalled {len(recalled)}: {[r.description[:50] for r in recalled]}')
    assert len(recalled) == 1
    
    # Forget
    mgr.forget(mem.name)
    recalled2 = mgr.recall('营收增长')
    assert len(recalled2) == 0
    print('memory OK')
"
```

预期输出: `Saved: ...`, `Recalled 1: ...`, `memory OK`

---

### Task 7: 子代理系统 `agents/agentic/sub_agent.py`

**Files:**
- Create: `agents/agentic/sub_agent.py`

**Purpose:** 三种子代理类型，独立上下文执行，支持 background 并行。

- [ ] **Step 1: 写入 `sub_agent.py`**

```python
"""子代理系统 — 任务分解与并行执行"""
import asyncio
import uuid
from typing import Any

from agents.agentic.types import SubAgentConfig, ToolCall, ToolResult

# ══════════════════════════════════════════════════════════════════
# 子代理类型配置
# ══════════════════════════════════════════════════════════════════

SUBAGENT_TYPES: dict[str, SubAgentConfig] = {
    "retrieval": SubAgentConfig(
        description="聚焦的信息检索：搜索、读取、筛选证据",
        tools=["semantic_search", "keyword_search", "graph_search", "read_chunk", "finish"],
        max_iterations=5,
        system_prompt_override="你是信息检索专家。快速定位相关信息，返回结构化结果。不做深度分析推理。",
        model_hint="small",
    ),
    "comparison": SubAgentConfig(
        description="多源数据对比分析，找出差异和一致点",
        tools=["semantic_search", "keyword_search", "read_chunk", "finish"],
        max_iterations=8,
        system_prompt_override="你是财务分析专家。仔细对比多源数据，标注矛盾点和一致点。输出表格对比。",
        model_hint="large",
    ),
    "computation": SubAgentConfig(
        description="精确数值计算、比率分析、趋势计算",
        tools=["finish"],
        max_iterations=3,
        system_prompt_override="你是财务计算专家。精确计算并展示推导过程。只计算，不检索。",
        model_hint="small",
    ),
}


class SubAgentManager:
    """子代理管理器 — 派发、并行执行、结果合并"""

    def __init__(self, agent_factory):
        self.agent_factory = agent_factory

    async def dispatch(self, task: str, agent_type: str = "retrieval",
                       background: bool = False) -> dict:
        """派发一个子代理任务

        Args:
            task: 子任务描述
            agent_type: 子代理类型
            background: 是否后台执行（目前同步返回结果）

        Returns:
            {"type": "subagent_result", "task": str, "findings": str, "iterations": int}
        """
        config = SUBAGENT_TYPES.get(agent_type, SUBAGENT_TYPES["retrieval"])

        sub_agent = self.agent_factory(config)
        result = sub_agent.run(task)

        return {
            "type": "subagent_result",
            "agent_type": agent_type,
            "task": task,
            "findings": result.final_answer,
            "iterations": result.iterations,
            "tool_calls": result.total_tool_calls,
        }

    async def dispatch_parallel(self, tasks: list[dict]) -> list[dict]:
        """并行派发多个无依赖的子代理"""
        coroutines = [
            self.dispatch(
                task=t["task"],
                agent_type=t.get("agent_type", "retrieval"),
                background=t.get("background", False),
            )
            for t in tasks
        ]
        return await asyncio.gather(*coroutines)

    @staticmethod
    def get_config(agent_type: str) -> SubAgentConfig:
        return SUBAGENT_TYPES.get(agent_type, SUBAGENT_TYPES["retrieval"])
```

- [ ] **Step 2: 验证子代理配置**

```bash
cd C:/lib/codes/python_projects/core && python -c "
from agents.agentic.sub_agent import SUBAGENT_TYPES, SubAgentManager

assert 'retrieval' in SUBAGENT_TYPES
assert 'comparison' in SUBAGENT_TYPES
assert 'computation' in SUBAGENT_TYPES
r = SUBAGENT_TYPES['retrieval']
assert r.max_iterations == 5
assert 'semantic_search' in r.tools
c = SUBAGENT_TYPES['computation']
assert c.tools == ['finish']
print('sub_agent configs OK')
"
```

预期输出: `sub_agent configs OK`

---

### Task 8: Agent 主循环 `agents/agentic/agent.py`

**Files:**
- Create: `agents/agentic/agent.py`

**Purpose:** 核心 ReAct 循环，整合所有模块。

- [ ] **Step 1: 写入 `agent.py`**

```python
"""ReAct Agent 主循环 — 工具驱动的 while 循环"""
import json
import time
import uuid
import asyncio
from pathlib import Path

from agents.agentic.types import AgentState, AgentResult, ToolCall, ToolResult
from agents.agentic.tools import ToolRegistry
from agents.agentic.skills import SkillManager
from agents.agentic.context import ContextManager
from agents.agentic.memory import MemoryManager


class Agent:
    """ReAct Agent — Claude Code 风格的工具驱动循环"""

    def __init__(self, model_config=None, model_size: str = "large",
                 language: str = "zh", max_iterations: int = 15,
                 enable_subagents: bool = True):
        self.model_config = model_config
        self.model_size = model_size
        self.language = language
        self.max_iterations = max_iterations
        self.enable_subagents = enable_subagents and model_size != "small"

        self.tools = ToolRegistry(model_size=model_size)
        self.skills = SkillManager(model_size=model_size)
        self.context = ContextManager(model_size=model_size)
        self.memory = MemoryManager()

    def run(self, query: str) -> AgentResult:
        """执行 Agent 主循环"""
        state = AgentState(query=query)

        # 0. 召回相关记忆
        recalled = self.memory.recall(query, top_k=3)
        state.memories_recalled = len(recalled)
        memory_context = ""
        if recalled:
            memory_context = "\n\n[相关历史记忆]\n" + "\n".join(
                f"- [{mem.type}] {mem.description[:200]}" for mem in recalled
            )

        # 1. 组装 System Prompt
        system_prompt = self.skills.build_system_prompt(query, self.language)
        if memory_context:
            system_prompt += memory_context

        # 2. 初始化消息列表
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        # 3. 工具 schema
        tool_schemas = self.tools.get_all_schemas()

        # 4. ReAct 循环
        no_tool_streak = 0
        while state.iterations < self.max_iterations and not state.finished:
            state.iterations += 1

            # 上下文压缩检查
            if self.context.should_compress(messages):
                messages, event = self.context.compress(messages)
                state.compression_events.append(event)

            # LLM 调用
            response = self._chat(messages, tool_schemas)

            if response.get("tool_calls"):
                no_tool_streak = 0
                tool_calls = response["tool_calls"]
                assistant_msg = {
                    "role": "assistant",
                    "content": response.get("content") or "",
                    "tool_calls": tool_calls,
                }
                messages.append(assistant_msg)

                for tc_data in tool_calls:
                    call = ToolCall(
                        id=tc_data["id"],
                        name=tc_data["function"]["name"],
                        args=json.loads(tc_data["function"]["arguments"]),
                        timestamp=time.time(),
                    )

                    if call.name == "finish":
                        # 生命周期工具：结束
                        result = ToolResult(
                            call_id=call.id, tool_name="finish",
                            success=True,
                            content=json.dumps(call.args, ensure_ascii=False),
                            confidence=call.args.get("confidence", 1.0),
                        )
                        state.final_answer = call.args.get("answer", "")
                        state.finished = True
                        state.add_tool_call(call, result)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": result.content,
                        })
                        break

                    elif call.name == "dispatch_subagent" and self.enable_subagents:
                        # 元工具：子代理
                        sub_result = self._run_subagent_sync(
                            task=call.args.get("task", ""),
                            agent_type=call.args.get("agent_type", "retrieval"),
                        )
                        result = ToolResult(
                            call_id=call.id, tool_name=call.name,
                            success=True,
                            content=json.dumps(sub_result, ensure_ascii=False),
                        )
                        state.subagents_dispatched += 1
                        state.add_tool_call(call, result)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": result.content,
                        })

                    elif call.name == "remember":
                        # 元工具：记忆
                        self.memory.save(
                            content=call.args["content"],
                            mem_type=call.args.get("type", "evidence"),
                            query=query,
                        )
                        result = ToolResult(
                            call_id=call.id, tool_name=call.name,
                            success=True, content="Memory saved.",
                        )
                        state.add_tool_call(call, result)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": "Memory saved.",
                        })

                    elif call.name == "plan_steps":
                        # 元工具：任务规划
                        steps = call.args.get("steps", [])
                        result = ToolResult(
                            call_id=call.id, tool_name=call.name,
                            success=True,
                            content=f"Plan created with {len(steps)} steps.",
                        )
                        state.add_tool_call(call, result)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": result.content,
                        })

                    else:
                        # 检索工具
                        result = self.tools.execute(call)
                        state.add_tool_call(call, result)

                        # 自动记忆：高置信度或发现矛盾
                        if result.confidence > 0.8 and not result.is_empty:
                            self.memory.save(
                                content=result.content[:500],
                                mem_type="evidence",
                                query=query,
                            )

                        messages.append({
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": result.content,
                        })

            else:
                # 纯文本响应（推理）
                content = response.get("content", "").strip()
                no_tool_streak += 1
                if content:
                    messages.append({"role": "assistant", "content": content})
                else:
                    no_tool_streak += 1  # 连空消息也算

            # 停止条件：连续 3 轮无工具调用
            if no_tool_streak >= 3:
                if not state.final_answer:
                    state.final_answer = self._force_answer(messages)
                state.finished = True

        # 5. 兜底：达到最大迭代仍未 finish
        if not state.final_answer:
            state.final_answer = self._force_answer(messages)

        state.skills_activated = self.skills.get_active_skill_names()

        return AgentResult(
            answer=state.final_answer,
            confidence=0.8,
            iterations=state.iterations,
            total_tool_calls=state.total_tool_calls,
            trace=state.trace,
            skills_used=state.skills_activated,
            subagent_count=state.subagents_dispatched,
            memories_used=state.memories_recalled,
            compression_events=[
                {"strategy": e.strategy, "before": e.before_tokens, "after": e.after_tokens}
                for e in state.compression_events
            ],
        )

    def _chat(self, messages: list[dict], tools: list[dict]) -> dict:
        """调用 LLM"""
        from llm.client import agent_chat_json, MODEL_CONFIGS
        if self.model_config is None:
            from config import AGENT_LLM_MODEL
            # 复用 MODEL_CONFIGS 中的配置
            import os
            base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:9097/v1")
            from llm.client import ModelConfig
            mc = ModelConfig(
                url=base_url,
                model_name=AGENT_LLM_MODEL,
                temperature=0.7,
                top_p=0.8,
            )
        else:
            mc = self.model_config

        # 使用 OpenAI SDK 的原生 tool calling
        from openai import OpenAI
        client = mc.get_client()

        try:
            response = client.chat.completions.create(
                model=mc.model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=mc.temperature,
                top_p=mc.top_p,
                max_tokens=2048,
            )
            choice = response.choices[0]
            result = {"content": choice.message.content or ""}
            if choice.message.tool_calls:
                result["tool_calls"] = [
                    {
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in choice.message.tool_calls
                ]
            return result
        except Exception:
            # Fallback: 使用项目的 agent_chat_json
            return agent_chat_json(messages, model_config=mc)

    def _run_subagent_sync(self, task: str, agent_type: str) -> dict:
        """同步运行子代理（简化版 — 在同步上下文中运行）"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 在已有的 event loop 中，创建新的
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._run_subagent(task, agent_type)
                    )
                    return future.result(timeout=60)
            else:
                return asyncio.run(self._run_subagent(task, agent_type))
        except RuntimeError:
            return asyncio.run(self._run_subagent(task, agent_type))

    async def _run_subagent(self, task: str, agent_type: str) -> dict:
        """异步运行子代理"""
        from agents.agentic.sub_agent import SUBAGENT_TYPES, SubAgentManager
        config = SUBAGENT_TYPES.get(agent_type, SUBAGENT_TYPES["retrieval"])

        def factory(cfg):
            return Agent(
                model_config=self.model_config,
                model_size=cfg.model_hint,
                language=self.language,
                max_iterations=cfg.max_iterations,
                enable_subagents=False,  # 子代理不再派发子代理
            )

        mgr = SubAgentManager(factory)
        return await mgr.dispatch(task=task, agent_type=agent_type)

    def _force_answer(self, messages: list[dict]) -> str:
        """强制生成答案（兜底）"""
        summary_prompt = ("Based on the evidence collected above, "
                          "provide a final answer to the original query.")
        try:
            from llm.client import agent_chat
            return agent_chat(summary_prompt, model_config=self.model_config)
        except Exception:
            return "无法生成答案（搜索未能收集到足够信息）"
```

- [ ] **Step 2: 验证 Agent 导入和基础结构**

```bash
cd C:/lib/codes/python_projects/core && python -c "
from agents.agentic.agent import Agent
agent = Agent(model_size='small', language='zh', max_iterations=2)
assert agent.tools is not None
assert agent.skills is not None
print('Agent initialized OK')
"
```

预期输出: `Agent initialized OK`

---

### Task 9: PipelineRouter `pipeline_router.py`

**Files:**
- Create: `pipeline_router.py`

**Purpose:** PEV 和 Agent 的统一入口，根据 mode 参数分发。

- [ ] **Step 1: 写入 `pipeline_router.py`**

```python
"""PipelineRouter — PEV 与 Agent 统一入口分发"""
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class PipelineRouter:
    """根据 mode 参数路由到 PEV 或 Agent pipeline"""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.default_mode = config.get("default_mode", "agent")

        # Agent 配置
        agent_model = config.get("agent_model")
        self._agent = None
        self._agent_model = agent_model
        self._agent_model_size = config.get("agent_model_size", "large")
        self._agent_language = config.get("agent_language", "zh")
        self._agent_max_iterations = config.get("agent_max_iterations", 15)
        self._agent_enable_subagents = config.get("agent_enable_subagents", True)

        # PEV 模型名
        self._pev_model_name = config.get("pev_model_name")

        # PEV 配置
        self._pev_enable_verifier = config.get("pev_enable_verifier", True)
        self._pev_enabled_tools = config.get("pev_enabled_tools")

    @property
    def agent(self):
        if self._agent is None:
            from llm.client import ModelConfig
            import os
            model_config = None
            if self._agent_model:
                url = os.environ.get("VLLM_BASE_URL", "http://localhost:9097/v1")
                model_config = ModelConfig(
                    url=url,
                    model_name=self._agent_model,
                    temperature=0.7,
                    top_p=0.8,
                )
            from agents.agentic.agent import Agent
            self._agent = Agent(
                model_config=model_config,
                model_size=self._agent_model_size,
                language=self._agent_language,
                max_iterations=self._agent_max_iterations,
                enable_subagents=self._agent_enable_subagents,
            )
        return self._agent

    def run(self, query: str, mode: str | None = None) -> dict:
        mode = mode or self.default_mode

        if mode == "pev":
            return self._run_pev(query)
        elif mode == "agent":
            return self._run_agent(query)
        elif mode == "compare":
            return self._run_both(query)
        else:
            raise ValueError(f"Unknown mode: {mode} (expected pev/agent/compare)")

    def _run_agent(self, query: str) -> dict:
        t0 = time.time()
        result = self.agent.run(query)
        latency = time.time() - t0

        return {
            "mode": "agent",
            "answer": result.answer,
            "iterations": result.iterations,
            "total_tool_calls": result.total_tool_calls,
            "trace": result.trace,
            "latency_ms": round(latency * 1000, 2),
            "metadata": {
                "skills_activated": result.skills_used,
                "subagents_dispatched": result.subagent_count,
                "memories_recalled": result.memories_used,
                "compression_events": result.compression_events,
            },
        }

    def _run_pev(self, query: str) -> dict:
        from agents.graph import build_graph, run_query

        t0 = time.time()
        graph = build_graph(
            enable_verifier=self._pev_enable_verifier,
            enabled_tools=self._pev_enabled_tools,
        )
        state = run_query(query, graph=graph)
        latency = time.time() - t0

        evidence_summary = []
        for e in state.get("evidence", []):
            evidence_summary.append({
                "step_id": e.get("step_id"),
                "sub_query": e.get("sub_query"),
                "tool": e.get("tool"),
                "num_results": len(e.get("results", [])),
            })

        return {
            "mode": "pev",
            "answer": state.get("final_answer", ""),
            "iterations": state.get("iteration_count", 0),
            "total_tool_calls": state.get("total_tool_calls", 0),
            "trace": state.get("trace", []),
            "latency_ms": round(latency * 1000, 2),
            "metadata": {
                "plan": state.get("plan", []),
                "verification": state.get("verification_result"),
                "evidence_summary": evidence_summary,
            },
        }

    def _run_both(self, query: str) -> dict:
        pev_result = self._run_pev(query)
        agent_result = self._run_agent(query)

        return {
            "mode": "compare",
            "query": query,
            "pev": pev_result,
            "agent": agent_result,
            "comparison": {
                "answers_differ": pev_result["answer"] != agent_result["answer"],
                "tool_calls_diff": (agent_result["total_tool_calls"] -
                                    pev_result["total_tool_calls"]),
                "iterations_diff": (agent_result["iterations"] -
                                    pev_result["iterations"]),
            },
        }
```

- [ ] **Step 2: 验证 Router 导入**

```bash
cd C:/lib/codes/python_projects/core && python -c "
from pipeline_router import PipelineRouter
r = PipelineRouter({'default_mode': 'agent'})
print('PipelineRouter initialized OK')
"
```

预期输出: `PipelineRouter initialized OK`

---

### Task 10: 配置扩展 `config.py` (修改)

**Files:**
- Modify: `config.py`

- [ ] **Step 1: 在 `config.py` 末尾追加 Agent 配置**

```python
# ── Agentic Agent 配置 ──
AGENT_CONFIG = {
    "default_mode": "agent",         # pev | agent | compare
    "agent_model": os.environ.get("AGENT_LLM_MODEL", "Qwen3-32B"),
    "agent_model_size": os.environ.get("AGENT_MODEL_SIZE", "large"),  # small | mid | large
    "agent_language": os.environ.get("PROMPT_LANG", "zh"),
    "agent_max_iterations": int(os.environ.get("AGENT_MAX_ITERATIONS", "15")),
    "agent_enable_subagents": os.environ.get("AGENT_ENABLE_SUBAGENTS", "true").lower() == "true",
    "pev_enable_verifier": True,
    "pev_enabled_tools": None,
}
```

- [ ] **Step 2: 验证配置导入**

```bash
cd C:/lib/codes/python_projects/core && python -c "
from config import AGENT_CONFIG
assert 'default_mode' in AGENT_CONFIG
assert AGENT_CONFIG['default_mode'] == 'agent'
print('AGENT_CONFIG OK')
"
```

预期输出: `AGENT_CONFIG OK`

---

### Task 11: 对比评估 `evaluation/compare.py`

**Files:**
- Create: `evaluation/compare.py`

**Purpose:** PEV vs Agent 的 A/B 对比评估，用同一个 judge 模型评估两边。

- [ ] **Step 1: 写入 `evaluation/compare.py`**

```python
"""PEV vs Agent A/B 对比评估"""
import time
from dataclasses import dataclass, field


@dataclass
class SingleEval:
    correctness: float     # 0-1
    faithfulness: float    # 0-1
    latency_ms: float
    tool_calls: int
    iterations: int


@dataclass
class ComparisonReport:
    total_queries: int = 0
    pev_scores: list[SingleEval] = field(default_factory=list)
    agent_scores: list[SingleEval] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)

    def avg_pev_correctness(self) -> float:
        if not self.pev_scores: return 0
        return sum(s.correctness for s in self.pev_scores) / len(self.pev_scores)

    def avg_agent_correctness(self) -> float:
        if not self.agent_scores: return 0
        return sum(s.correctness for s in self.agent_scores) / len(self.agent_scores)

    def avg_pev_faithfulness(self) -> float:
        if not self.pev_scores: return 0
        return sum(s.faithfulness for s in self.pev_scores) / len(self.pev_scores)

    def avg_agent_faithfulness(self) -> float:
        if not self.agent_scores: return 0
        return sum(s.faithfulness for s in self.agent_scores) / len(self.agent_scores)

    def avg_pev_latency_ms(self) -> float:
        if not self.pev_scores: return 0
        return sum(s.latency_ms for s in self.pev_scores) / len(self.pev_scores)

    def avg_agent_latency_ms(self) -> float:
        if not self.agent_scores: return 0
        return sum(s.latency_ms for s in self.agent_scores) / len(self.agent_scores)

    def winner_stats(self) -> dict:
        agent_wins = pev_wins = ties = 0
        for a, p in zip(self.agent_scores, self.pev_scores):
            if a.correctness > p.correctness:
                agent_wins += 1
            elif p.correctness > a.correctness:
                pev_wins += 1
            else:
                ties += 1
        return {"agent_wins": agent_wins, "pev_wins": pev_wins, "ties": ties}

    def summary(self) -> str:
        win = self.winner_stats()
        return (
            f"=== PEV vs Agent Comparison ({self.total_queries} queries) ===\n"
            f"Correctness — PEV: {self.avg_pev_correctness():.3f}  "
            f"Agent: {self.avg_agent_correctness():.3f}\n"
            f"Faithfulness — PEV: {self.avg_pev_faithfulness():.3f}  "
            f"Agent: {self.avg_agent_faithfulness():.3f}\n"
            f"Latency — PEV: {self.avg_pev_latency_ms():.0f}ms  "
            f"Agent: {self.avg_agent_latency_ms():.0f}ms\n"
            f"Wins — Agent: {win['agent_wins']}  PEV: {win['pev_wins']}  "
            f"Ties: {win['ties']}"
        )


class PipelineComparator:
    """PEV vs Agent 对比评估器"""

    def __init__(self, router, judge_model_config=None):
        self.router = router
        self.judge_config = judge_model_config

    def evaluate(self, queries: list[str],
                 ground_truths: list[str] | None = None) -> ComparisonReport:
        """运行对比评估

        Args:
            queries: 查询列表
            ground_truths: 标注答案列表（可选）
        """
        report = ComparisonReport(total_queries=len(queries))
        ground_truths = ground_truths or [""] * len(queries)

        for i, (query, gt) in enumerate(zip(queries, ground_truths)):
            # Run both
            pev = self.router._run_pev(query)
            agent = self.router._run_agent(query)

            # Judge correctness
            pev_correct = self._judge_correctness(pev["answer"], gt, query)
            agent_correct = self._judge_correctness(agent["answer"], gt, query)

            # Judge faithfulness
            pev_faith = self._judge_faithfulness(pev["answer"], pev["trace"], query)
            agent_faith = self._judge_faithfulness(agent["answer"], agent["trace"], query)

            report.pev_scores.append(SingleEval(
                correctness=pev_correct, faithfulness=pev_faith,
                latency_ms=pev["latency_ms"],
                tool_calls=pev["total_tool_calls"],
                iterations=pev["iterations"],
            ))
            report.agent_scores.append(SingleEval(
                correctness=agent_correct, faithfulness=agent_faith,
                latency_ms=agent["latency_ms"],
                tool_calls=agent["total_tool_calls"],
                iterations=agent["iterations"],
            ))
            report.queries.append(query)

        return report

    def _judge_correctness(self, answer: str, ground_truth: str, query: str) -> float:
        """用 judge 模型评分 correctness (0-1)"""
        if not ground_truth:
            return 0.5
        from llm.client import judge_chat_json
        prompt = (
            f"Score the answer's correctness against the ground truth.\n"
            f"Query: {query}\n"
            f"Ground truth: {ground_truth}\n"
            f"Answer: {answer}\n"
            f"Return JSON: {{\"score\": <0-1 float>, \"reason\": \"<brief>\"}}"
        )
        try:
            result = judge_chat_json(prompt)
            return float(result.get("score", 0.5))
        except Exception:
            return 0.5

    def _judge_faithfulness(self, answer: str, trace: list[dict], query: str) -> float:
        """用 judge 模型评分 faithfulness (0-1)"""
        if not trace:
            return 0.5
        from llm.client import judge_chat_json
        evidence_text = "\n".join(
            t.get("result_summary", "") for t in trace[-5:]
        )
        prompt = (
            f"Score whether the answer is fully supported by the evidence.\n"
            f"Query: {query}\n"
            f"Evidence: {evidence_text[:2000]}\n"
            f"Answer: {answer}\n"
            f"Return JSON: {{\"score\": <0-1 float>, \"reason\": \"<brief>\"}}"
        )
        try:
            result = judge_chat_json(prompt)
            return float(result.get("score", 0.5))
        except Exception:
            return 0.5
```

- [ ] **Step 2: 验证模块导入**

```bash
cd C:/lib/codes/python_projects/core && python -c "
from evaluation.compare import PipelineComparator, ComparisonReport
r = ComparisonReport(total_queries=5)
assert r.summary()
print('compare module OK')
"
```

预期输出: `compare模块 OK`

---

### Task 12: API 扩展 `api/server.py` (修改)

**Files:**
- Modify: `api/server.py`

- [ ] **Step 1: 修改 `api/server.py` — 新增路由和 mode 支持**

在现有 `QueryRequest` 中新增 `mode` 字段，在文件末尾新增 `/query/agent` 和 `/query/compare` 路由：

```python
# 替换原有的 QueryRequest
class QueryRequest(BaseModel):
    question: str
    verbose: bool = False
    mode: str = "agent"  # pev | agent | compare


# 在 /health 之后新增以下路由

@app.post("/query/agent", response_model=QueryResponse)
def query_agent_endpoint(req: QueryRequest):
    """显式 Agent 模式"""
    from pipeline_router import PipelineRouter
    from config import AGENT_CONFIG
    from llm.client import stats

    stats.reset()
    router = PipelineRouter(AGENT_CONFIG)
    result = router.run(query=req.question, mode="agent")

    return QueryResponse(
        answer=result["answer"],
        query_type="agent",
        iteration_count=result["iterations"],
        total_tool_calls=result["total_tool_calls"],
        evidence_summary=result.get("metadata", {}).get("evidence_summary", []),
        trace=result["trace"],
        latency=result["latency_ms"] / 1000,
    )


@app.post("/query/compare")
def query_compare_endpoint(req: QueryRequest):
    """对比模式：同时跑 PEV 和 Agent"""
    from pipeline_router import PipelineRouter
    from config import AGENT_CONFIG

    router = PipelineRouter(AGENT_CONFIG)
    return router.run(query=req.question, mode="compare")


@app.get("/health")
def health():
    from config import AGENT_CONFIG
    return {
        "status": "ok",
        "pev_available": True,
        "agent_available": True,
        "mode": AGENT_CONFIG.get("default_mode", "agent"),
        "agent_model_size": AGENT_CONFIG.get("agent_model_size", "large"),
    }
```

- [ ] **Step 2: 验证 API 导入**

```bash
cd C:/lib/codes/python_projects/core && python -c "
import sys
sys.path.insert(0, '.')
# 只验证语法正确，不启动服务器
import ast
with open('api/server.py', 'r', encoding='utf-8') as f:
    ast.parse(f.read())
print('api/server.py syntax OK')
"
```

预期输出: `api/server.py syntax OK`

---

### Task 13: 端到端集成测试

**Files:**
- Create: `tests/test_agentic.py`

- [ ] **Step 1: 写入集成测试**

```python
"""Agentic Agent 端到端集成测试"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAgentTypes:
    def test_agent_state_creation(self):
        from agents.agentic.types import AgentState
        state = AgentState(query="测试查询")
        assert state.query == "测试查询"
        assert state.iterations == 0
        assert state.final_answer == ""
        assert not state.finished

    def test_tool_meta_to_schema(self):
        from agents.agentic.types import ToolMeta
        meta = ToolMeta(
            name="test_tool",
            category="retrieval",
            description="测试工具",
            when_to_use="当需要测试时",
            when_not_to_use="不需要测试时",
            parameters={"type": "object", "properties": {}},
        )
        schema = meta.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test_tool"


class TestToolRegistry:
    def test_all_tools_registered(self):
        from agents.agentic.tools import ToolRegistry
        r = ToolRegistry(model_size="large")
        schemas = r.get_all_schemas()
        names = [s["function"]["name"] for s in schemas]
        assert "semantic_search" in names
        assert "keyword_search" in names
        assert "graph_search" in names
        assert "dispatch_subagent" in names
        assert "finish" in names

    def test_small_model_no_subagent(self):
        from agents.agentic.tools import ToolRegistry
        r = ToolRegistry(model_size="small")
        names = [s["function"]["name"] for s in r.get_all_schemas()]
        assert "dispatch_subagent" not in names


class TestSkills:
    def test_financial_skill_matches(self):
        from agents.agentic.skills import SkillManager
        sm = SkillManager(model_size="large")
        matched = sm.match("分析这家公司的营收增长率和净利润")
        assert len(matched) == 1
        assert matched[0].name == "financial-statement-analysis"

    def test_no_skill_match(self):
        from agents.agentic.skills import SkillManager
        sm = SkillManager(model_size="large")
        matched = sm.match("今天天气怎么样")
        assert len(matched) == 0

    def test_small_model_only_one_skill(self):
        from agents.agentic.skills import SkillManager
        sm = SkillManager(model_size="small")
        matched = sm.match("对比公司A和B的营收差异和风险状况")
        assert len(matched) == 1


class TestContext:
    def test_count_tokens(self):
        from agents.agentic.context import count_tokens
        msgs = [{"role": "system", "content": "You are helpful."}]
        n = count_tokens(msgs)
        assert n > 0

    def test_should_compress(self):
        from agents.agentic.context import ContextManager
        mgr = ContextManager(model_size="small", max_tokens=100)
        msgs = [{"role": "system", "content": "A" * 500}]
        assert mgr.should_compress(msgs)

    def test_small_aggressive(self):
        from agents.agentic.context import ContextManager
        mgr = ContextManager(model_size="small", max_tokens=2000)
        msgs = [
            {"role": "system", "content": "Base prompt"},
            {"role": "user", "content": "查询营收"},
            {"role": "assistant", "content": "searching..."},
            {"role": "tool", "content": "result: 123亿营收"},
            {"role": "assistant", "content": "searching more..."},
            {"role": "tool", "content": "result: 456亿营收"},
            {"role": "assistant", "content": "still searching..."},
            {"role": "tool", "content": "result: nothing new"},
        ]
        compressed, event = mgr.compress(msgs)
        assert event.strategy == "aggressive"
        assert len(compressed) <= len(msgs) + 1  # +1 for summary


class TestMemory:
    def test_save_and_recall(self, tmp_path):
        from agents.agentic.memory import MemoryManager
        mgr = MemoryManager(base_dir=str(tmp_path))
        mgr.save("2024年营收为123亿元", "evidence", "查询营收数据")
        recalled = mgr.recall("营收")
        assert len(recalled) == 1
        assert recalled[0].type == "evidence"

    def test_forget(self, tmp_path):
        from agents.agentic.memory import MemoryManager
        mgr = MemoryManager(base_dir=str(tmp_path))
        mem = mgr.save("测试数据", "evidence", "测试查询")
        mgr.forget(mem.name)
        recalled = mgr.recall("测试")
        assert len(recalled) == 0


class TestPipelineRouter:
    def test_router_init(self):
        from pipeline_router import PipelineRouter
        r = PipelineRouter({"default_mode": "agent"})
        assert r.default_mode == "agent"

    def test_unknown_mode_raises(self):
        from pipeline_router import PipelineRouter
        r = PipelineRouter({"default_mode": "agent"})
        with pytest.raises(ValueError, match="Unknown mode"):
            r.run("test query", mode="invalid")
```

- [ ] **Step 2: 运行测试**

```bash
cd C:/lib/codes/python_projects/core && python -m pytest tests/test_agentic.py -v --tb=short 2>&1 | head -60
```

预期: 大部分测试 PASS（依赖检索索引的测试需要实际数据）。

- [ ] **Step 3: 运行无检索依赖的快速验证**

```bash
cd C:/lib/codes/python_projects/core && python -m pytest tests/test_agentic.py -v -k "not ToolRegistry" --tb=short
```

预期: 所有测试 PASS（排除需要检索索引的）。

---

## 实施顺序

按依赖关系，建议实施顺序为：**Task 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13**

即严格按照文件依赖图从上到下实现。
