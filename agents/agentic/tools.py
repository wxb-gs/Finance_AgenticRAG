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
        name="activate_skill",
        category="meta",
        description="Activate a domain skill to get specialized workflow guidance. Call when the query matches a skill's description.",
        when_to_use="查询涉及特定领域（财报分析、风险评估、对比分析）且需要专业工作流指引",
        when_not_to_use="简单查询不需要专业领域知识时",
        parameters={
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "Name of the skill to activate (e.g. financial-statement-analysis)",
                },
            },
            "required": ["skill_name"],
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

    def get_all_schemas(self) -> list[dict[str, Any]]:
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
        elif name in ("dispatch_subagent", "activate_skill", "remember", "plan_steps"):
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
        # graph_search accepts query as first param; entity name becomes the query
        results = graph_search(query=call.args["entity"])
        return ToolResult(
            call_id=call.id, tool_name=call.name, success=True,
            content=self._format_results(results),
            raw=results, is_empty=len(results) == 0,
        )

    def _exec_hybrid_search(self, call: ToolCall) -> ToolResult:
        from retrieval.hybrid_search import multi_tool_search
        from retrieval.semantic_search import semantic_search
        from retrieval.keyword_search import keyword_search

        tool_registry = {
            "semantic_search": semantic_search,
            "keyword_search": keyword_search,
        }
        tool_names = call.args.get("tools", ["keyword_search", "semantic_search"])
        results = multi_tool_search(query=call.args["query"],
                                     tool_names=tool_names,
                                     tool_registry=tool_registry)
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

    def _format_results(self, results: list[dict[str, Any]]) -> str:
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
