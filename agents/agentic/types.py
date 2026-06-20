"""Agentic Agent 共享类型定义"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Literal, Any

# Truncation limits for recording tool call results
_TOOL_CALL_CONTENT_MAX = 2000
_TRACE_RESULT_MAX = 500


@dataclass
class ToolCall:
    """一次工具调用"""
    id: str
    name: str
    args: dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolResult:
    """一次工具调用的结果"""
    call_id: str
    tool_name: str
    success: bool
    content: str                  # 序列化后的文本结果
    raw: list[dict[str, Any]] | None = None  # 原始结构化结果
    confidence: float = 1.0       # 结果置信度 (0-1)
    is_empty: bool = False        # 是否为空结果
    has_contradiction: bool = False


@dataclass
class CompressionEvent:
    """一次压缩事件"""
    layer: Literal["snip", "microcompact", "session_memory", "ai_summary"]
    before_tokens: int
    after_tokens: int
    messages_removed: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class CompactBoundary:
    """Session Memory 折叠边界元数据"""
    fold_count: int
    collapsed_msg_count: int
    pre_tokens: int
    post_tokens: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentState:
    """Agent 运行时状态"""
    query: str
    iterations: int = 0
    total_tool_calls: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    skills_used: list[str] = field(default_factory=list)
    subagent_count: int = 0
    memories_used: int = 0
    compression_events: list[CompressionEvent] = field(default_factory=list)
    finished: bool = False
    plan: Plan | None = None
    token_usage: int = 0

    def add_tool_call(self, call: ToolCall, result: ToolResult):
        self.tool_calls.append({
            "call_id": call.id,
            "name": call.name,
            "args": call.args,
            "success": result.success,
            "content": result.content[:_TOOL_CALL_CONTENT_MAX],
        })
        self.trace.append({
            "iteration": self.iterations,
            "tool_call": call.name,
            "args": call.args,
            "result_summary": result.content[:_TRACE_RESULT_MAX],
        })
        self.total_tool_calls += 1

    def to_result(self) -> dict[str, Any]:
        return {
            "final_answer": self.final_answer,
            "iterations": self.iterations,
            "total_tool_calls": self.total_tool_calls,
            "trace": self.trace,
            "skills_used": self.skills_used,
            "subagent_count": self.subagent_count,
            "memories_used": self.memories_used,
            "compression_events": [
                {"layer": e.layer, "before": e.before_tokens, "after": e.after_tokens}
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
    trace: list[dict[str, Any]]
    skills_used: list[str]
    subagent_count: int
    memories_used: int
    compression_events: list[CompressionEvent]
    trace_id: str = ""                      # Langfuse trace ID
    no_tool_streak: int = 0                 # 最长连续无工具轮次
    premature_finish: bool = False          # 是否过早终止（非 finish 工具触发）
    plan_steps_count: int = 0               # 计划步骤数
    evidence_summary: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ToolMeta:
    """工具元信息 — 用于 LLM 工具选择"""
    name: str
    category: Literal["retrieval", "meta", "lifecycle"]
    description: str
    when_to_use: str
    when_not_to_use: str
    parameters: dict[str, Any]
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
