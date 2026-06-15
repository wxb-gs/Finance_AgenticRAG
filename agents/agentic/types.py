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
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    skills_used: list[str] = field(default_factory=list)
    subagent_count: int = 0
    memories_used: int = 0
    compression_events: list[CompressionEvent] = field(default_factory=list)
    finished: bool = False
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
    trace: list[dict[str, Any]]
    skills_used: list[str]
    subagent_count: int
    memories_used: int
    compression_events: list[CompressionEvent]
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
