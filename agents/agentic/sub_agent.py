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


class SubAgentManager:
    """子代理管理器 — 派发、并行执行、结果合并"""

    def __init__(self, agent_factory):
        self.agent_factory = agent_factory

    async def dispatch(self, task: str, agent_type: str = "retrieval",
                       background: bool = False) -> dict:
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
