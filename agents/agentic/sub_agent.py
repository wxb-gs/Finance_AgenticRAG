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
