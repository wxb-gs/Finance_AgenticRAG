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
        self._mcp_initialized = False

    def run(self, query: str) -> AgentResult:
        """执行 Agent 主循环"""
        state = AgentState(query=query)

        # 0. 初始化 MCP 连接（首次运行）
        if not self._mcp_initialized:
            from config import MCP_SERVERS
            if MCP_SERVERS:
                import asyncio
                asyncio.run(self.tools.discover_mcp(MCP_SERVERS))
            self._mcp_initialized = True

        # 1. 召回相关记忆
        recalled = self.memory.recall(query, top_k=3)
        state.memories_used = len(recalled)
        memory_context = ""
        if recalled:
            memory_context = "\n\n[相关历史记忆]\n" + "\n".join(
                f"- [{mem.type}] {mem.description[:200]}" for mem in recalled
            )

        # 1. 组装 System Prompt
        system_prompt = self.skills.build_system_prompt(self.language)
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
                        sub_result = self._run_subagent_sync(
                            task=call.args.get("task", ""),
                            agent_type=call.args.get("agent_type", "retrieval"),
                        )
                        result = ToolResult(
                            call_id=call.id, tool_name=call.name,
                            success=True,
                            content=json.dumps(sub_result, ensure_ascii=False),
                        )
                        state.subagent_count += 1
                        state.add_tool_call(call, result)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": result.content,
                        })

                    elif call.name == "activate_skill":
                        skill = self.skills.activate(call.args.get("skill_name", ""))
                        result = ToolResult(
                            call_id=call.id, tool_name=call.name,
                            success=skill is not None,
                            content=(
                                f"技能 '{skill.name}' 已激活：\n\n{skill.content}"
                                if skill
                                else f"未找到技能：{call.args.get('skill_name')}"
                            ),
                        )
                        state.add_tool_call(call, result)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": result.content,
                        })

                    elif call.name == "remember":
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

                        # 自动记忆：高置信度
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

            # 停止条件：连续 3 轮无工具调用
            if no_tool_streak >= 3:
                if not state.final_answer:
                    state.final_answer = self._force_answer(messages)
                state.finished = True

        # 5. 兜底：达到最大迭代仍未 finish
        if not state.final_answer:
            state.final_answer = self._force_answer(messages)

        state.skills_used = self.skills.get_active_skill_names()

        return AgentResult(
            answer=state.final_answer,
            confidence=0.8,
            iterations=state.iterations,
            total_tool_calls=state.total_tool_calls,
            trace=state.trace,
            skills_used=state.skills_used,
            subagent_count=state.subagent_count,
            memories_used=state.memories_used,
            compression_events=state.compression_events,
        )

    def _chat(self, messages: list[dict], tools: list[dict]) -> dict:
        """调用 LLM，使用 OpenAI SDK 原生 tool calling"""
        if self.model_config is None:
            import os
            from config import AGENT_LLM_MODEL
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
            from llm.client import agent_chat_json
            return agent_chat_json(messages, model_config=mc)

    def _run_subagent_sync(self, task: str, agent_type: str) -> dict:
        """同步运行子代理（处理 event loop 冲突）"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
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
                enable_subagents=False,
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
