"""Agentic Agent 端到端集成测试"""
import sys
import os
import tempfile

import pytest

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

    def test_tool_call_timestamp(self):
        from agents.agentic.types import ToolCall
        import time
        tc = ToolCall(id="1", name="test", args={})
        assert tc.timestamp > 0

    def test_agent_state_add_tool_call(self):
        from agents.agentic.types import AgentState, ToolCall, ToolResult
        state = AgentState(query="test")
        tc = ToolCall(id="1", name="semantic_search", args={"query": "test"})
        tr = ToolResult(call_id="1", tool_name="semantic_search", success=True, content="result")
        state.add_tool_call(tc, tr)
        assert state.total_tool_calls == 1
        assert len(state.tool_calls) == 1
        assert len(state.trace) == 1

    def test_agent_state_uses_corrected_names(self):
        from agents.agentic.types import AgentState
        state = AgentState(query="test")
        assert hasattr(state, "skills_used")
        assert hasattr(state, "subagent_count")
        assert hasattr(state, "memories_used")


class TestToolRegistry:
    def test_all_tools_registered(self):
        from agents.agentic.tools import ToolRegistry
        r = ToolRegistry(model_size="large")
        schemas = r.get_all_schemas()
        names = [s["function"]["name"] for s in schemas]
        assert "semantic_search" in names
        assert "keyword_search" in names
        assert "dispatch_subagent" in names
        assert "finish" in names

    def test_small_model_no_subagent(self):
        from agents.agentic.tools import ToolRegistry
        r = ToolRegistry(model_size="small")
        names = [s["function"]["name"] for s in r.get_all_schemas()]
        assert "dispatch_subagent" not in names

    def test_tool_categories(self):
        from agents.agentic.tools import ToolRegistry
        r = ToolRegistry()
        assert r.is_retrieval_tool("semantic_search")
        assert r.is_meta_tool("dispatch_subagent")
        assert r.is_lifecycle_tool("finish")


class TestSkills:
    def test_all_skills_loaded(self):
        from agents.agentic.skills import SkillManager
        sm = SkillManager(model_size="large")
        assert len(sm.registry) == 3
        assert "financial-statement-analysis" in sm.registry
        assert "risk-assessment" in sm.registry
        assert "multi-hop-comparison" in sm.registry

    def test_skill_has_description(self):
        from agents.agentic.skills import SkillManager
        sm = SkillManager(model_size="large")
        skill = sm.registry["financial-statement-analysis"]
        assert len(skill.description) > 20
        assert "财报" in skill.description

    def test_skill_listing_text(self):
        from agents.agentic.skills import SkillManager
        sm = SkillManager(model_size="large")
        text = sm.get_listing_text()
        assert "financial-statement-analysis" in text
        assert "risk-assessment" in text
        assert "activate_skill" in text

    def test_activate_valid_skill(self):
        from agents.agentic.skills import SkillManager
        sm = SkillManager(model_size="large")
        skill = sm.activate("financial-statement-analysis")
        assert skill is not None
        assert skill.name == "financial-statement-analysis"
        assert len(skill.content) > 50
        assert sm.get_active_skill_names() == ["financial-statement-analysis"]

    def test_activate_invalid_skill(self):
        from agents.agentic.skills import SkillManager
        sm = SkillManager(model_size="large")
        skill = sm.activate("nonexistent-skill")
        assert skill is None
        assert sm.get_active_skill_names() == []

    def test_build_system_prompt(self):
        from agents.agentic.skills import SkillManager
        sm = SkillManager(model_size="large")
        prompt = sm.build_system_prompt("zh")
        # 技能列表在 system prompt 中
        assert "可用技能" in prompt
        assert "financial-statement-analysis" in prompt
        # 描述中包含模型需要判断的信息
        assert "深度解析" in prompt
        assert "activate_skill" in prompt

    def test_fuzzy_activate(self):
        from agents.agentic.skills import SkillManager
        sm = SkillManager(model_size="large")
        skill = sm.activate("financial")
        assert skill is not None
        assert "financial" in skill.name


class TestPrompts:
    def test_small_prompt_no_subagent(self):
        from agents.agentic.prompts import get_system_prompt
        p = get_system_prompt("small", "zh")
        assert "dispatch_subagent" not in p

    def test_large_prompt_has_subagent(self):
        from agents.agentic.prompts import get_system_prompt
        p = get_system_prompt("large", "zh")
        assert "dispatch_subagent" in p

    def test_invalid_model_size(self):
        from agents.agentic.prompts import get_system_prompt
        with pytest.raises(ValueError):
            get_system_prompt("tiny", "zh")

    def test_invalid_language(self):
        from agents.agentic.prompts import get_system_prompt
        with pytest.raises(ValueError):
            get_system_prompt("small", "fr")

    def test_tool_descriptions_zh(self):
        from agents.agentic.prompts import get_tool_descriptions
        t = get_tool_descriptions("zh")
        assert "semantic_search" in t
        assert "keyword_search" in t

    def test_tool_descriptions_en(self):
        from agents.agentic.prompts import get_tool_descriptions
        t = get_tool_descriptions("en")
        assert "semantic_search" in t


class TestContext:
    def test_count_tokens(self):
        from agents.agentic.context import count_tokens
        msgs = [{"role": "system", "content": "You are helpful."}]
        n = count_tokens(msgs)
        assert n > 0

    def test_should_compress(self):
        from agents.agentic.context import ContextManager
        mgr = ContextManager(model_size="small", max_tokens=30)
        msgs = [{"role": "system", "content": "A" * 500}]
        assert mgr.should_compress(msgs)

    def test_small_aggressive(self):
        from agents.agentic.context import ContextManager, count_tokens
        mgr = ContextManager(model_size="small", max_tokens=2000)
        msgs = [
            {"role": "system", "content": "Base prompt"},
            {"role": "user", "content": "查询营收"},
            {"role": "assistant", "content": "searching..."},
            {"role": "tool", "content": "[1] chunk_id=001 score=0.9\n    result: 123亿营收"},
            {"role": "assistant", "content": "searching more..."},
            {"role": "tool", "content": "[1] chunk_id=002 score=0.8\n    result: 456亿营收"},
            {"role": "assistant", "content": "still searching..."},
            {"role": "tool", "content": "[1] chunk_id=003 score=0.7\n    result: nothing new"},
        ]
        compressed, event = mgr.compress(msgs)
        assert event.strategy == "aggressive"
        assert len(compressed) <= len(msgs) + 1

    def test_mid_strategy(self):
        from agents.agentic.context import ContextManager
        mgr = ContextManager(model_size="mid", max_tokens=2000)
        msgs = [
            {"role": "system", "content": "Base prompt"},
            {"role": "user", "content": "query"},
        ]
        for i in range(8):
            msgs.append({"role": "assistant", "content": f"step {i}"})
            msgs.append({"role": "tool", "content": f"[1] chunk_id=00{i} score=0.9\n    result"})
        compressed, event = mgr.compress(msgs)
        assert event.strategy == "summarize_old"


class TestMemory:
    def test_save_and_recall(self):
        from agents.agentic.memory import MemoryManager
        with tempfile.TemporaryDirectory() as tmp:
            mgr = MemoryManager(base_dir=tmp)
            mem = mgr.save("2024年营收为123亿元", "evidence", "查询营收数据")
            recalled = mgr.recall("营收")
            assert len(recalled) == 1
            assert recalled[0].type == "evidence"

    def test_forget(self):
        from agents.agentic.memory import MemoryManager
        with tempfile.TemporaryDirectory() as tmp:
            mgr = MemoryManager(base_dir=tmp)
            mem = mgr.save("测试数据", "evidence", "测试查询")
            mgr.forget(mem.name)
            recalled = mgr.recall("测试")
            assert len(recalled) == 0

    def test_multiple_types(self):
        from agents.agentic.memory import MemoryManager
        with tempfile.TemporaryDirectory() as tmp:
            mgr = MemoryManager(base_dir=tmp)
            mgr.save("contradiction data found in source A vs B", "contradiction", "check contradiction")
            mgr.save("gap missing Q3 cash flow data", "gap", "check gap")
            recalled = mgr.recall("contradiction gap")
            assert len(recalled) == 2


class TestSubAgent:
    def test_configs_exist(self):
        from agents.agentic.sub_agent import SUBAGENT_TYPES
        assert "retrieval" in SUBAGENT_TYPES
        assert "comparison" in SUBAGENT_TYPES
        assert "computation" in SUBAGENT_TYPES

    def test_retrieval_config(self):
        from agents.agentic.sub_agent import SUBAGENT_TYPES
        r = SUBAGENT_TYPES["retrieval"]
        assert r.max_iterations == 5
        assert "semantic_search" in r.tools

    def test_computation_finish_only(self):
        from agents.agentic.sub_agent import SUBAGENT_TYPES
        c = SUBAGENT_TYPES["computation"]
        assert c.tools == ["finish"]
        assert c.max_iterations == 3


class TestAgent:
    def test_agent_init(self):
        from agents.agentic.agent import Agent
        agent = Agent(model_size="small", language="zh", max_iterations=2)
        assert agent.tools is not None
        assert agent.skills is not None
        assert agent.context is not None
        assert agent.memory is not None

    def test_small_model_no_subagents(self):
        from agents.agentic.agent import Agent
        agent = Agent(model_size="small")
        assert not agent.enable_subagents

    def test_large_model_has_subagents(self):
        from agents.agentic.agent import Agent
        agent = Agent(model_size="large")
        assert agent.enable_subagents


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


class TestConfig:
    def test_agent_config_exists(self):
        from config import AGENT_CONFIG
        assert "default_mode" in AGENT_CONFIG
        assert AGENT_CONFIG["default_mode"] == "agent"
        assert "agent_model_size" in AGENT_CONFIG


class TestMCPToolIntegration:
    def test_mcp_tools_in_schema(self):
        from agents.agentic.tools import ToolRegistry
        r = ToolRegistry(model_size="large")
        r._mcp["mock_server__ping"] = {
            "name": "ping",
            "description": "Test ping tool",
            "inputSchema": {"type": "object", "properties": {}},
        }
        schemas = r.get_all_schemas()
        mcp_names = [s["function"]["name"] for s in schemas if s["function"]["name"].startswith("mcp__")]
        assert "mcp__mock_server__ping" in mcp_names

    def test_text_to_sql_tool_registered(self):
        from agents.agentic.tools import ToolRegistry
        r = ToolRegistry(model_size="large")
        schemas = r.get_all_schemas()
        names = [s["function"]["name"] for s in schemas]
        assert "text_to_sql" in names

    def test_text_to_sql_is_retrieval_tool(self):
        from agents.agentic.tools import ToolRegistry
        r = ToolRegistry()
        assert r.is_retrieval_tool("text_to_sql")

    def test_text_to_sql_available_in_small_model(self):
        from agents.agentic.tools import ToolRegistry
        r = ToolRegistry(model_size="small")
        names = [s["function"]["name"] for s in r.get_all_schemas()]
        assert "text_to_sql" in names

    def test_discover_mcp_empty_config(self):
        from agents.agentic.tools import ToolRegistry
        r = ToolRegistry()
        import asyncio
        asyncio.run(r.discover_mcp(servers=[]))
        assert len(r._mcp_clients) == 0
