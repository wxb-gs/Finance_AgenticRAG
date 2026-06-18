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

    def test_tool_descriptions_has_text_to_sql(self):
        from agents.agentic.prompts import get_tool_descriptions
        t = get_tool_descriptions("zh")
        assert "text_to_sql" in t

    def test_tool_descriptions_has_mcp_section(self):
        from agents.agentic.prompts import get_tool_descriptions
        t = get_tool_descriptions("zh")
        assert "MCP" in t


class TestContext:
    def test_estimate_tokens(self):
        from agents.agentic.context import estimate_tokens
        msgs = [{"role": "system", "content": "You are helpful."}]
        n = estimate_tokens(msgs)
        assert n > 0

    def test_estimate_tokens_fallback(self):
        """字符估算 fallback：中文混合文本"""
        from agents.agentic.context import estimate_tokens
        msgs = [{"role": "user", "content": "你好世界 hello world"}]
        n = estimate_tokens(msgs)
        assert n > 0

    def test_init_defaults(self):
        from agents.agentic.context import ContextManager
        mgr = ContextManager()
        assert mgr.max_tokens == 32768
        assert mgr.snip_enabled is False
        assert mgr.micro_keep_recent == 5
        assert mgr.sm_min_tokens == 10000

    def test_init_custom(self):
        from agents.agentic.context import ContextManager
        mgr = ContextManager(
            max_tokens=16000,
            snip_enabled=True,
            micro_keep_recent=3,
            sm_min_tokens=5000,
        )
        assert mgr.max_tokens == 16000
        assert mgr.snip_enabled is True
        assert mgr.micro_keep_recent == 3

    def test_compress_trigger(self):
        from agents.agentic.context import ContextManager
        mgr = ContextManager(max_tokens=200, output_reserve=40, compress_threshold=0.5)
        # effective = 160, trigger = 80
        small = [{"role": "system", "content": "hi"}]
        assert not mgr.should_compress(small)
        # 用变长文本保证 token 数超过 trigger（BPE 会合并重复字符）
        large = [{"role": "system", "content": " ".join(f"token{i}" for i in range(200))}]
        assert mgr.should_compress(large)

    def test_microcompact_idle_trigger(self):
        from agents.agentic.context import ContextManager
        mgr = ContextManager(micro_trigger_threshold=999, micro_idle_minutes=0)
        mgr._last_active_at = 0
        msgs = [
            {"role": "system", "content": "Base"},
            {"role": "user", "content": "query"},
        ]
        for i in range(20):
            msgs.append({"role": "assistant", "content": f"step {i}",
                         "tool_calls": [{"id": f"tc{i}", "function": {"name": "keyword_search", "arguments": "{}"}}]})
            msgs.append({"role": "tool", "tool_call_id": f"tc{i}",
                         "content": f"result {i}" * 50})
        result, events = mgr.compress(msgs)
        assert any(e.layer == "microcompact" for e in events)

    def test_microcompact_count_trigger(self):
        from agents.agentic.context import ContextManager
        mgr = ContextManager(micro_trigger_threshold=3, micro_idle_minutes=999, micro_keep_recent=1)
        msgs = [
            {"role": "system", "content": "Base"},
            {"role": "user", "content": "query"},
        ]
        for i in range(10):
            msgs.append({"role": "assistant", "content": f"step {i}",
                         "tool_calls": [{"id": f"tc{i}", "function": {"name": "keyword_search", "arguments": "{}"}}]})
            msgs.append({"role": "tool", "tool_call_id": f"tc{i}",
                         "content": f"result {i}" * 50})
        result, events = mgr.compress(msgs)
        assert any(e.layer == "microcompact" for e in events)
        cleared = sum(1 for m in result if isinstance(m.get("content"), str)
                     and "Old tool result content cleared" in m["content"])
        assert cleared > 0

    def test_microcompact_skips_protected_tools(self):
        from agents.agentic.context import ContextManager
        mgr = ContextManager(micro_trigger_threshold=3, micro_idle_minutes=999, micro_keep_recent=1)
        msgs = [
            {"role": "system", "content": "Base"},
            {"role": "user", "content": "query"},
        ]
        # finish 调用不应被压缩
        msgs.append({"role": "assistant", "content": "",
                     "tool_calls": [{"id": "fin1", "function": {"name": "finish", "arguments": '{"answer":"done"}'}}]})
        msgs.append({"role": "tool", "tool_call_id": "fin1",
                     "content": '{"answer":"done"}'})
        for i in range(8):
            msgs.append({"role": "assistant", "content": f"step {i}",
                         "tool_calls": [{"id": f"tc{i}", "function": {"name": "keyword_search", "arguments": "{}"}}]})
            msgs.append({"role": "tool", "tool_call_id": f"tc{i}",
                         "content": f"result {i}" * 50})
        result, events = mgr.compress(msgs)
        finish_results = [m for m in result if m.get("tool_call_id") == "fin1"]
        assert len(finish_results) == 1
        assert "answer" in finish_results[0]["content"]

    def test_snip_disabled_by_default(self):
        from agents.agentic.context import ContextManager
        mgr = ContextManager()
        msgs = [
            {"role": "system", "content": "Base"},
            {"role": "user", "content": "query"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "t1", "function": {"name": "keyword_search", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "t1", "content": "未找到相关结果"},
        ]
        result, events = mgr.compress(msgs)
        assert len(result) == len(msgs)

    def test_snip_removes_empty_results(self):
        from agents.agentic.context import ContextManager
        mgr = ContextManager(snip_enabled=True, micro_trigger_threshold=999, micro_idle_minutes=999)
        msgs = [
            {"role": "system", "content": "Base"},
            {"role": "user", "content": "query"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "t1", "function": {"name": "keyword_search", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "t1", "content": "共 0 条结果"},
        ]
        result, events = mgr.compress(msgs)
        assert len(result) < len(msgs)
        assert any(e.layer == "snip" for e in events)

    def test_compression_pipeline_order(self):
        """验证管线执行顺序：snip → microcompact → (session_memory / ai_summary)"""
        from agents.agentic.context import ContextManager
        mgr = ContextManager(
            snip_enabled=True,
            micro_trigger_threshold=1,
            micro_idle_minutes=0,
            sm_min_tokens=1,
            sm_step_tokens=1,
            max_tokens=100,
            output_reserve=20,
            compress_threshold=0.1,
            circuit_breaker=5,
        )
        mgr._last_active_at = 0
        msgs = [
            {"role": "system", "content": "Base prompt " * 20},
            {"role": "user", "content": "test query"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "empty1", "function": {"name": "semantic_search", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "empty1", "content": "未找到"},
        ]
        for i in range(10):
            msgs.append({"role": "assistant", "content": f"step {i}",
                         "tool_calls": [{"id": f"tc{i}", "function": {"name": "keyword_search", "arguments": "{}"}}]})
            msgs.append({"role": "tool", "tool_call_id": f"tc{i}",
                         "content": f"result {i} " * 30})
        result, events = mgr.compress(msgs)

        layers = [e.layer for e in events]
        if "snip" in layers and "microcompact" in layers:
            snip_idx = layers.index("snip")
            mc_idx = layers.index("microcompact")
            assert snip_idx < mc_idx, f"Expected snip before microcompact, got {layers}"
        from agents.agentic.context import estimate_tokens
        assert estimate_tokens(result) < estimate_tokens(msgs)


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


class TestPlanIntegration:
    def test_plan_creation_and_status(self):
        from agents.agentic.types import Plan, PlanStep

        plan = Plan(
            query="test query",
            steps=[
                PlanStep(id="s1", description="step 1"),
                PlanStep(id="s2", description="step 2", depends_on=["s1"]),
                PlanStep(id="s3", description="step 3"),
            ],
        )

        ready = plan.ready_steps()
        assert len(ready) == 2
        assert {s.id for s in ready} == {"s1", "s3"}

        plan.mark_step("s1", "completed", "done step 1")
        ready = plan.ready_steps()
        assert len(ready) == 2
        assert {s.id for s in ready} == {"s2", "s3"}

        plan.mark_step("s2", "completed")
        plan.mark_step("s3", "completed")
        assert plan.all_done()

    def test_plan_format_status(self):
        from agents.agentic.types import Plan, PlanStep

        plan = Plan(
            query="test",
            steps=[
                PlanStep(id="s1", description="retrieve data",
                         status="completed", result_summary="found 3 items"),
                PlanStep(id="s2", description="calculate ROE",
                         depends_on=["s1"]),
            ],
        )
        text = plan.format_status()
        assert "current plan" in text.lower() or "当前计划" in text
        assert "s1" in text
        assert "s2" in text
        assert "completed" in text

    def test_agent_state_has_plan(self):
        from agents.agentic.types import AgentState, Plan, PlanStep

        state = AgentState(query="test")
        state.plan = Plan(
            query="test",
            steps=[PlanStep(id="s1", description="step 1")],
        )
        assert state.plan is not None
        assert len(state.plan.steps) == 1

    def test_plan_version_increment(self):
        from agents.agentic.types import Plan, PlanStep

        plan = Plan(query="test", steps=[PlanStep(id="s1", description="s1")])
        v1 = plan.version
        plan.version += 1
        assert plan.version == v1 + 1

    def test_plan_failed_step(self):
        from agents.agentic.types import Plan, PlanStep

        plan = Plan(query="test", steps=[
            PlanStep(id="s1", description="step 1"),
            PlanStep(id="s2", description="step 2"),
        ])
        plan.mark_step("s1", "failed")
        plan.mark_step("s2", "failed")
        assert plan.steps[0].status == "failed"
        assert plan.steps[1].status == "failed"
        # failed steps count as done for all_done
        assert plan.all_done()
