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

        agent_model = config.get("agent_model")
        self._agent = None
        self._agent_model = agent_model
        self._agent_model_size = config.get("agent_model_size", "large")
        self._agent_language = config.get("agent_language", "zh")
        self._agent_max_iterations = config.get("agent_max_iterations", 15)
        self._agent_enable_subagents = config.get("agent_enable_subagents", True)

        self._pev_model_name = config.get("pev_model_name")
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
        from config import LANGFUSE_ENABLED, AGENT_LLM_MODEL
        from monitoring.tracer import Tracer

        mode = mode or self.default_mode
        tracer = Tracer(enabled=LANGFUSE_ENABLED)
        tracer.start_trace(query=query, mode=mode, model=AGENT_LLM_MODEL)

        try:
            if mode == "pev":
                return self._run_pev(query, tracer)
            elif mode == "agent":
                return self._run_agent(query, tracer)
            elif mode == "compare":
                return self._run_both(query, tracer)
            else:
                raise ValueError(f"Unknown mode: {mode} (expected pev/agent/compare)")
        except Exception as e:
            tracer.end_trace(error=e)
            tracer.flush()
            raise

    def _run_agent(self, query: str, tracer=None) -> dict:
        t0 = time.time()
        result = self.agent.run(query, tracer=tracer)
        latency = time.time() - t0
        trace_id = result.trace_id

        output = {
            "mode": "agent",
            "answer": result.answer,
            "iterations": result.iterations,
            "total_tool_calls": result.total_tool_calls,
            "trace": result.trace,
            "latency_ms": round(latency * 1000, 2),
            "trace_id": trace_id,
            "metadata": {
                "skills_activated": result.skills_used,
                "subagents_dispatched": result.subagent_count,
                "memories_recalled": result.memories_used,
                "compression_events": [
                    {"strategy": e.layer, "before": e.before_tokens, "after": e.after_tokens}
                    for e in result.compression_events
                ],
                "no_tool_streak": result.no_tool_streak,
                "premature_finish": result.premature_finish,
                "plan_steps_count": result.plan_steps_count,
            },
        }

        if tracer:
            tracer.end_trace(result)
        return output

    def _run_pev(self, query: str, tracer=None) -> dict:
        from agents.pev.graph import build_graph, run_query
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
        if tracer:
            tracer.end_trace()
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

    def _run_both(self, query: str, tracer=None) -> dict:
        pev_result = self._run_pev(query, tracer=None)
        agent_result = self._run_agent(query, tracer=tracer)
        return {
            "mode": "compare",
            "query": query,
            "pev": pev_result,
            "agent": agent_result,
            "comparison": {
                "answers_differ": pev_result["answer"] != agent_result["answer"],
                "tool_calls_diff": agent_result["total_tool_calls"] - pev_result["total_tool_calls"],
                "iterations_diff": agent_result["iterations"] - pev_result["iterations"],
            },
        }
