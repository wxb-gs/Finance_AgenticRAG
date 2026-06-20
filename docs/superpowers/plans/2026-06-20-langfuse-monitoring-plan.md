# Langfuse Monitoring & Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate self-hosted Langfuse tracing into the production Agent loop, with evaluation metric reporting and bad-case-driven optimization suggestions.

**Architecture:** A new `monitoring/` module containing `Tracer` (Langfuse span lifecycle), `EvalReporter` (metrics → Langfuse scores), and `BadCaseRouter` (threshold-based classification → optimization suggestions). Tracer is injected at PipelineRouter and passed through Agent.run() as an optional parameter, defaulting to no-op when `LANGFUSE_ENABLED=false`.

**Tech Stack:** langfuse >= 3.0.0, existing OpenAI SDK, existing judge_chat for LLM-as-Judge scoring

---

### Task 1: Add Langfuse configuration and dependency

**Files:**
- Modify: `config.py:93-94` (append after MCP section)
- Modify: `requirements.txt:40` (append)

- [ ] **Step 1: Add Langfuse env vars to config.py**

Append after line 93 (`MCP_SERVERS = _json.loads(_mcp_override)`):

```python

# ── Langfuse 监控配置 ──
LANGFUSE_ENABLED = os.environ.get("LANGFUSE_ENABLED", "false").lower() == "true"
LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")
LANGFUSE_FLUSH_INTERVAL = int(os.environ.get("LANGFUSE_FLUSH_INTERVAL", "5"))
LANGFUSE_SAMPLE_RATE = float(os.environ.get("LANGFUSE_SAMPLE_RATE", "1.0"))
```

- [ ] **Step 2: Add langfuse to requirements.txt**

Append after line 40:

```
# Langfuse 监控追踪
langfuse>=3.0.0
```

- [ ] **Step 3: Commit**

```bash
git add config.py requirements.txt
git commit -m "feat: add Langfuse configuration and dependency"
```

---

### Task 2: Create Tracer class

**Files:**
- Create: `monitoring/__init__.py`
- Create: `monitoring/tracer.py`

- [ ] **Step 1: Create monitoring/__init__.py**

```python
"""Langfuse 监控与评测模块"""
from monitoring.tracer import Tracer

__all__ = ["Tracer"]
```

- [ ] **Step 2: Create monitoring/tracer.py**

```python
"""Tracer — Langfuse Trace/Span/Generation 生命周期管理

LANGFUSE_ENABLED=false 时所有方法退化为 no-op，零 SDK 开销。
"""

import time
import random
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)


class Tracer:
    """Langfuse 追踪器，包装 Trace → Span → Generation 层级结构"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._client = None
        self._trace = None
        self._trace_id: Optional[str] = None
        self._iter_span = None
        self._sample = True  # 采样标志

    # ── 懒加载 Langfuse client ──

    @property
    def client(self):
        if self._client is None and self.enabled:
            from langfuse import Langfuse
            from config import (
                LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY,
                LANGFUSE_FLUSH_INTERVAL, LANGFUSE_SAMPLE_RATE,
            )
            self._client = Langfuse(
                host=LANGFUSE_HOST,
                public_key=LANGFUSE_PUBLIC_KEY,
                secret_key=LANGFUSE_SECRET_KEY,
                flush_interval=LANGFUSE_FLUSH_INTERVAL,
            )
            # 采样判断
            if LANGFUSE_SAMPLE_RATE < 1.0:
                self._sample = random.random() < LANGFUSE_SAMPLE_RATE
        return self._client

    # ── Trace 级别 ──

    def start_trace(
        self, query: str, mode: str = "agent", model: str = "",
        metadata: dict | None = None,
    ) -> Optional[str]:
        """创建根 Trace，返回 trace_id"""
        if not self.enabled or not self._sample:
            return None
        try:
            self._trace = self.client.trace(
                name="agent-query",
                input={"query": query},
                metadata={
                    "mode": mode,
                    "model": model,
                    **(metadata or {}),
                },
            )
            self._trace_id = self._trace.id
            return self._trace_id
        except Exception as e:
            logger.warning(f"Langfuse start_trace failed: {e}")
            self.enabled = False
            return None

    def end_trace(self, result: Any = None, error: Exception | None = None):
        """结束根 Trace，附加输出或错误"""
        if not self.enabled or not self._trace:
            return
        try:
            if error:
                self._trace.update(
                    output=None,
                    level="ERROR",
                    status_message=str(error),
                )
            elif result is not None:
                output = {
                    "answer": getattr(result, "answer", ""),
                    "iterations": getattr(result, "iterations", 0),
                    "total_tool_calls": getattr(result, "total_tool_calls", 0),
                }
                self._trace.update(output=output)
        except Exception as e:
            logger.warning(f"Langfuse end_trace failed: {e}")

    # ── Iteration 级别 ──

    def start_iteration(self, iter_num: int):
        """为当前迭代创建 Span，作为当前活跃 Span"""
        if not self.enabled or not self._trace:
            return
        try:
            self._iter_span = self._trace.span(
                name=f"iteration_{iter_num}",
                input={"iteration": iter_num},
            )
        except Exception as e:
            logger.warning(f"Langfuse start_iteration failed: {e}")

    def end_iteration(self, metadata: dict | None = None):
        """结束当前迭代 Span"""
        if not self.enabled or not self._iter_span:
            return
        try:
            self._iter_span.update(metadata=metadata or {})
            self._iter_span.end()
            self._iter_span = None
        except Exception as e:
            logger.warning(f"Langfuse end_iteration failed: {e}")

    # ── LLM Generation ──

    def log_generation(
        self,
        model: str,
        messages_count: int,
        tool_calls_count: int,
        latency_ms: float,
        tokens_in: int = 0,
        tokens_out: int = 0,
        has_tool_calls: bool = False,
    ):
        """在活跃 Span 下创建 LLM Generation"""
        if not self.enabled or not self._iter_span:
            return
        try:
            gen = self._iter_span.generation(
                name="llm_call",
                model=model,
                input={"messages_count": messages_count},
                output={
                    "tool_calls_count": tool_calls_count,
                    "has_tool_calls": has_tool_calls,
                },
                usage={
                    "input": tokens_in,
                    "output": tokens_out,
                } if (tokens_in or tokens_out) else None,
                metadata={"latency_ms": round(latency_ms, 2)},
            )
            gen.end()
        except Exception as e:
            logger.warning(f"Langfuse log_generation failed: {e}")

    # ── 工具执行 Span ──

    def log_tool_call(
        self,
        tool_name: str,
        args: dict,
        result_content: str,
        success: bool,
        confidence: float,
        latency_ms: float,
        is_empty: bool = False,
    ):
        """为工具调用创建 Span"""
        if not self.enabled or not self._iter_span:
            return
        try:
            span = self._iter_span.span(
                name=f"tool:{tool_name}",
                input={"args": args},
                output={
                    "success": success,
                    "confidence": confidence,
                    "is_empty": is_empty,
                    "result_preview": result_content[:500],
                },
                metadata={"latency_ms": round(latency_ms, 2)},
            )
            span.end()
        except Exception as e:
            logger.warning(f"Langfuse log_tool_call failed: {e}")

    # ── 特殊事件 ──

    def log_compression(
        self, layer: str, tokens_before: int, tokens_after: int,
        messages_removed: int = 0,
    ):
        """记录压缩事件（作为迭代 Span 下的 event）"""
        if not self.enabled or not self._iter_span:
            return
        try:
            self._iter_span.event(
                name=f"compression_{layer}",
                input={
                    "tokens_before": tokens_before,
                    "tokens_after": tokens_after,
                    "messages_removed": messages_removed,
                },
            )
        except Exception as e:
            logger.warning(f"Langfuse log_compression failed: {e}")

    def log_subagent(
        self, sub_type: str, task: str, iterations: int,
    ):
        """记录子代理派发"""
        if not self.enabled or not self._iter_span:
            return
        try:
            span = self._iter_span.span(
                name=f"subagent:{sub_type}",
                input={"task": task[:500]},
                output={"iterations": iterations},
            )
            span.end()
        except Exception as e:
            logger.warning(f"Langfuse log_subagent failed: {e}")

    def log_recall(self, count: int):
        """记录记忆召回"""
        if not self.enabled or not self._trace:
            return
        try:
            span = self._trace.span(
                name="recall",
                output={"memories_recalled": count},
            )
            span.end()
        except Exception as e:
            logger.warning(f"Langfuse log_recall failed: {e}")

    # ── 评分 ──

    def score(self, name: str, value: float, metadata: dict | None = None):
        """为当前 Trace 添加评分"""
        if not self.enabled or not self._trace:
            return
        try:
            self._trace.score(name=name, value=value, comment=str(metadata or {}))
        except Exception as e:
            logger.warning(f"Langfuse score failed: {e}")

    def score_many(self, scores: dict[str, float]):
        """批量添加评分"""
        for name, value in scores.items():
            self.score(name, value)

    # ── 生命周期 ──

    def flush(self):
        """异步 flush 上报数据"""
        if not self.enabled or not self._client:
            return
        try:
            self._client.flush()
        except Exception as e:
            logger.warning(f"Langfuse flush failed: {e}")

    @staticmethod
    def noop() -> "Tracer":
        """返回一个 no-op Tracer，用于未启用 Langfuse 的场景"""
        return Tracer(enabled=False)
```

- [ ] **Step 3: Commit**

```bash
git add monitoring/__init__.py monitoring/tracer.py
git commit -m "feat: add Tracer class for Langfuse span lifecycle management"
```

---

### Task 3: Integrate Tracer into Agent.run() loop

**Files:**
- Modify: `agents/agentic/types.py:104-115` (AgentResult add diagnostic fields)
- Modify: `agents/agentic/agent.py:1-412` (3 instrumentation points)

- [ ] **Step 1: Extend AgentResult with diagnostic fields**

In `agents/agentic/types.py`, replace the `AgentResult` dataclass (lines 103-115):

```python
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
    trace_id: str = ""                      # Langfuse trace ID
    no_tool_streak: int = 0                 # 最长连续无工具轮次
    premature_finish: bool = False          # 是否过早终止（非 finish 工具触发）
    plan_steps_count: int = 0               # 计划步骤数
```

- [ ] **Step 2: Add tracer parameter to Agent.__init__ and Agent.run()**

In `agents/agentic/agent.py`, change `run()` signature and add tracer injection points.

First, change the `run` method signature (line 49):

```python
def run(self, query: str, tracer=None) -> AgentResult:
    """执行 Agent 主循环"""
    from monitoring.tracer import Tracer
    tracer = tracer or Tracer.noop()
    state = AgentState(query=query)
```

Insert after memory recall (after line 63 `state.memories_used = len(recalled)`):

```python
        # 埋点: 记忆召回
        tracer.log_recall(len(recalled))
```

Insert inside the main loop, before the LLM call (after line 106, before `current_messages = ...`):

```python
            # 埋点: 迭代开始
            tracer.start_iteration(state.iterations)
            t_iter_start = time.time()
```

Insert after `_chat()` call (after line 108, replace the simple `response = self._chat(...)` line):

```python
            t_llm_start = time.time()
            response = self._chat(current_messages, tool_schemas)
            t_llm_end = time.time()

            # 埋点: LLM Generation
            tracer.log_generation(
                model=self.model_config.model_name if self.model_config else "Qwen3-32B",
                messages_count=len(current_messages),
                tool_calls_count=len(response.get("tool_calls", [])),
                latency_ms=(t_llm_end - t_llm_start) * 1000,
                has_tool_calls=bool(response.get("tool_calls")),
            )
```

Add timing and tracer logging inside each tool dispatch branch. The pattern is: add `t0 = time.time()` at branch start, and tracer logging right after `messages.append(...)`. Apply to all 7 branches.

**Pattern for `finish` branch (line 128):**

```python
                    if call.name == "finish":
                        t0 = time.time()
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
                        tracer.log_tool_call(
                            tool_name=call.name, args=call.args,
                            result_content=result.content, success=result.success,
                            confidence=result.confidence,
                            latency_ms=(time.time() - t0) * 1000,
                            is_empty=result.is_empty,
                        )
                        break
```

**Pattern for `dispatch_subagent` branch (line 145):**

```python
                    elif call.name == "dispatch_subagent" and self.enable_subagents:
                        t0 = time.time()
                        step_id = call.args.get("step_id")
                        if step_id and state.plan:
                            state.plan.mark_step(step_id, "in_progress")
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
                        if step_id and state.plan:
                            state.plan.mark_step(
                                step_id, "completed",
                                sub_result.get("findings", "")[:200],
                            )
                        state.add_tool_call(call, result)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": result.content,
                        })
                        tracer.log_tool_call(
                            tool_name=call.name, args=call.args,
                            result_content=result.content, success=result.success,
                            confidence=result.confidence,
                            latency_ms=(time.time() - t0) * 1000,
                            is_empty=result.is_empty,
                        )
                        tracer.log_subagent(
                            sub_type=call.args.get("agent_type", "retrieval"),
                            task=call.args.get("task", ""),
                            iterations=sub_result.get("iterations", 0),
                        )
```

**Pattern for `else` (retrieval tools) branch (line 272):**

```python
                    else:
                        t0 = time.time()
                        result = self.tools.execute(call)
                        state.add_tool_call(call, result)
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
                        tracer.log_tool_call(
                            tool_name=call.name, args=call.args,
                            result_content=result.content, success=result.success,
                            confidence=result.confidence,
                            latency_ms=(time.time() - t0) * 1000,
                            is_empty=result.is_empty,
                        )
```

**Apply the same pattern** (`t0 = time.time()` at branch start, `tracer.log_tool_call(...)` after `messages.append(...)`) to the remaining 4 branches: `activate_skill`, `remember`, `plan_query`, `plan_update`.

Insert compression tracing around line 102 (inside the compression check block). Replace lines 99-103:

```python
            # 上下文压缩检查
            self.context.touch()
            if self.context.should_compress(messages):
                tokens_before = self.context.estimate_tokens(messages)
                messages, events = self.context.compress(messages)
                state.compression_events.extend(events)
                tokens_after = self.context.estimate_tokens(messages)
                # 埋点: 压缩事件
                for evt in events:
                    tracer.log_compression(
                        layer=evt.layer,
                        tokens_before=evt.before_tokens if hasattr(evt, 'before_tokens') else tokens_before,
                        tokens_after=evt.after_tokens if hasattr(evt, 'after_tokens') else tokens_after,
                        messages_removed=evt.messages_removed,
                    )
```

End of iteration, after the `no_tool_streak` check block (after line 302, before `# 5. 兜底`):

```python
            # 埋点: 迭代结束
            tracer.end_iteration(metadata={
                "no_tool_streak": no_tool_streak,
                "finished": state.finished,
            })
```

- [ ] **Step 3: Add diagnostic fields to AgentResult construction**

In `agents/agentic/agent.py`, modify the return statement (lines 310-320) to include new fields:

```python
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
            trace_id=getattr(tracer, '_trace_id', '') or '',
            no_tool_streak=no_tool_streak,
            premature_finish=(not state.final_answer.startswith("finish")
                              if hasattr(state, '_finished_by_finish') else False),
            plan_steps_count=len(state.plan.steps) if state.plan else 0,
        )
```

Actually, the premature_finish logic is more nuanced. Simplify:

```python
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
            trace_id=getattr(tracer, '_trace_id', '') or '',
            no_tool_streak=no_tool_streak,
            premature_finish=False,
            plan_steps_count=len(state.plan.steps) if state.plan else 0,
        )
```

- [ ] **Step 4: Mark premature_finish correctly**

After the loop end (line 302, in the `no_tool_streak >= 3` block), track that finish was via streak. We'll set `premature_finish = True` in the AgentResult. But better: just check at construction time. Change the no_tool_streak block (lines 299-302):

```python
            # 停止条件：连续 3 轮无工具调用
            if no_tool_streak >= 3:
                if not state.final_answer:
                    state.final_answer = self._force_answer(messages)
                state.finished = True
                state._finished_by_streak = True  # 标记为非 finish 工具终止
```

And update the return statement's premature_finish:

```python
            premature_finish=getattr(state, '_finished_by_streak', False),
```

- [ ] **Step 5: Commit**

```bash
git add agents/agentic/types.py agents/agentic/agent.py
git commit -m "feat: integrate Tracer instrumentation into Agent run loop"
```

---

### Task 4: Integrate Tracer into PipelineRouter

**Files:**
- Modify: `pipeline_router.py:1-129`

- [ ] **Step 1: Create tracer in PipelineRouter.run() and pass to Agent**

Replace `pipeline_router.py` entirely or make targeted edits:

Replace `run` method (lines 52-61):

```python
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
        finally:
            tracer.flush()
```

- [ ] **Step 2: Update _run_agent to accept and use tracer**

Replace `_run_agent` method (lines 63-83):

```python
    def _run_agent(self, query: str, tracer=None) -> dict:
        import time
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
```

- [ ] **Step 3: Update _run_pev to accept tracer (no-op for PEV)**

Replace `_run_pev` signature (line 85):

```python
    def _run_pev(self, query: str, tracer=None) -> dict:
```

And add end_trace at the end (before return):

```python
        if tracer:
            tracer.end_trace()
        return {
            ...
        }
```

- [ ] **Step 4: Update _run_both to propagate tracer**

Replace `_run_both` (lines 116-129):

```python
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
```

- [ ] **Step 5: Commit**

```bash
git add pipeline_router.py
git commit -m "feat: integrate Tracer into PipelineRouter for trace lifecycle"
```

---

### Task 5: Create EvalReporter

**Files:**
- Create: `monitoring/eval_reporter.py`
- Modify: `monitoring/__init__.py`

- [ ] **Step 1: Create monitoring/eval_reporter.py**

```python
"""EvalReporter — 评测指标计算与 Langfuse Score 上报"""

import logging
from collections import Counter

logger = logging.getLogger(__name__)


class EvalReporter:
    """为单次 Agent 执行计算评测指标并上报到 Langfuse Trace"""

    def __init__(self, tracer=None):
        self.tracer = tracer

    # ── 1. 工具选择 Accuracy ──

    def report_tool_selection(
        self,
        actual_tools: list[str],
        expected_tools: list[str],
    ) -> dict:
        """工具选择 Precision/Recall/F1

        Args:
            actual_tools: Agent 实际调用的工具名列表（去重）
            expected_tools: Ground truth 标注的工具集
        """
        actual_set = set(actual_tools)
        expected_set = set(expected_tools)

        tp = len(actual_set & expected_set)
        precision = tp / len(actual_set) if actual_set else 0.0
        recall = tp / len(expected_set) if expected_set else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        scores = {
            "tool_selection_precision": round(precision, 4),
            "tool_selection_recall": round(recall, 4),
            "tool_selection_f1": round(f1, 4),
        }

        if self.tracer:
            self.tracer.score_many(scores)

        return scores

    # ── 2. 参数质量（LLM-as-Judge）──

    def report_argument_quality(
        self,
        tool_calls: list[dict],
        query: str,
        judge_model: str = "gpt-oss-120b",
    ) -> dict:
        """对每次工具调用的参数质量打分 1-5

        Args:
            tool_calls: [{name, args, result_preview}, ...]
            query: 原始用户查询
            judge_model: 评分用的 Judge 模型名
        """
        if not tool_calls:
            scores = {"arg_quality_avg": 0.0, "arg_quality_min": 0.0}
            if self.tracer:
                self.tracer.score_many(scores)
            return scores

        ratings = []
        from llm.client import judge_chat_json

        for tc in tool_calls:
            name = tc.get("name", "unknown")
            args = tc.get("args", {})
            result_preview = tc.get("result_preview", "")

            prompt = f"""评估以下工具调用的参数质量。评分标准：
- 5: 参数精准覆盖了用户查询中的所有关键实体、约束条件
- 4: 参数覆盖了大部分关键要素，有轻微遗漏
- 3: 参数部分覆盖，缺少关键实体或时限约束
- 2: 参数与查询意图部分偏离，较大缺陷
- 1: 参数与查询无关或完全偏离

用户查询：{query}
工具名称：{name}
调用参数：{args}
返回结果摘要：{result_preview}

请返回JSON格式：{{"score": <1-5>, "reason": "<简要理由>"}}"""

            try:
                result = judge_chat_json(prompt, model=judge_model)
                if result and "score" in result:
                    ratings.append(min(5, max(1, int(result["score"]))))
                else:
                    ratings.append(3)  # 默认分数
            except Exception as e:
                logger.warning(f"Argument quality judge failed: {e}")
                ratings.append(3)

        avg_quality = sum(ratings) / len(ratings)
        min_quality = min(ratings)

        scores = {
            "arg_quality_avg": round(avg_quality, 2),
            "arg_quality_min": float(min_quality),
        }

        if self.tracer:
            self.tracer.score_many(scores)

        return scores

    # ── 3. 调用效率（自动）──

    def report_call_efficiency(
        self,
        tool_calls: list[dict],
        expected_tool_count: int = 0,
    ) -> dict:
        """冗余率 / 重复率 / 步数效率

        Args:
            tool_calls: [{"name": ..., "is_empty": bool, "args": {...}}, ...]
            expected_tool_count: ground truth 期望的工具类型数
        """
        total = len(tool_calls)
        if total == 0:
            scores = {
                "redundancy_rate": 0.0,
                "repetition_count": 0,
                "step_efficiency": 0.0,
            }
            if self.tracer:
                self.tracer.score_many(scores)
            return scores

        # 冗余率：空结果 或 success=false 的调用占比
        empty_count = sum(
            1 for tc in tool_calls
            if tc.get("is_empty") or not tc.get("success", True)
        )
        redundancy_rate = empty_count / total

        # 重复率：相同 (name, query_string) 的重复调用
        call_sigs = [
            (tc.get("name", ""), str(tc.get("args", {})))
            for tc in tool_calls
        ]
        sig_counts = Counter(call_sigs)
        repetition_count = sum(c - 1 for c in sig_counts.values())

        # 步数效率：理论最小 / 实际
        min_steps = max(1, expected_tool_count + 1)  # +1 for finish
        step_efficiency = min_steps / total if total > 0 else 1.0

        scores = {
            "redundancy_rate": round(redundancy_rate, 4),
            "repetition_count": repetition_count,
            "step_efficiency": round(min(1.0, step_efficiency), 4),
        }

        if self.tracer:
            self.tracer.score_many(scores)

        return scores

    # ── 4. 规划合理性 ──

    def report_planning(
        self,
        plan_steps_desc: list[str],
        ground_truth_hops: list[str],
    ) -> dict:
        """计算计划步骤对 ground truth hop 的覆盖率

        Args:
            plan_steps_desc: Agent plan 中各步骤的 description 文本列表
            ground_truth_hops: Ground truth 的 hop 描述列表
        """
        if not ground_truth_hops:
            scores = {"plan_hop_precision": 0.0, "plan_hop_recall": 0.0}
            if self.tracer:
                self.tracer.score_many(scores)
            return scores

        if not plan_steps_desc:
            scores = {"plan_hop_precision": 0.0, "plan_hop_recall": 0.0}
            if self.tracer:
                self.tracer.score_many(scores)
            return scores

        # 简单 token 重叠匹配：每个 plan step 匹配最佳 hop
        import jieba

        def _tokenize(text: str) -> set:
            return set(jieba.lcut(text.lower()))

        plan_tokens_list = [_tokenize(s) for s in plan_steps_desc]
        hop_tokens_list = [_tokenize(h) for h in ground_truth_hops]

        # 每个 hop 找最佳匹配 plan step
        matched_hops = 0
        used_plans = set()
        for hop_tokens in hop_tokens_list:
            best_overlap = 0
            best_idx = -1
            for i, plan_tokens in enumerate(plan_tokens_list):
                if i in used_plans:
                    continue
                overlap = len(hop_tokens & plan_tokens) / max(1, len(hop_tokens))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = i
            if best_overlap >= 0.2:  # 至少 20% token 重叠
                matched_hops += 1
                used_plans.add(best_idx)

        plan_recall = matched_hops / len(ground_truth_hops)
        plan_precision = matched_hops / len(plan_steps_desc) if plan_steps_desc else 0.0

        scores = {
            "plan_hop_precision": round(plan_precision, 4),
            "plan_hop_recall": round(plan_recall, 4),
        }

        if self.tracer:
            self.tracer.score_many(scores)

        return scores
```

- [ ] **Step 2: Update monitoring/__init__.py**

```python
"""Langfuse 监控与评测模块"""
from monitoring.tracer import Tracer
from monitoring.eval_reporter import EvalReporter

__all__ = ["Tracer", "EvalReporter"]
```

- [ ] **Step 3: Commit**

```bash
git add monitoring/eval_reporter.py monitoring/__init__.py
git commit -m "feat: add EvalReporter for metric computation and Langfuse score reporting"
```

---

### Task 6: Create BadCaseRouter

**Files:**
- Create: `monitoring/badcase_router.py`
- Modify: `monitoring/__init__.py`

- [ ] **Step 1: Create monitoring/badcase_router.py**

```python
"""BadCaseRouter — Bad Case 识别 → 分类 → Prompt/Schema 优化建议"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BadCase:
    """单个 Bad Case 记录"""
    query: str
    category: str          # tool_selection_low / arg_quality_poor / ...
    scores: dict            # 触发该分类的分数快照
    trace_id: str = ""
    optimize_target: str = ""  # "prompt" | "tool_schema"


# 分类规则定义
BADCASE_RULES = [
    {
        "name": "tool_selection_low",
        "category": "工具选择错误",
        "condition": lambda s: s.get("tool_selection_f1", 1.0) < 0.5,
        "target": "tool_schema",
        "description": "Agent 调用的工具类型与预期严重不一致",
    },
    {
        "name": "arg_quality_poor",
        "category": "参数质量差",
        "condition": lambda s: s.get("arg_quality_avg", 5.0) < 3.0,
        "target": "prompt",
        "description": "工具调用参数缺少关键实体或约束条件",
    },
    {
        "name": "high_redundancy",
        "category": "无效调用过多",
        "condition": lambda s: s.get("redundancy_rate", 0.0) > 0.4,
        "target": "tool_schema",
        "description": "超过 40% 的工具调用返回空结果或无效应答",
    },
    {
        "name": "step_inefficient",
        "category": "步数效率低",
        "condition": lambda s: s.get("step_efficiency", 1.0) < 0.5,
        "target": "prompt",
        "description": "Agent 用了远超必要数量的工具调用才完成任务",
    },
    {
        "name": "early_finish",
        "category": "过早终止",
        "condition": lambda s: s.get("premature_finish", False),
        "target": "prompt",
        "description": "Agent 在未收集足够证据时提前结束循环",
    },
    {
        "name": "plan_mismatch",
        "category": "规划遗漏",
        "condition": lambda s: s.get("plan_hop_recall", 1.0) < 0.5,
        "target": "prompt",
        "description": "Agent 的计划未能覆盖 ground truth 的关键 hop",
    },
]

# 触发建议的最小同类累积数
MIN_ACCUMULATE = 5


class BadCaseRouter:
    """根据评测分数自动分类 Bad Case 并生成优化建议"""

    def __init__(self):
        self._accumulator: dict[str, list[BadCase]] = {}

    def classify(self, query: str, scores: dict, trace_id: str = "") -> list[BadCase]:
        """根据分数快照自动分类，返回匹配的 BadCase 列表"""
        matched = []
        for rule in BADCASE_RULES:
            try:
                if rule["condition"](scores):
                    bc = BadCase(
                        query=query,
                        category=rule["name"],
                        scores=scores,
                        trace_id=trace_id,
                        optimize_target=rule["target"],
                    )
                    matched.append(bc)
                    self._accumulate(bc)
            except Exception as e:
                logger.warning(f"BadCase rule {rule['name']} failed: {e}")
        return matched

    def _accumulate(self, bc: BadCase):
        """累积同类 Bad Case"""
        key = bc.category
        if key not in self._accumulator:
            self._accumulator[key] = []
        self._accumulator[key].append(bc)

    def get_pending_suggestions(self) -> list[dict]:
        """返回所有达到阈值的优化建议"""
        suggestions = []
        for category, cases in self._accumulator.items():
            if len(cases) >= MIN_ACCUMULATE:
                rule = next((r for r in BADCASE_RULES if r["name"] == category), None)
                if rule:
                    # 生成具体优化建议
                    suggestion = self._build_suggestion(rule, cases)
                    suggestions.append(suggestion)
                    # 重置，避免重复触发
                    self._accumulator[category] = []
        return suggestions

    def _build_suggestion(self, rule: dict, cases: list[BadCase]) -> dict:
        """基于累积的 Bad Case 生成优化建议"""
        query_samples = [c.query[:120] for c in cases[:5]]
        score_summary = self._aggregate_scores(cases)

        if rule["target"] == "prompt":
            suggestion_text = self._build_prompt_suggestion(
                rule, query_samples, score_summary
            )
        else:
            suggestion_text = self._build_schema_suggestion(
                rule, query_samples, score_summary
            )

        return {
            "category": rule["name"],
            "target": rule["target"],
            "case_count": len(cases),
            "query_samples": query_samples,
            "score_summary": score_summary,
            "suggestion": suggestion_text,
        }

    def _aggregate_scores(self, cases: list[BadCase]) -> dict:
        """聚合多个 Bad Case 的分数统计"""
        if not cases:
            return {}
        keys = cases[0].scores.keys()
        result = {}
        for k in keys:
            values = [c.scores[k] for c in cases if k in c.scores]
            if values:
                result[k] = {
                    "min": round(min(values), 4),
                    "max": round(max(values), 4),
                    "avg": round(sum(values) / len(values), 4),
                }
        return result

    def _build_prompt_suggestion(
        self, rule: dict, samples: list[str], scores: dict,
    ) -> str:
        """生成 Prompt 优化建议"""
        suggestions = {
            "arg_quality_poor": (
                "建议在 System Prompt 中增加工具调用参数规范：\n"
                "1. 要求每个检索 query 必须包含所有关键实体名称和时效约束\n"
                "2. 增加 few-shot 示例展示高质量参数格式\n"
                "3. 在工具描述中增加参数反例说明"
            ),
            "step_inefficient": (
                "建议在 System Prompt 中强化工具调用效率意识：\n"
                "1. 要求优先使用 hybrid_search 覆盖多维度，减少单工具多次调用\n"
                "2. 增加\"调用前先思考是否已有足够信息\"的提醒\n"
                "3. 设置每轮工具调用上限提示"
            ),
            "early_finish": (
                "建议在 System Prompt 中强化完备性检查：\n"
                "1. 要求 finish 前必须确认所有 plan 步骤已完成\n"
                "2. 增加\"在信息不足时继续搜索而非提前终止\"的规则\n"
                "3. 提高连续无工具调用的终止阈值（当前 3）或移除该规则"
            ),
            "plan_mismatch": (
                "建议优化 Plan 生成 Prompt：\n"
                "1. 在 plan_query 工具描述中增加 hop 分解示例\n"
                "2. 要求 plan 步骤显式标注每个步骤的目标信息类型\n"
                "3. 增加多跳问题 plan 模板"
            ),
        }
        base = suggestions.get(rule["name"], "建议审查并优化相关 Prompt 模板。")
        return (
            f"## Bad Case 类型：{rule['category']}\n"
            f"涉及 {len(samples)} 个案例，例如：{samples[0] if samples else 'N/A'}\n\n"
            f"{base}\n\n"
            f"修改文件：agents/agentic/prompts.py"
        )

    def _build_schema_suggestion(
        self, rule: dict, samples: list[str], scores: dict,
    ) -> str:
        """生成 Tool Schema 优化建议"""
        suggestions = {
            "tool_selection_low": (
                "建议调整工具描述和优先级：\n"
                "1. 检查 when_to_use / when_not_to_use 描述是否与实际使用场景匹配\n"
                "2. 考虑调整工具 priority 值以影响模型选择偏好\n"
                "3. 在工具描述中增加典型查询示例"
            ),
            "high_redundancy": (
                "建议优化工具 Schema 减少无效调用：\n"
                "1. 检查工具 description 中是否过度承诺了不存在的功能\n"
                "2. 为参数增加更严格的约束（如 min_length、enum 等）\n"
                "3. 在 when_not_to_use 中增加会导致空结果的典型场景"
            ),
        }
        base = suggestions.get(rule["name"], "建议审查并优化相关工具 Schema 定义。")
        return (
            f"## Bad Case 类型：{rule['category']}\n"
            f"涉及 {len(samples)} 个案例，例如：{samples[0] if samples else 'N/A'}\n\n"
            f"{base}\n\n"
            f"修改文件：agents/agentic/tools.py (_RETRIEVAL_TOOL_DEFS)"
        )

    def reset(self):
        """清空累积的 Bad Case"""
        self._accumulator.clear()
```

- [ ] **Step 2: Update monitoring/__init__.py**

```python
"""Langfuse 监控与评测模块"""
from monitoring.tracer import Tracer
from monitoring.eval_reporter import EvalReporter
from monitoring.badcase_router import BadCaseRouter

__all__ = ["Tracer", "EvalReporter", "BadCaseRouter"]
```

- [ ] **Step 3: Commit**

```bash
git add monitoring/badcase_router.py monitoring/__init__.py
git commit -m "feat: add BadCaseRouter for automated bad case classification and optimization suggestions"
```

---

### Task 7: Integration test — verify end-to-end trace flow

**Files:**
- Create: `tests/test_monitoring.py`

- [ ] **Step 1: Write tests for Tracer no-op mode**

```python
"""监控模块单元测试"""
import pytest
from unittest.mock import patch, MagicMock


class TestTracer:
    """Tracer 单元测试"""

    def test_noop_mode_returns_without_error(self):
        from monitoring.tracer import Tracer
        tracer = Tracer.noop()
        assert tracer.enabled is False
        # 所有方法应无错误返回
        assert tracer.start_trace("test query") is None
        tracer.end_trace()
        tracer.start_iteration(1)
        tracer.end_iteration()
        tracer.log_generation("test-model", 10, 2, 150.0)
        tracer.log_tool_call("search", {"q": "test"}, "result", True, 0.9, 50.0)
        tracer.log_compression("session_memory", 10000, 5000)
        tracer.log_subagent("retrieval", "find X", 3)
        tracer.log_recall(2)
        tracer.score("test_score", 0.8)
        tracer.flush()  # no-op

    def test_enabled_tracer_creates_trace(self):
        from monitoring.tracer import Tracer
        tracer = Tracer(enabled=True)
        # 不连接真实 Langfuse，mock client
        tracer._client = MagicMock()
        tracer._sample = True
        trace_id = tracer.start_trace("测试查询", mode="agent", model="Qwen3-32B")
        tracer._client.trace.assert_called_once()
        assert trace_id is not None

    def test_disabled_tracer_skips_all(self):
        from monitoring.tracer import Tracer
        tracer = Tracer(enabled=False)
        tracer._client = MagicMock()
        assert tracer.start_trace("q") is None
        tracer._client.trace.assert_not_called()


class TestEvalReporter:
    """EvalReporter 单元测试"""

    def test_tool_selection_perfect(self):
        from monitoring.eval_reporter import EvalReporter
        reporter = EvalReporter(tracer=None)
        scores = reporter.report_tool_selection(
            actual_tools=["semantic_search", "keyword_search", "read_chunk"],
            expected_tools=["semantic_search", "keyword_search", "read_chunk"],
        )
        assert scores["tool_selection_precision"] == 1.0
        assert scores["tool_selection_recall"] == 1.0
        assert scores["tool_selection_f1"] == 1.0

    def test_tool_selection_miss(self):
        from monitoring.eval_reporter import EvalReporter
        reporter = EvalReporter(tracer=None)
        scores = reporter.report_tool_selection(
            actual_tools=["semantic_search"],
            expected_tools=["semantic_search", "keyword_search"],
        )
        assert scores["tool_selection_precision"] == 1.0
        assert scores["tool_selection_recall"] == 0.5

    def test_tool_selection_empty(self):
        from monitoring.eval_reporter import EvalReporter
        reporter = EvalReporter(tracer=None)
        scores = reporter.report_tool_selection(
            actual_tools=[],
            expected_tools=["semantic_search"],
        )
        assert scores["tool_selection_precision"] == 0.0
        assert scores["tool_selection_recall"] == 0.0

    def test_call_efficiency_perfect(self):
        from monitoring.eval_reporter import EvalReporter
        reporter = EvalReporter(tracer=None)
        scores = reporter.report_call_efficiency(
            tool_calls=[
                {"name": "semantic_search", "args": {"q": "x"}, "success": True, "is_empty": False},
                {"name": "read_chunk", "args": {"id": "1"}, "success": True, "is_empty": False},
            ],
            expected_tool_count=2,
        )
        assert scores["redundancy_rate"] == 0.0
        assert scores["repetition_count"] == 0

    def test_call_efficiency_with_redundancy(self):
        from monitoring.eval_reporter import EvalReporter
        reporter = EvalReporter(tracer=None)
        scores = reporter.report_call_efficiency(
            tool_calls=[
                {"name": "search", "args": {"q": "x"}, "success": True, "is_empty": True},
                {"name": "search", "args": {"q": "x"}, "success": True, "is_empty": False},
                {"name": "search", "args": {"q": "y"}, "success": True, "is_empty": False},
            ],
            expected_tool_count=1,
        )
        assert scores["redundancy_rate"] == 1 / 3
        assert scores["repetition_count"] == 1  # search+x 重复 1 次

    def test_planning_perfect_match(self):
        from monitoring.eval_reporter import EvalReporter
        reporter = EvalReporter(tracer=None)
        scores = reporter.report_planning(
            plan_steps_desc=["查找腾讯 2024 年营收", "分析营收增长原因"],
            ground_truth_hops=["腾讯 2024 年营收", "营收增长原因分析"],
        )
        # 每个 hop 应能匹配到对应的 plan step
        assert scores["plan_hop_recall"] == 1.0


class TestBadCaseRouter:
    """BadCaseRouter 单元测试"""

    def test_classify_tool_selection_low(self):
        from monitoring.badcase_router import BadCaseRouter
        router = BadCaseRouter()
        bad = router.classify(
            query="test query",
            scores={"tool_selection_f1": 0.3},
        )
        assert len(bad) == 1
        assert bad[0].category == "tool_selection_low"
        assert bad[0].optimize_target == "tool_schema"

    def test_classify_no_match_when_scores_good(self):
        from monitoring.badcase_router import BadCaseRouter
        router = BadCaseRouter()
        bad = router.classify(
            query="test query",
            scores={
                "tool_selection_f1": 0.9,
                "arg_quality_avg": 4.5,
                "redundancy_rate": 0.1,
                "step_efficiency": 0.8,
                "premature_finish": False,
                "plan_hop_recall": 0.9,
            },
        )
        assert len(bad) == 0

    def test_accumulate_and_trigger_suggestion(self):
        from monitoring.badcase_router import BadCaseRouter
        router = BadCaseRouter()
        # 累积 5 个同类 Bad Case
        for i in range(5):
            router.classify(
                query=f"query_{i}",
                scores={"tool_selection_f1": 0.3},
            )
        suggestions = router.get_pending_suggestions()
        assert len(suggestions) == 1
        assert suggestions[0]["case_count"] == 5
        assert "tool_schema" in suggestions[0]["target"]

    def test_no_suggestion_below_threshold(self):
        from monitoring.badcase_router import BadCaseRouter
        router = BadCaseRouter()
        # 只累积 3 个，不到 5 个阈值
        for i in range(3):
            router.classify(
                query=f"query_{i}",
                scores={"tool_selection_f1": 0.3},
            )
        suggestions = router.get_pending_suggestions()
        assert len(suggestions) == 0
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
pytest tests/test_monitoring.py -v
```

Expected: all 12 tests PASS

- [ ] **Step 3: Run existing agent tests to verify no regression**

```bash
pytest tests/test_agentic.py -v
```

Expected: all existing tests PASS (37 tests)

- [ ] **Step 4: Commit**

```bash
git add tests/test_monitoring.py
git commit -m "test: add monitoring module unit tests"
```

---

## Implementation Order

Tasks must be executed in order: 1 → 2 → 3 → 4 → 5 → 6 → 7

Each task depends on the previous one. Within Task 3, the types.py change must precede the agent.py change.

## Verification Checklist

After all tasks:
- [ ] `LANGFUSE_ENABLED=false` — Agent runs normally with zero Langfuse overhead
- [ ] `LANGFUSE_ENABLED=true` — traces appear in self-hosted Langfuse dashboard
- [ ] Span hierarchy correct: Trace → iteration_X → llm_call + tool:*
- [ ] EvalReporter scores appear on traces as Langfuse Scores
- [ ] BadCaseRouter correctly classifies based on score thresholds
- [ ] All 37 existing agent tests still pass
- [ ] All 12 new monitoring tests pass
