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
        self._sample = True

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
            if LANGFUSE_SAMPLE_RATE < 1.0:
                self._sample = random.random() < LANGFUSE_SAMPLE_RATE
        return self._client

    # ── Trace 级别 ──

    def start_trace(self, query: str, mode: str = "agent", model: str = "",
                    metadata: dict | None = None) -> Optional[str]:
        """创建根 Trace，返回 trace_id"""
        if not self.enabled or not self._sample:
            return None
        try:
            self._trace = self.client.trace(
                name="agent-query",
                input={"query": query},
                metadata={"mode": mode, "model": model, **(metadata or {})},
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
                self._trace.update(output=None, level="ERROR", status_message=str(error))
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
        """为当前迭代创建 Span"""
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

    def log_generation(self, model: str, messages_count: int,
                       tool_calls_count: int, latency_ms: float,
                       tokens_in: int = 0, tokens_out: int = 0,
                       has_tool_calls: bool = False):
        """在活跃 Span 下创建 LLM Generation"""
        if not self.enabled or not self._iter_span:
            return
        try:
            gen = self._iter_span.generation(
                name="llm_call",
                model=model,
                input={"messages_count": messages_count},
                output={"tool_calls_count": tool_calls_count, "has_tool_calls": has_tool_calls},
                usage={"input": tokens_in, "output": tokens_out} if (tokens_in or tokens_out) else None,
                metadata={"latency_ms": round(latency_ms, 2)},
            )
            gen.end()
        except Exception as e:
            logger.warning(f"Langfuse log_generation failed: {e}")

    # ── 工具执行 Span ──

    def log_tool_call(self, tool_name: str, args: dict, result_content: str,
                      success: bool, confidence: float, latency_ms: float,
                      is_empty: bool = False):
        """为工具调用创建 Span"""
        if not self.enabled or not self._iter_span:
            return
        try:
            span = self._iter_span.span(
                name=f"tool:{tool_name}",
                input={"args": args},
                output={
                    "success": success, "confidence": confidence,
                    "is_empty": is_empty, "result_preview": result_content[:500],
                },
                metadata={"latency_ms": round(latency_ms, 2)},
            )
            span.end()
        except Exception as e:
            logger.warning(f"Langfuse log_tool_call failed: {e}")

    # ── 特殊事件 ──

    def log_compression(self, layer: str, tokens_before: int,
                        tokens_after: int, messages_removed: int = 0):
        """记录压缩事件"""
        if not self.enabled or not self._iter_span:
            return
        try:
            self._iter_span.event(
                name=f"compression_{layer}",
                input={"tokens_before": tokens_before, "tokens_after": tokens_after,
                       "messages_removed": messages_removed},
            )
        except Exception as e:
            logger.warning(f"Langfuse log_compression failed: {e}")

    def log_subagent(self, sub_type: str, task: str, iterations: int):
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
            span = self._trace.span(name="recall", output={"memories_recalled": count})
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
        """返回一个 no-op Tracer"""
        return Tracer(enabled=False)
