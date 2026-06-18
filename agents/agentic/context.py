"""上下文压缩 — Claude Code 风格四层渐进式管线

Layer 1: Snip          — 移除低价值整轮 turn（零 API 调用）
Layer 2: MicroCompact   — 清理旧 tool_result 为占位符（缓存感知）
Layer 3: Session Memory — 增量 LLM 摘要折叠旧消息
Layer 4: AI Summary     — 全量 9 段结构化摘要 + 截断兜底
"""
import re
import time
import tiktoken
from typing import Literal, Optional

from agents.agentic.types import CompressionEvent


def estimate_tokens(messages: list[dict], model: str = "gpt-4") -> int:
    """估算消息列表的 token 数，tiktoken 不可用时退化为字符估算"""
    try:
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        total = 0
        for msg in messages:
            for key in ("content", "tool_calls", "tool_call_id"):
                if key in msg and msg[key]:
                    if isinstance(msg[key], str):
                        total += len(enc.encode(msg[key]))
                    elif isinstance(msg[key], list):
                        for item in msg[key]:
                            if isinstance(item, dict) and "function" in item:
                                total += len(enc.encode(str(item["function"])))
            if "role" in msg:
                total += 4
        return total
    except Exception:
        total = 0
        for msg in messages:
            for key in ("content", "tool_calls", "tool_call_id"):
                if key in msg and msg[key]:
                    if isinstance(msg[key], str):
                        text = msg[key]
                        chinese = sum(1 for c in text if '一' <= c <= '鿿')
                        other = len(text) - chinese
                        total += (chinese // 2) + (other // 4)
                    elif isinstance(msg[key], list):
                        for item in msg[key]:
                            if isinstance(item, dict) and "function" in item:
                                total += len(str(item["function"])) // 4
            if "role" in msg:
                total += 1
        return max(total, 1)


# Backward compatibility alias
count_tokens = estimate_tokens


class ContextManager:
    """四层渐进式上下文压缩管线"""

    # Layer 2 可压缩工具白名单
    _COMPACTABLE_TOOLS = frozenset({
        "semantic_search", "keyword_search", "graph_search",
        "hybrid_search", "read_chunk", "text_to_sql",
    })

    _CLEARED_MARKER = "[Old tool result content cleared]"

    def __init__(
        self,
        max_tokens: int = 32768,
        compress_threshold: float = 0.75,
        output_reserve: int = 4096,
        snip_enabled: bool = False,
        micro_keep_recent: int = 5,
        micro_trigger_threshold: int = 10,
        micro_idle_minutes: int = 60,
        sm_min_tokens: int = 10000,
        sm_step_tokens: int = 5000,
        sm_keep_recent: int = 8,
        summary_max_tokens: int = 4000,
        circuit_breaker: int = 3,
    ):
        self.max_tokens = max_tokens
        self.compress_threshold = compress_threshold
        self.output_reserve = output_reserve
        self.snip_enabled = snip_enabled
        self.micro_keep_recent = micro_keep_recent
        self.micro_trigger_threshold = micro_trigger_threshold
        self.micro_idle_minutes = micro_idle_minutes
        self.sm_min_tokens = sm_min_tokens
        self.sm_step_tokens = sm_step_tokens
        self.sm_keep_recent = sm_keep_recent
        self.summary_max_tokens = summary_max_tokens
        self.circuit_breaker_limit = circuit_breaker

        # 内部状态
        self._last_active_at: float = time.time()
        self._fold_count: int = 0
        self._session_memory_parts: list[str] = []
        self._circuit_breaker_count: int = 0
        self._collapsed_msg_count: int = 0

    @property
    def _effective_tokens(self) -> int:
        return self.max_tokens - self.output_reserve

    @property
    def _compress_trigger(self) -> int:
        return int(self._effective_tokens * self.compress_threshold)

    # ═══════════════════════════════════════════════════════════
    # 顶层接口
    # ═══════════════════════════════════════════════════════════

    def touch(self):
        """更新活跃时间（每次 LLM 请求前调用）"""
        self._last_active_at = time.time()

    def should_compress(self, messages: list[dict]) -> bool:
        """检查 token 是否超过 Layer 3/4 触发线"""
        return estimate_tokens(messages) > self._compress_trigger

    def compress(self, messages: list[dict]) -> tuple[list[dict], list[CompressionEvent]]:
        """执行完整压缩管线

        Returns:
            (压缩后消息列表, 压缩事件列表)
        """
        events: list[CompressionEvent] = []
        current = list(messages)

        # Layer 1: Snip
        if self.snip_enabled:
            current, event = self._snip(current)
            if event:
                events.append(event)

        # Layer 2: MicroCompact
        current, event = self._microcompact(current)
        if event:
            events.append(event)

        # Layer 3: Session Memory
        if self.should_compress(current):
            current, event = self._session_memory_compact(current)
            if event:
                events.append(event)

        # Layer 4: AI Summary
        if self.should_compress(current):
            current, event = self._ai_summary(current)
            if event:
                events.append(event)

        return current, events

    # ═══════════════════════════════════════════════════════════
    # Layer 1: Snip — 移除整轮低价值 turn
    # ═══════════════════════════════════════════════════════════

    def _snip(self, messages: list[dict]) -> tuple[list[dict], Optional[CompressionEvent]]:
        """移除空结果/低价值 turn"""
        if len(messages) <= 3:
            return messages, None

        before = estimate_tokens(messages)
        result = self._do_snip(messages)

        if len(result) == len(messages):
            return messages, None

        after = estimate_tokens(result)
        return result, CompressionEvent(
            layer="snip",
            before_tokens=before,
            after_tokens=after,
            messages_removed=len(messages) - len(result),
            timestamp=time.time(),
        )

    def _do_snip(self, messages: list[dict]) -> list[dict]:
        """识别并移除空结果 turn (assistant+tool 对)"""
        _protected_tools = {"finish", "dispatch_subagent", "plan_query",
                           "plan_update", "remember"}
        dead: set[int] = set()

        for i, msg in enumerate(messages):
            if i in dead:
                continue
            if msg.get("role") != "assistant":
                continue
            if not msg.get("tool_calls"):
                continue

            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                if fn.get("name", "") in _protected_tools:
                    continue
                tc_id = tc.get("id", "")
                # 找对应的 tool_result
                for j in range(i + 1, min(i + 10, len(messages))):
                    if messages[j].get("role") == "tool" and \
                       messages[j].get("tool_call_id") == tc_id:
                        if self._is_empty_result(messages[j]):
                            dead.add(i)
                            dead.add(j)
                        break

        if not dead:
            return messages
        return [m for idx, m in enumerate(messages) if idx not in dead]

    def _is_empty_result(self, tool_msg: dict) -> bool:
        """判断 tool_result 是否空/低价值"""
        content = tool_msg.get("content", "")
        if not content or not content.strip():
            return True
        if any(kw in content for kw in ("未找到", "No results", "共 0 条",
                                          "0 results", "未找到技能")):
            return True
        return False

    # ═══════════════════════════════════════════════════════════
    # Layer 2: MicroCompact — 缓存感知 tool_result 清理
    # ═══════════════════════════════════════════════════════════

    def _microcompact(self, messages: list[dict]) -> tuple[list[dict], Optional[CompressionEvent]]:
        """清理旧 tool_result 为占位符"""
        compactable = self._find_compactable_indices(messages)

        if len(compactable) <= self.micro_keep_recent:
            return messages, None

        if not self._should_microcompact_trigger(compactable):
            return messages, None

        before = estimate_tokens(messages)
        result = self._do_microcompact(messages, compactable)
        after = estimate_tokens(result)

        if after == before:
            return messages, None

        return result, CompressionEvent(
            layer="microcompact",
            before_tokens=before,
            after_tokens=after,
            messages_removed=0,
            timestamp=time.time(),
        )

    def _find_compactable_indices(self, messages: list[dict]) -> list[int]:
        """找到所有可压缩且尚未清理的 tool 消息索引"""
        indices = []
        for i, msg in enumerate(messages):
            if msg.get("role") != "tool":
                continue
            if msg.get("content") == self._CLEARED_MARKER:
                continue
            tool_name = self._resolve_tool_name(messages, i)
            if tool_name and tool_name in self._COMPACTABLE_TOOLS:
                indices.append(i)
        return indices

    def _should_microcompact_trigger(self, compactable_indices: list[int]) -> bool:
        """检查 MicroCompact 两种触发条件"""
        # 条件1: 计数 — 可压缩结果数超阈值
        if len(compactable_indices) > self.micro_trigger_threshold:
            return True
        # 条件2: 时间 — 空闲超时，缓存已冷
        idle = time.time() - self._last_active_at
        if idle > self.micro_idle_minutes * 60:
            return True
        return False

    def _do_microcompact(self, messages: list[dict],
                         compactable_indices: list[int]) -> list[dict]:
        """保留最近 K 条完整内容，其余替换为占位符"""
        to_clear = compactable_indices[:-self.micro_keep_recent]
        if not to_clear:
            return messages

        result = list(messages)
        for idx in to_clear:
            result[idx] = {**result[idx], "content": self._CLEARED_MARKER}
        return result

    def _resolve_tool_name(self, messages: list[dict], tool_idx: int) -> Optional[str]:
        """从 tool 消息回溯对应的工具名"""
        tc_id = messages[tool_idx].get("tool_call_id", "")
        if not tc_id:
            return None
        for i in range(tool_idx - 1, max(tool_idx - 20, -1), -1):
            msg = messages[i]
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    if tc.get("id") == tc_id:
                        return tc.get("function", {}).get("name")
        return None

    # ═══════════════════════════════════════════════════════════
    # Layer 3: Session Memory — 增量式上下文折叠
    # ═══════════════════════════════════════════════════════════

    def _session_memory_compact(self, messages: list[dict]) -> tuple[list[dict], Optional[CompressionEvent]]:
        """增量折叠：每次折叠约 sm_step_tokens 的最早未折叠消息"""
        current = estimate_tokens(messages)

        # 未达首次触发线
        if current < self.sm_min_tokens:
            return messages, None

        # 未达下一次增量触发线
        next_trigger = self.sm_min_tokens + self._fold_count * self.sm_step_tokens
        if current < next_trigger:
            return messages, None

        system_msgs = [m for m in messages if m.get("role") == "system"]
        conversation = [m for m in messages if m.get("role") != "system"]

        if len(conversation) <= self.sm_keep_recent:
            return messages, None

        # 确定折叠范围
        fold_start = self._collapsed_msg_count
        fold_end = self._compute_fold_end(conversation, fold_start)

        if fold_end <= fold_start:
            return messages, None

        fold_msgs = conversation[fold_start:fold_end]

        # 生成增量摘要
        incremental = self._generate_sm_summary(fold_msgs)
        if not incremental:
            return messages, None

        before = estimate_tokens(messages)

        self._session_memory_parts.append(incremental)
        self._fold_count += 1
        self._collapsed_msg_count = fold_end

        sm_text = self._build_session_memory_text()

        result = [*system_msgs, {"role": "user", "content": sm_text},
                  *conversation[fold_end:]]

        after = estimate_tokens(result)

        return result, CompressionEvent(
            layer="session_memory",
            before_tokens=before,
            after_tokens=after,
            messages_removed=fold_end - fold_start,
            timestamp=time.time(),
        )

    def _compute_fold_end(self, conversation: list[dict], start: int) -> int:
        """计算本批折叠到哪条消息（约 sm_step_tokens）"""
        keep_from_end = self.sm_keep_recent
        max_end = len(conversation) - keep_from_end
        if start >= max_end:
            return start

        accumulated = 0
        for i in range(start, max_end):
            accumulated += estimate_tokens([conversation[i]])
            if accumulated >= self.sm_step_tokens:
                return i + 1
        return max_end

    def _generate_sm_summary(self, messages: list[dict]) -> str:
        """调用 LLM 生成增量摘要"""
        from agents.agentic.prompts import build_sm_summary_prompt
        prompt = build_sm_summary_prompt(messages, self._fold_count + 1)
        try:
            from llm.client import judge_chat
            return judge_chat(prompt)
        except Exception:
            return self._fallback_sm_summary(messages)

    def _fallback_sm_summary(self, messages: list[dict]) -> str:
        """LLM 不可用时的规则摘要"""
        tools = []
        findings = []
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    tools.append(tc.get("function", {}).get("name", "?"))
            elif msg.get("role") == "tool":
                c = msg.get("content", "")
                if c and c != self._CLEARED_MARKER:
                    findings.append(c[:150])

        lines = [f"### 第 {self._fold_count + 1} 段折叠（{len(messages)} 条消息）\n"]
        if tools:
            lines.append(f"工具调用: {', '.join(tools[:10])}")
        if findings:
            lines.append("关键发现:")
            for f in findings[:5]:
                lines.append(f"- {f}")
        return "\n".join(lines)

    def _build_session_memory_text(self) -> str:
        """构建完整 SM 文本"""
        header = (f"<session-memory>\n"
                  f"## 对话折叠摘要（已折叠 {self._fold_count} 段，"
                  f"覆盖 {self._collapsed_msg_count} 条消息）\n\n")
        body = "\n\n".join(self._session_memory_parts)
        return header + body + "\n</session-memory>"

    # ═══════════════════════════════════════════════════════════
    # Layer 4: AI Summary — 全量摘要 + 熔断兜底
    # ═══════════════════════════════════════════════════════════

    def _ai_summary(self, messages: list[dict]) -> tuple[list[dict], Optional[CompressionEvent]]:
        """全量 9 段结构化摘要，熔断后走纯截断"""
        if self._circuit_breaker_count >= self.circuit_breaker_limit:
            return self._truncation_fallback(messages)

        before = estimate_tokens(messages)

        try:
            raw = self._generate_full_summary(messages)
            if not raw:
                self._circuit_breaker_count += 1
                return self._truncation_fallback(messages)
        except Exception:
            self._circuit_breaker_count += 1
            return self._truncation_fallback(messages)

        summary = self._truncate_summary(raw)

        system_msgs = [m for m in messages if m.get("role") == "system"]
        conversation = [m for m in messages if m.get("role") != "system"]

        result = [*system_msgs, {"role": "system", "content": summary},
                  *conversation[-self.micro_keep_recent:]]

        after = estimate_tokens(result)
        self._circuit_breaker_count = 0

        return result, CompressionEvent(
            layer="ai_summary",
            before_tokens=before,
            after_tokens=after,
            messages_removed=len(messages) - len(result),
            timestamp=time.time(),
        )

    def _generate_full_summary(self, messages: list[dict]) -> str:
        """Fork 子代理生成 9 段摘要"""
        from agents.agentic.prompts import build_ai_summary_prompt
        prompt = build_ai_summary_prompt(messages)
        try:
            from llm.client import judge_chat
            response = judge_chat(prompt)
            return self._strip_analysis(response)
        except Exception:
            return ""

    def _strip_analysis(self, raw: str) -> str:
        """移除 <analysis> 块，只保留 <summary>"""
        match = re.search(r'<summary>(.*?)</summary>', raw, re.DOTALL)
        if match:
            return f"<summary>\n{match.group(1).strip()}\n</summary>"
        return raw

    def _truncate_summary(self, summary: str) -> str:
        """截断摘要到 summary_max_tokens"""
        current = estimate_tokens([{"role": "system", "content": summary}])
        if current <= self.summary_max_tokens:
            return summary

        sections = summary.split("### ")
        header = sections[0] if sections else ""

        # 优先级: 8 (当前工作) > 1 > 6 > 7 > 4 > 5 > 2 > 9 > 3
        priority = [8, 1, 6, 7, 4, 5, 2, 9, 3]
        kept = []
        for num in priority:
            for sec in sections[1:]:
                if sec.startswith(f"{num}."):
                    kept.append(f"### {sec}")
                    break

        result = header
        for s in kept:
            candidate = result + s
            if estimate_tokens([{"role": "system", "content": candidate}]) > self.summary_max_tokens:
                result += "### ...[证据已截断]...\n"
                break
            result = candidate
        return result.strip()

    def _truncation_fallback(self, messages: list[dict]) -> tuple[list[dict], Optional[CompressionEvent]]:
        """熔断兜底：保留 SM 累积摘要 + 最近 K 条"""
        before = estimate_tokens(messages)
        system_msgs = [m for m in messages if m.get("role") == "system"]
        conversation = [m for m in messages if m.get("role") != "system"]

        keep = min(self.micro_keep_recent, len(conversation))
        result = list(system_msgs)

        if self._session_memory_parts:
            result.append({"role": "user", "content": self._build_session_memory_text()})

        result.extend(conversation[-keep:])

        after = estimate_tokens(result)
        return result, CompressionEvent(
            layer="ai_summary",
            before_tokens=before,
            after_tokens=after,
            messages_removed=len(messages) - len(result),
            timestamp=time.time(),
        )
