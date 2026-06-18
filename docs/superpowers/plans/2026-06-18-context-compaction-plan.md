# 上下文压缩四层架构实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将当前按模型尺寸分三档的上下文压缩，替换为对齐 Claude Code 的四层渐进式管线（Snip → MicroCompact → Session Memory → AI Summary + 截断）。

**Architecture:** `ContextManager` 统一管理四层管线，每层有独立触发条件和可配参数。Layer 1/2 零 API 成本，Layer 3 增量调用 LLM，Layer 4 全量摘要加熔断。压缩事件以 `CompressionEvent` 列表返回。

**Tech Stack:** Python 3.10+, tiktoken, 项目已有 `judge_chat`/`agent_chat` LLM 接口

---

## 文件结构

| 文件 | 职责 | 变更类型 |
|------|------|----------|
| `agents/agentic/types.py` | `CompressionEvent` 重定义 + 新增 `CompactBoundary` | 修改 |
| `agents/agentic/prompts.py` | Layer 3 SM 增量摘要 prompt + Layer 4 9段摘要 prompt | 修改 |
| `agents/agentic/context.py` | 四层管线 `ContextManager` | 完全重写 |
| `agents/agentic/agent.py` | 适配新 `ContextManager` 接口 | 修改 |
| `config.py` | 新增压缩参数到 `AGENT_CONFIG` | 修改 |
| `tests/test_agentic.py` | 更新 `TestContext` 测试 | 修改 |

---

### Task 1: 更新 `types.py` — 重定义 CompressionEvent + 新增 CompactBoundary

**Files:**
- Modify: `agents/agentic/types.py`

- [ ] **Step 1: 重定义 CompressionEvent**

将当前按 `strategy` 字段的 `CompressionEvent` 替换为按 `layer` 字段：

```python
@dataclass
class CompressionEvent:
    """一次压缩事件"""
    layer: Literal["snip", "microcompact", "session_memory", "ai_summary"]
    before_tokens: int
    after_tokens: int
    messages_removed: int = 0
    timestamp: float = field(default_factory=time.time)
```

打开 `agents/agentic/types.py`，找到第 35-40 行的 `CompressionEvent`，替换为上述代码。

- [ ] **Step 2: 新增 CompactBoundary**

在 `CompressionEvent` 下方新增：

```python
@dataclass
class CompactBoundary:
    """Session Memory 折叠边界元数据"""
    fold_count: int
    collapsed_msg_count: int
    pre_tokens: int
    post_tokens: int
    timestamp: float = field(default_factory=time.time)
```

- [ ] **Step 3: 更新 AgentState.compression_events 的序列化字段**

将 `to_result()` 中对 `compression_events` 的序列化从 `strategy` 改为 `layer`。在 `types.py` 第 85-88 行：

```python
"compression_events": [
    {"layer": e.layer, "before": e.before_tokens, "after": e.after_tokens}
    for e in self.compression_events
],
```

- [ ] **Step 4: 运行现有测试确认类型变更不破坏引用**

```bash
pytest tests/test_agentic.py::TestContext -v
```

预期：test_count_tokens 通过，test_should_compress 等旧测试失败（因为 ContextManager 接口变更）—— 这是预期的，Task 3 会补全。

- [ ] **Step 5: 提交**

```bash
git add agents/agentic/types.py
git commit -m "refactor: update CompressionEvent layer field and add CompactBoundary"
```

---

### Task 2: 新增压缩相关 Prompt 模板

**Files:**
- Modify: `agents/agentic/prompts.py`

- [ ] **Step 1: 添加 Layer 3 Session Memory 增量摘要 prompt**

在 `prompts.py` 末尾新增函数 `build_sm_summary_prompt`：

```python
def build_sm_summary_prompt(messages: list[dict], fold_index: int, language: str = "zh") -> str:
    """构建 Layer 3 Session Memory 增量摘要 prompt

    Args:
        messages: 本轮要折叠的消息段
        fold_index: 第几次折叠（1-based）
        language: zh | en
    """
    conversation_text = _format_messages_for_summary(messages)

    if language == "zh":
        return f"""你是对话摘要助手。请将以下对话片段压缩为结构化摘要。

这是第 {fold_index} 次增量折叠，只描述本段新增内容。

<对话片段>
{conversation_text}
</对话片段>

输出精简摘要（150字以内）：
- 本段执行的检索工具及参数
- 新发现的关键数据/证据
- 新出现的矛盾或信息缺口
- 当前推理方向"""

    return f"""You are a conversation summarizer. Compress the following conversation segment into a structured summary.

This is incremental fold #{fold_index}. Only describe what's NEW in this segment.

<conversation>
{conversation_text}
</conversation>

Output a concise summary:
- Retrieval tools called in this segment with parameters
- New key data/evidence discovered
- New contradictions or information gaps
- Current reasoning direction"""
```

- [ ] **Step 2: 添加 Layer 4 AI Summary 9 段摘要 prompt**

在 `build_sm_summary_prompt` 下方新增 `build_ai_summary_prompt`：

```python
def build_ai_summary_prompt(messages: list[dict], language: str = "zh") -> str:
    """构建 Layer 4 AI Summary 9 段全量摘要 prompt"""
    conversation_text = _format_messages_for_summary(messages)

    if language == "zh":
        return f"""你是对话摘要助手。你需要将整个对话历史压缩为结构化摘要，保留所有关键信息。

重要：先输出 <analysis> 块做按时间线的对话分析（该块不会被保留），再输出 <summary> 块。

<对话历史>
{conversation_text}
</对话历史>

<summary> 必须包含以下 9 段：
### 1. 原始查询与意图
### 2. 关键金融概念
### 3. 已检索的文件与数据（含 chunk_id + 完整证据片段）
### 4. 发现的数据矛盾与处理
### 5. 已解决的问题
### 6. 所有用户消息（逐字保留）
### 7. 待完成任务
### 8. 当前工作（压缩前正在做的事，含具体文件名/代码片段/证据）
### 9. 下一步建议（引用对话原文作为依据）

禁止调用任何工具。直接输出分析文本。"""

    return f"""You are a conversation summarizer. Compress the entire conversation history into a structured summary preserving all critical information.

Important: First output an <analysis> block with chronological conversation analysis (this block will not be retained), then output the <summary> block.

<conversation>
{conversation_text}
</conversation>

<summary> must contain these 9 sections:
### 1. Primary Request and Intent
### 2. Key Financial Concepts
### 3. Files and Data Retrieved (with chunk_id + full evidence snippets)
### 4. Data Contradictions Discovered and How They Were Handled
### 5. Problems Solved
### 6. All User Messages (verbatim)
### 7. Pending Tasks
### 8. Current Work (what was being worked on before compaction, with specific filenames/code/evidence)
### 9. Next Step (with direct quotes from conversation as justification)

Do NOT call any tools. Output analysis text directly."""
```

- [ ] **Step 3: 添加 messages 格式化辅助函数**

在两个 prompt 函数之前添加 `_format_messages_for_summary`：

```python
def _format_messages_for_summary(messages: list[dict]) -> str:
    """将消息列表格式化为 LLM 可读的文本"""
    lines = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if role == "system":
            lines.append(f"[{i}] system: {content[:200]}")
        elif role == "user":
            lines.append(f"[{i}] user: {content[:500]}")
        elif role == "assistant":
            if msg.get("tool_calls"):
                tools = [tc.get("function", {}).get("name", "?") for tc in msg["tool_calls"]]
                lines.append(f"[{i}] assistant: calls {tools}")
                if content:
                    lines.append(f"  reasoning: {content[:200]}")
            else:
                lines.append(f"[{i}] assistant: {content[:300]}")
        elif role == "tool":
            tc_id = msg.get("tool_call_id", "")[:8]
            lines.append(f"[{i}] tool_result({tc_id}): {content[:300]}")
    return "\n".join(lines)
```

- [ ] **Step 4: 运行测试确认导入正确**

```bash
python -c "from agents.agentic.prompts import build_sm_summary_prompt, build_ai_summary_prompt; print('OK')"
```

- [ ] **Step 5: 提交**

```bash
git add agents/agentic/prompts.py
git commit -m "feat: add Layer 3 SM and Layer 4 AI summary prompt templates"
```

---

### Task 3: 重写 `context.py` — 四层管线 ContextManager

**Files:**
- Modify: `agents/agentic/context.py`

- [ ] **Step 1: 保留并增强 token 估算函数**

将文件顶部的 `count_tokens` 改为保留 tiktoken 主路径，添加纯字符 fallback。同时新增文件头部注释：

```python
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
```

- [ ] **Step 2: 定义 ContextManager 类框架和 `__init__`**

```python
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
```

- [ ] **Step 3: 实现 `should_compress`、`compress`、`touch` 顶层方法**

```python
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
```

- [ ] **Step 4: 实现 Layer 1: Snip**

```python
    # ═══════════════════════════════════════════════════════
    # Layer 1: Snip — 移除整轮低价值 turn
    # ═══════════════════════════════════════════════════════

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
```

- [ ] **Step 5: 实现 Layer 2: MicroCompact**

```python
    # ═══════════════════════════════════════════════════════
    # Layer 2: MicroCompact — 缓存感知 tool_result 清理
    # ═══════════════════════════════════════════════════════

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
```

- [ ] **Step 6: 实现 Layer 3: Session Memory**

```python
    # ═══════════════════════════════════════════════════════
    # Layer 3: Session Memory — 增量式上下文折叠
    # ═══════════════════════════════════════════════════════

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
```

- [ ] **Step 7: 实现 Layer 4: AI Summary + 截断**

```python
    # ═══════════════════════════════════════════════════════
    # Layer 4: AI Summary — 全量摘要 + 熔断兜底
    # ═══════════════════════════════════════════════════════

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
```

- [ ] **Step 8: 运行测试**

```bash
pytest tests/test_agentic.py::TestContext -v
```

预期：test_count_tokens 通过；其他旧测试因接口变更可能失败。Task 6 会更新测试。

- [ ] **Step 9: 提交**

```bash
git add agents/agentic/context.py
git commit -m "feat: rewrite ContextManager with 4-layer compaction pipeline"
```

---

### Task 4: 适配 `agent.py` 调用新 ContextManager

**Files:**
- Modify: `agents/agentic/agent.py`

- [ ] **Step 1: 更新 ContextManager 初始化参数**

在 `Agent.__init__` 中，第 29 行 `self.context = ContextManager(model_size=model_size)` 替换为从 config 读取参数：

```python
from config import AGENT_CONFIG

self.context = ContextManager(
    max_tokens=AGENT_CONFIG.get("context_max_tokens", 32768),
    compress_threshold=AGENT_CONFIG.get("context_compress_threshold", 0.75),
    output_reserve=AGENT_CONFIG.get("context_output_reserve", 4096),
    snip_enabled=AGENT_CONFIG.get("context_snip_enabled", False),
    micro_keep_recent=AGENT_CONFIG.get("context_micro_keep_recent", 5),
    micro_trigger_threshold=AGENT_CONFIG.get("context_micro_trigger_threshold", 10),
    micro_idle_minutes=AGENT_CONFIG.get("context_micro_idle_minutes", 60),
    sm_min_tokens=AGENT_CONFIG.get("context_sm_min_tokens", 10000),
    sm_step_tokens=AGENT_CONFIG.get("context_sm_step_tokens", 5000),
    sm_keep_recent=AGENT_CONFIG.get("context_sm_keep_recent", 8),
    summary_max_tokens=AGENT_CONFIG.get("context_summary_max_tokens", 4000),
    circuit_breaker=AGENT_CONFIG.get("context_circuit_breaker", 3),
)
```

- [ ] **Step 2: 更新 ReAct 循环中的压缩调用**

在 `agent.py` 第 83-86 行，将旧接口：

```python
if self.context.should_compress(messages):
    messages, event = self.context.compress(messages)
    state.compression_events.append(event)
```

替换为新接口（compress 返回事件列表），并在每次 LLM 调用前 `touch()`：

```python
self.context.touch()
if self.context.should_compress(messages):
    messages, events = self.context.compress(messages)
    state.compression_events.extend(events)
```

- [ ] **Step 3: 验证导入和初始化**

```bash
python -c "from agents.agentic.agent import Agent; a = Agent(model_size='small'); print(type(a.context).__name__)"
```

预期输出: `ContextManager`

- [ ] **Step 4: 提交**

```bash
git add agents/agentic/agent.py
git commit -m "refactor: adapt Agent to new 4-layer ContextManager interface"
```

---

### Task 5: 更新 `config.py` 添加压缩参数

**Files:**
- Modify: `config.py`

- [ ] **Step 1: 在 AGENT_CONFIG 中添加压缩参数**

```python
AGENT_CONFIG = {
    # ... 现有配置 ...
    "agent_enable_subagents": os.environ.get("AGENT_ENABLE_SUBAGENTS", "true").lower() == "true",

    # ── 上下文压缩配置 ──
    "context_max_tokens": int(os.environ.get("CONTEXT_MAX_TOKENS", "32768")),
    "context_compress_threshold": float(os.environ.get("CONTEXT_COMPRESS_THRESHOLD", "0.75")),
    "context_output_reserve": int(os.environ.get("CONTEXT_OUTPUT_RESERVE", "4096")),
    "context_snip_enabled": os.environ.get("CONTEXT_SNIP_ENABLED", "false").lower() == "true",
    "context_micro_keep_recent": int(os.environ.get("CONTEXT_MICRO_KEEP_RECENT", "5")),
    "context_micro_trigger_threshold": int(os.environ.get("CONTEXT_MICRO_TRIGGER_THRESHOLD", "10")),
    "context_micro_idle_minutes": int(os.environ.get("CONTEXT_MICRO_IDLE_MINUTES", "60")),
    "context_sm_min_tokens": int(os.environ.get("CONTEXT_SM_MIN_TOKENS", "10000")),
    "context_sm_step_tokens": int(os.environ.get("CONTEXT_SM_STEP_TOKENS", "5000")),
    "context_sm_keep_recent": int(os.environ.get("CONTEXT_SM_KEEP_RECENT", "8")),
    "context_summary_max_tokens": int(os.environ.get("CONTEXT_SUMMARY_MAX_TOKENS", "4000")),
    "context_circuit_breaker": int(os.environ.get("CONTEXT_CIRCUIT_BREAKER", "3")),
}
```

- [ ] **Step 2: 验证配置导入**

```bash
python -c "from config import AGENT_CONFIG; print(AGENT_CONFIG['context_max_tokens'])"
```

预期输出: `32768`

- [ ] **Step 3: 提交**

```bash
git add config.py
git commit -m "feat: add context compaction parameters to AGENT_CONFIG"
```

---

### Task 6: 更新测试

**Files:**
- Modify: `tests/test_agentic.py`

- [ ] **Step 1: 重写 TestContext 测试类**

替换 `tests/test_agentic.py` 中第 185-227 行的 `TestContext` 类：

```python
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
        large = [{"role": "system", "content": "A" * 400}]
        assert mgr.should_compress(large)

    def test_microcompact_idle_trigger(self):
        from agents.agentic.context import ContextManager
        import time
        mgr = ContextManager(micro_trigger_threshold=999, micro_idle_minutes=0)
        # micro_idle_minutes=0 意味着任何正空闲时间都触发
        mgr._last_active_at = 0  # 模拟很久以前活动
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
        # 应该触发 microcompact
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
        # 检查只有最近 1 条保留完整内容
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
        # finish 的 tool_result 应保持完整
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
        # snip 关闭，消息数不应减少
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
        # 设置一个会触发全部层的场景
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
        # snip 在 microcompact 之前
        if "snip" in layers and "microcompact" in layers:
            snip_idx = layers.index("snip")
            mc_idx = layers.index("microcompact")
            assert snip_idx < mc_idx, f"Expected snip before microcompact, got {layers}"
        # 压缩后 token 应减少
        from agents.agentic.context import estimate_tokens
        assert estimate_tokens(result) < estimate_tokens(msgs)
```

- [ ] **Step 2: 运行测试**

```bash
pytest tests/test_agentic.py::TestContext -v
```

预期：所有测试通过。

- [ ] **Step 3: 修复因 Agent.__init__ 变更导致的 TestAgent 测试失败**

运行完整测试套件：

```bash
pytest tests/test_agentic.py -v
```

检查 `TestAgent::test_agent_init` 是否失败。如果 `AGENT_CONFIG` 中缺少某个 key 导致 KeyError，在 `agent.py` 的初始化中使用 `.get()` 并保留默认值兜底。

- [ ] **Step 4: 提交**

```bash
git add tests/test_agentic.py
git commit -m "test: rewrite TestContext for 4-layer compaction pipeline"
```
