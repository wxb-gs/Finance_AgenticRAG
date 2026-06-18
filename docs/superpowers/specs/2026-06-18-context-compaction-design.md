# 上下文压缩四层架构设计

## 概述

将对齐 Claude Code 的四层渐进式上下文压缩管线，替代当前按模型尺寸分三档的硬编码策略。

## 预算分配（32k 上下文）

| 项目 | Token 数 | 说明 |
|------|---------|------|
| 总上下文窗口 `max_tokens` | 32768 | |
| 输出预留 `output_reserve` | 4096 | 模型回答空间 |
| 有效上下文 | 28672 | |
| Layer 4 缓冲 | ~4000 | |
| Layer 4 触发线 | ~24672 (75%) | `compress_threshold` 控制 |
| Layer 2 触发 | 每次 LLM 请求前 | 热路径，无阈值 |

---

## Layer 1: Snip

**定位**：零成本移除低价值完整 turn。

**触发**：每次 LLM 调用前检查，token > 70% 时执行。

**移除对象**：
- 工具返回空结果（0 条命中）
- `ToolResult.is_empty == True`
- 连续重复调用同一工具且返回相同 chunk_id
- 无效的 `activate_skill`（未匹配到技能）

**不删**：system 消息、user 原始查询、finish、plan_query/plan_update、dispatch_subagent、remember。

**实现**：纯 list 过滤，O(n)，不调 API。

**可配参数**：`snip_enabled`（默认 `False`，开关关闭）。

---

## Layer 2: MicroCompact

**定位**：清理旧 tool_result 内容，替换为占位符，不破坏消息结构。

**触发**：每次 LLM 调用前（热路径）。

**可清理工具**（输出可重放）：
- `semantic_search`、`keyword_search`、`graph_search`、`hybrid_search`
- `read_chunk`、`text_to_sql`

**不可清理工具**：
- `finish`、`dispatch_subagent`、`activate_skill`
- `remember`、`plan_query`、`plan_update`、`mcp__*`

**逻辑**：保留最近 `micro_keep_recent` 条（默认 6）tool_result 完整内容，更早的 → 替换为 `[Old tool result content cleared]`。

**可配参数**：`micro_keep_recent`（默认 6）。

---

## Layer 3: Session Memory（增量式上下文折叠）

**定位**：用 LLM 生成增量摘要折叠中间旧消息，减少 token 的同时保留上下文脉络。

**触发**：
- 首次：总 token ≥ `sm_min_tokens`（默认 10000）
- 后续：每次再增加 `sm_step_tokens`（默认 5000）

**流程**：
1. 找出最早一批未被折叠的消息（约 sm_step_tokens 量）
2. 对这批消息生成增量摘要
3. 增量摘要追加到已有 Session Memory 后面
4. 被折叠消息从数组中移除，替换为更新后的 Session Memory

**压缩后消息结构**：
```
[system, compactBoundary, SM(user消息), ...保留最近 sm_keep_recent 条]
```

**Session Memory 格式**：
```
<session-memory>
## 对话折叠摘要（已折叠 N 段，覆盖消息 a~b）

### 第 X 段折叠（消息 m~n）
- 检索路径
- 关键发现
- 矛盾/缺口

### 当前待解决问题
- ...
</session-memory>
```

**摘要 LLM**：调用 `judge_chat`，不带工具，只处理增量段。

**可配参数**：`sm_min_tokens`（默认 10000）、`sm_step_tokens`（默认 5000）、`sm_keep_recent`（默认 8）。

---

## Layer 4: AI Summary + 截断

**定位**：Layer 3 兜不住时的最后手段。Fork 子代理做全量结构化摘要。

**触发**：Layer 3 增量折叠后 token 仍 ≥ `compress_threshold × (max_tokens - output_reserve)`（即 75% 触发线）。

**流程**：
1. Fork 子代理（不带工具）
2. 生成 9 段 + `<analysis>` 结构化摘要
3. `<analysis>` 被 strip，不进上下文
4. `<summary>` 截断到 `summary_max_tokens`
5. 重置消息数组

**9 段摘要模板**：
```
<analysis>（chronology 分析——最终 strip）</analysis>
<summary>
### 1. 原始查询与意图
### 2. 关键金融概念
### 3. 已检索的文件与数据（含 chunk_id）
### 4. 发现的数据矛盾与处理
### 5. 已解决的问题
### 6. 所有用户消息（逐字）
### 7. 待完成任务
### 8. 当前工作（压缩前正在做的事，含证据片段）
### 9. 下一步建议（引用原文）
</summary>
```

**截断规则**：
- `<summary>` 上限 `summary_max_tokens`（默认 4000）
- 第 8 节优先保留，第 3 节次之
- 超出从第 3 节开始截断，标记 `...[证据已截断]...`

**压缩后消息结构**：
```
[system, boundaryMarker, summary(system消息), ...保留最近 micro_keep_recent 条, ...re-attach(技能/附件)]
```

**熔断**：连续 `circuit_breaker` 次（默认 3）Layer 4 失败 → 禁用 AI Summary，走纯截断兜底。

**可配参数**：`summary_max_tokens`（默认 4000）、`circuit_breaker`（默认 3）。

---

## 数据模型

### CompressionEvent（替代现有）

```python
@dataclass
class CompressionEvent:
    layer: Literal["snip", "microcompact", "session_memory", "ai_summary"]
    before_tokens: int
    after_tokens: int
    messages_removed: int
    timestamp: float
```

### CompactBoundary

```python
@dataclass
class CompactBoundary:
    fold_count: int
    collapsed_ranges: list[tuple[int, int]]
    pre_tokens: int
    post_tokens: int
    timestamp: float
```

---

## 全部可配参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_tokens` | 32768 | 总上下文窗口 |
| `compress_threshold` | 0.75 | 触发压缩的利用率 |
| `output_reserve` | 4096 | 输出预留 |
| `snip_enabled` | False | Layer 1 开关 |
| `micro_keep_recent` | 6 | Layer 2 保留最近 K 条 |
| `sm_min_tokens` | 10000 | Layer 3 首次触发 |
| `sm_step_tokens` | 5000 | Layer 3 增量步长 |
| `sm_keep_recent` | 8 | Layer 3 保留最近 K 条 |
| `summary_max_tokens` | 4000 | Layer 4 摘要上限 |
| `circuit_breaker` | 3 | Layer 4 熔断次数 |

---

## 文件变更清单

| 文件 | 变更 |
|------|------|
| `agents/agentic/context.py` | 完全重写，四层管线 |
| `agents/agentic/types.py` | 更新 `CompressionEvent`，新增 `CompactBoundary` |
| `agents/agentic/agent.py` | 调用方适配新的 `ContextManager` 接口 |
| `agents/agentic/prompts.py` | 新增 Layer 4 的 9 段摘要 prompt 模板 |
| `config.py` | 新增压缩相关可配参数 |
| `tests/test_agentic.py` | 更新 `TestContext` 测试用例 |
