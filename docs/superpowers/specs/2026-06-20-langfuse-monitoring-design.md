# Langfuse 监控与评测系统设计

## 概述

基于自托管 Langfuse 搭建 Agent 链路执行轨迹持久化与可视化追踪系统，集成到生产 Agent 循环中（非仅离线评测）。涵盖全链路 Tracing、多维评测指标上报、Bad Case 自动分类与优化建议生成。

## 模块结构

```
monitoring/
├── __init__.py          # 导出 Tracer, EvalReporter, BadCaseRouter
├── tracer.py            # Langfuse Trace/Span/Generation 生命周期管理
├── eval_reporter.py     # 评测分数计算 + 上报（Tool Accuracy, Hop Recall, LLM-Judge）
└── badcase_router.py    # Bad Case 识别 → 分类 → Prompt/Schema 优化建议
```

## Tracer 设计

### 核心接口

```python
class Tracer:
    def __init__(self, enabled: bool, langfuse_client: Langfuse)
    def start_trace(query, mode, model, metadata)       # 创建根 Trace
    def end_trace(result: AgentResult)                   # 结束 Trace，附加元数据
    def start_iteration(iter_num: int) -> Span           # 迭代 Span
    def end_iteration()
    def log_generation(model, messages, response, latency_ms, usage)  # LLM Generation
    def log_tool_call(name, args, result, latency_ms)    # 工具 Span
    def log_compression(layer, tokens_before, tokens_after)  # 压缩事件
    def log_subagent(sub_type, task, iterations, result) # 子代理 Span
    def log_recall(memories: list)                       # 记忆召回
    def score(name, value, metadata)                     # 上报分数
    def score_many(scores: dict)                         # 批量上报
    def log_error(error: Exception)                      # 异常记录
    def flush()                                          # 异步批量上传
```

### No-op 模式

`LANGFUSE_ENABLED=false` 时，所有方法检查 `self.enabled` 后直接 return。零 Langfuse SDK 开销，零网络 IO。

### Span 层级

```
Trace "用户查询" (query, mode, model)
├── Span "recall" (记忆召回, top_k 条)
├── Span "iteration_1"
│   ├── Generation "llm_call" (model, tokens_in, tokens_out, latency_ms)
│   ├── Span "tool:semantic_search" (args, result_count, confidence, latency_ms)
│   └── Span "tool:keyword_search"
├── Span "iteration_2"
│   ├── Generation "llm_call"
│   └── Span "tool:read_chunk"
├── Event "compression_l2" (tokens_before, tokens_after, timestamp)
├── Span "iteration_3"
│   ├── Generation "llm_call"
│   └── Span "tool:finish" (answer, confidence)
└── Scores: [tool_selection_f1, arg_quality_avg, step_efficiency, ...]
```

## Agent 循环埋点

### PipelineRouter 改动 (~15 行)

- `run()` 方法中创建 Tracer，调用 `start_trace()`
- 将 tracer 传入 Agent.run()
- 在 finally 中 `tracer.flush()`

### Agent.run() 改动 (~30 行)

- 新增 `tracer` 可选参数，默认为 `Tracer.noop()`
- 3 处埋点：记忆召回 → 每次迭代（LLM + 工具执行）→ 结束

### 配置

```python
# config.py 新增
LANGFUSE_ENABLED = os.getenv("LANGFUSE_ENABLED", "false").lower() == "true"
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_FLUSH_INTERVAL = int(os.getenv("LANGFUSE_FLUSH_INTERVAL", "5"))  # 秒
```

## 评测指标

### 1. 工具选择 Accuracy（自动）

```
actual_set = {工具名列表}, expected_set = {标注工具集}
Precision = |actual ∩ expected| / |actual|
Recall    = |actual ∩ expected| / |expected|
F1        = 2 × P × R / (P + R)
```

上报为 Langfuse Score：`tool_selection_precision`, `tool_selection_recall`, `tool_selection_f1`

### 2. 参数质量（LLM-as-Judge）

- 对每次工具调用，judge_chat 打分 1-5
- 评分标准：5=精准覆盖关键实体/约束, 3=部分覆盖, 1=严重偏离
- 上报为 Langfuse Score：`arg_quality_avg`, `arg_quality_min`

### 3. 调用效率（自动）

- 冗余率：`is_empty=true` 的调用数 / 总调用数
- 重复率：重复调用计数
- 步数效率：理论最小步数 / 实际步数
- 上报：`redundancy_rate`, `repetition_count`, `step_efficiency`

### 4. 规划合理性（自动）

- Agent plan_steps 与 ground_truth_hops 的语义匹配
- 计算 hop 级别 Precision/Recall
- 上报：`plan_hop_precision`, `plan_hop_recall`

## Bad Case 驱动迭代

### 分类规则（6 类）

| Bad Case 类型 | 触发条件 | 优化目标 |
|--------------|---------|---------|
| tool_selection_low | tool_selection_f1 < 0.5 | tool_schema |
| arg_quality_poor | arg_quality_avg < 3.0 | prompt |
| high_redundancy | redundancy_rate > 0.4 | tool_schema |
| step_inefficient | step_efficiency < 0.5 | prompt |
| early_finish | no_tool_streak >= 3 且 answer不正确 | prompt |
| plan_mismatch | plan_hop_recall < 0.5 | prompt |

### 闭环流程

```
BadCaseRouter.classify() → 分类 → 累积同类 ≥ 5个 → 生成优化建议
                                                      ├→ prompt 类 → 修改 prompts.py
                                                      └→ schema 类 → 修改 tools.py schema
                                                      ↓
                                                  重新评测验证
```

### 设计决策

- 分类用规则阈值（确定性、低成本），仅在生成优化建议时用 LLM
- 同类累积 ≥ 5 个才触发建议，避免单点过拟合
- 输出格式为具体修改建议（diff 级别），而非抽象描述

## 文件改动量估算

| 文件 | 改动 | 说明 |
|------|------|------|
| `config.py` | +8 行 | Langfuse 配置项 |
| `monitoring/__init__.py` | 新建 | 模块导出 |
| `monitoring/tracer.py` | 新建 ~150 行 | Tracer 类 |
| `monitoring/eval_reporter.py` | 新建 ~200 行 | 评测上报 |
| `monitoring/badcase_router.py` | 新建 ~150 行 | Bad Case 分类 |
| `pipeline_router.py` | +20 行 | 根 Trace 创建/结束 |
| `agents/agentic/agent.py` | +35 行 | 循环内 3 处埋点 |
| `agents/agentic/types.py` | +5 行 | AgentResult 增加 trace_id, no_tool_streak 等诊断字段 |
| `requirements.txt` | +1 行 | langfuse SDK |

总计约 570 行新代码 + 70 行改动。

## 依赖

- `langfuse >= 3.0.0`：SDK，支持异步 flush、OpenAI 兼容
- 自托管 Langfuse 实例（需提前部署并配置 LANGFUSE_HOST/KEYS）
