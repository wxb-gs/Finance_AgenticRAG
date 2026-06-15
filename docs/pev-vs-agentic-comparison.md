# PEV (LangGraph) vs Agentic (ReAct) 架构对比分析

日期: 2026-06-15

## 1. 核心差异：控制权归属

两种架构的本质区别在于**谁决定下一步做什么**。

| | PEV (LangGraph) | Agentic (ReAct) |
|---|---|---|
| **编排方式** | 图结构，编译期确定的边决定流转 | while 循环，模型运行时决定每步行为 |
| **计划生成** | 前置：Planner 先分解子任务再执行 | 涌现：边搜边调整，无显式计划步骤 |
| **验证机制** | 独立 Verifier 节点评分，不通过则重搜 | 模型自行判断信息是否充分 |
| **状态管理** | TypedDict + Annotated reducer，LangGraph 内置 checkpoint | dataclass，无内置持久化 |
| **工具调用** | Executor 按计划串行，一个子任务一步 | 模型自主选工具，可单轮并行多工具 |
| **停止条件** | 图结构决定（验证通过→合成，否则回退） | 模型调用 `finish` 或 max_iterations 硬上限 |
| **模型角色** | 各节点可指定不同模型，分工明确 | 单一模型贯穿全程，承担所有决策 |

## 2. 架构图解

### PEV 流程

```
                    ┌──────────┐
                    │  Router  │  判断查询复杂度 (simple/multi_hop)
                    └────┬─────┘
                         │
                    ┌────▼─────┐
                    │  Planner │  分解为子任务列表
                    └────┬─────┘   [{id, sub_query, depends_on}]
                         │
              ┌──────────▼──────────┐
              │     Executor        │  按依赖顺序执行每个子任务
              │  semantic_search    │  调用检索工具 → 收集证据
              │  keyword_search     │
              └──────────┬──────────┘
                         │
                    ┌────▼─────┐
                    │ Verifier  │  评估证据充分性
                    └────┬─────┘  sufficient → 前进
                         │        insufficient → 返回 Planner 重规划
                    ┌────▼──────┐
                    │Synthesizer│  基于所有 evidence 生成最终答案
                    └───────────┘
```

控制流由 LangGraph StateGraph 的边决定，模型只在各节点内部做局部推理。

### Agentic 流程

```
   System Prompt + Tools + Skills
                │
   ┌────────────▼────────────┐
   │    think: 需要搜什么？    │  ← 模型推理
   └────────────┬────────────┘
                │
   ┌────────────▼────────────┐
   │    act: 调用工具          │  ← 可并行多工具
   │    semantic_search       │
   │    dispatch_subagent     │
   │    activate_skill        │
   └────────────┬────────────┘
                │
   ┌────────────▼────────────┐
   │    observe: 分析结果      │  ← 模型评估
   └────────────┬────────────┘
                │
         ┌──────┴──────┐
         │  够了？       │
         │  finish()    │  ← 模型自主决定
         │  不够？继续    │
         └──────────────┘
```

无固定路径。模型每轮自主决定：用什么工具、搜几次、何时结束。

## 3. 优缺点

### PEV 优点

**可控性强。** 每一步做什么由图的边决定，不会出现模型"跑偏"的情况。适合对流程有严格合规要求的金融场景。

**可审计。** Plan 是显式的子任务列表，Verifier 有独立的充分性判断，出问题时能精确定位到是 Planner 分解错了、Executor 搜漏了、还是 Verifier 判断过于宽松。

**LangGraph 生态。** 自带 checkpoint（中断恢复）、streaming（流式输出中间状态）、interrupt（人工审批节点）等能力。虽当前代码未深度使用，但架构上可随时接入。

**模型容错。** 各节点可以用不同规格的模型——Planner 用大模型保证分解质量，Executor 用小模型降低成本。一个节点失败不拖垮全局。

**确定性高。** 相同输入走几乎相同的执行路径，调试时能稳定复现。

### PEV 缺点

**僵化。** 简单查询也要走完整 5 节点流程（路由→规划→执行→验证→合成），浪费 token 和时间。一个"贵州茅台股票代码是多少"也要 Planner 先分解。

**Plan 质量瓶颈。** 如果 Planner 第一步就分解错了，整条链路都偏。没有中途修正机制——只能等 Verifier 说 insufficient 再重规划，但重规划本身也可能偏。

**串行化。** 子任务间的依赖关系是 Planner 静态指定的。即使两个子任务实际无依赖，也必须按序执行。

**无自主学习。** 每次查询从零开始，不会从历史中积累经验。同样的问题问第二遍，仍然走完整流程。

### Agentic 优点

**灵活性高。** 简单查询 1-2 轮直接 finish，复杂查询自动多搜几轮。不需要任何前置计划。

**自适应。** 搜到矛盾数据可以立即转向深入调查，不需要等外部 Verifier 下判断。发现某个方向是死胡同时自行切换。

**模型驱动。** 大模型能力越强，Agent 表现越好。不需要为每种查询类型设计流程——模型自己"理解"该怎么做。

**并行友好。** `dispatch_subagent` 天然支持无依赖子任务并行派发。不同公司、不同指标、不同时间段的数据获取可以同时跑。

**有记忆。** `remember` 工具 + `MemoryManager` 在跨会话层面积累知识。搜过的矛盾点、确认的信息缺口下次直接可用。

### Agentic 缺点

**不可控。** 小模型可能乱选工具、在同一个词上反复搜、不调用 finish。你只能设置 max_iterations 硬兜底，无法在"逻辑层面"约束它。

**状态不持久。** 没有 checkpoint。Agent 跑了 12 轮后崩溃，全部丢失重来。LangGraph 的 `MemorySaver` / `SqliteSaver` 对此是原生的。

**调试困难。** trace 是扁平的 `[{iteration, tool_call, args, result_summary}]` 列表。不如 PEV 的 `plan → evidence → verification` 结构清晰。

**模型强依赖。** 7B 小模型在 Agentic 模式下表现远不如 32B+。PEV 可以通过图结构弥补单个节点模型能力不足，Agentic 不行。

**上下文膨胀。** ReAct 循环把所有中间结果堆在 message history 里，长查询必定触发压缩。PEV 每个节点是独立 LLM 调用，上下文天然隔离。

## 4. 适用场景

| 场景 | 推荐架构 | 原因 |
|------|---------|------|
| 简单事实查询 ("XX公司股票代码") | Agentic | 1-2 轮直接 finish，PEV 也要走完整流程 |
| 流程固化的合规审查 | PEV | 每一步必须可审计，不能"模型自己决定" |
| 开放式探索性查询 | Agentic | 不知道最终需要多少证据，需模型自主判断 |
| 7B-14B 小模型 | PEV | 图结构弥补模型推理能力不足 |
| 32B+ 大模型 | Agentic | 充分发挥模型推理和工具选择能力 |
| 需要 checkpoint/中断恢复 | PEV | LangGraph 原生 checkpointer |
| 多实体并行对比 | Agentic | dispatch_subagent 天然并行 |
| 单实体多维度深度分析 | PEV | Planner 分解 + Verifier 把关更可靠 |
| 生产环境需要 SLA 保证 | PEV | 确定性高，延迟可控 |
| 实验/原型快速迭代 | Agentic | 无需设计图结构，改 System Prompt 即可 |

## 5. 当前实现的状态管理对比

```
                      PEV                          Agentic
                   ─────────                    ──────────
运行时状态     TypedDict (内存)              dataclass (内存)
持久化         无 (LangGraph checkpoint      无 (AgentResult 返回后丢弃)
               API 可用但未接入)
跨会话记忆     无                            MemoryManager (文件系统)
断点续跑       evaluation checkpoint         无
               (JSONL, 仅评测脚本)
```

两者的状态管理目前都是"跑完就丢"。PEV 理论上可通过 LangGraph 的 `checkpointer` 参数接入持久化（`MemorySaver` 或 `SqliteSaver`），但当前代码未启用。Agentic 则完全没有 checkpoint 机制——这是后续迭代的明确方向。

## 6. 总结

PEV 和 Agentic 不是替代关系，是互补关系。PEV 适合**流程确定、需要审计、模型能力有限**的场景；Agentic 适合**灵活探索、大模型驱动、需要自适应**的场景。当前代码的两者并存 + `PipelineRouter.compare` 对比模式，正好覆盖了从生产到实验的全谱需求。
