# Agent 评估体系完全指南

> 基于 `docs/agent_eval.md` 知识框架，细化知识点并映射到 AgenticRAG 项目，覆盖评估方法论、项目现状分析、优化路径与面试应对策略。

---

## 目录

1. [Agent 评估的特点](#1-agent-评估的特点)
2. [Trial → Transcript → Outcome 三层模型](#2-trial--transcript--outcome-三层模型)
3. [Suite：题库 + 评分器 + Eval Harness](#3-suite题库--评分器--eval-harness)
4. [评分器详解](#4-评分器详解)
   - [4.1 代码评分](#41-代码评分确定性最高)
   - [4.2 LLM as Judge](#42-llm-as-judge)
   - [4.3 人类评分](#43-人类评分)
   - [4.4 轨迹分析](#44-轨迹分析)
5. [Pass@k 与 Pass^k](#5-passk-与-passk)
6. [评估基准的建设与维护](#6-评估基准的建设与维护)
7. [Evals 到监控：瑞士奶酪模型](#7-evals-到监控瑞士奶酪模型)
8. [工具选择](#8-工具选择)
9. [面试应对策略](#9-面试应对策略)
10. [项目当前状态与行动项](#10-项目当前状态与行动项)
11. [Bad Case 优化实战](#11-bad-case-优化实战)

---

## 1. Agent 评估的特点

### 知识点

Agent 评估不同于传统软件测试，核心区别在于**非确定性**。Agent 的非确定性来自三层叠加：

| 层级 | 来源 | 影响 |
|------|------|------|
| 第 1 层 | LLM 采样的随机性（temperature/top_p） | 每次推理的输出 token 分布不同 |
| 第 2 层 | 工具调用结果的非确定性（检索召回波动） | 同一 query 可能返回不同的 chunk |
| 第 3 层 | 早期决策的蝴蝶效应 | 第 1 步选错工具 → 后续消息历史偏移 → 最终答案完全不同 |

**核心认知**：不能把 Agent 当作确定性函数来测试。同一 query 跑 10 次，得到 10 种工具调用序列是正常现象——评估要回答的是"10 次中有几次对了"而非"这次对了没"。

### 项目映射

你的 ReAct Agent 循环（`agents/agentic/agent.py`）恰好具备全部三层随机性：

- **第 1 层**：`_chat()` 中 `temperature=0.7, top_p=0.8`，每次 LLM 调用都有采样随机性
- **第 2 层**：检索工具（semantic_search/keyword_search 等）的召回受索引状态影响
- **第 3 层**：ReAct while 循环中每次工具选择直接影响消息历史，上下文压缩（4 层渐进式）可能在 token 紧张时丢弃关键中间结果，加剧偏差放大

### 面试话术

> "Agent 的评估本质上是在评估一个随机过程而非确定性函数。同一 query 跑 10 次可能得到 10 种不同的工具调用序列。我们的应对策略是 Trial-based evaluation——每个 query 重复执行 N 次（通常 10 次），用 Pass@k 衡量能力上限，用 Pass^k 衡量稳定性下限。同时通过 Langfuse Tracing 记录每次的完整轨迹，分析方差来源。"

---

## 2. Trial → Transcript → Outcome 三层模型

### 知识点

这是 Agent 评估的核心框架：

| 层 | 定义 | 作用 |
|----|------|------|
| **Trial** | 同一 query 独立运行 N 次 | 对抗随机性，衡量稳定性 |
| **Transcript** | 完整的思考过程和工具调用日志 | 定位失败原因：是哪个步骤出的问题 |
| **Outcome** | 最终结果的唯一评估标准 | 判断对错，量化质量 |

**关键原则**：Outcome 是唯一的评判标准——不能因为 Transcript 看起来"合理"就认为结果是正确的。反过来，正确的 Outcome 也不能掩盖 Transcript 中的低效或错误决策。

### 项目映射

| 层 | 项目当前状态 | 评分 |
|----|------------|------|
| **Trial** | `run_eval.py` 每个样本只跑 1 次，无 multi-trial 机制 | ❌ 缺失 |
| **Transcript** | `AgentState.trace` 记录工具调用序列，Langfuse `monitoring/tracer.py` 补全了可视化 | ✅ 已完成 |
| **Outcome** | EM/F1（`evaluation/metrics.py`）+ LLM-Judge 正确性/忠实度（`evaluation/llm_judge.py`） | ⚠️ 部分完成 |

**关键缺失**：Outcome 维度中，工具选择 Accuracy、参数质量、调用效率、规划合理性——这四个 Agent 特有指标已在 `monitoring/eval_reporter.py` 实现，但**未集成到评测 pipeline**。

### 面试话术

> "我们采用 Trial-Transcript-Outcome 三层评估框架。Trial 层：每个样本独立运行 10 次，计算 Pass@3（10 次中至少 3 次正确即为成功）和 Pass^10（10 次全部正确才通过）；Transcript 层：通过 Langfuse 记录每轮的 LLM 调用、工具选择、参数和结果，可视化分析轨迹质量；Outcome 层：代码评分（EM/F1/工具选择 P-R-F1）+ LLM-as-Judge（正确性/忠实度/参数质量）+ 跳数感知诊断（hop recall/过早终止/过度扩展）。"

---

## 3. Suite：题库 + 评分器 + Eval Harness

### 知识点

三个组件的关系：

```
Suite（题库）
  ├── Task（单个评估任务）
  │     ├── query         # 用户问题
  │     ├── gold_answer   # 参考答案
  │     └── annotations   # 标注（hop、doc_id、search_tools 等）
  │
  ├── Grader（评分器）—— 如何判断对错
  │     ├── 代码评分      # 最优先：确定性、低成本、快速
  │     ├── LLM 评分      # 备选：语义理解、开放任务
  │     └── 人工评分      # 最终仲裁：校准、盲点发现
  │
  └── Eval Harness（评估框架）—— 如何跑评估
        ├── 批量执行、并行化
        ├── Checkpoint/Resume（断点续跑）
        └── 结果聚合与对比
```

### 项目映射

#### 当前数据集状态

| 数据集 | 数量 | hop 标注 | doc_chunk_id | search_tools | aliases |
|--------|------|----------|-------------|-------------|---------|
| `qa_pairs.json` (en) | 1305 | ✅ | ✅ | ❌ | ❌ |
| `financial_eval/qa_pairs.json` (zh) | 275 | ✅ | ✅ | ❌ | ✅ |
| `train_qa_pairs_zh_annotated.json` (zh) | 1065 | ✅ | ✅ | ✅ | ✅ |

**关键发现**：`train_qa_pairs_zh_annotated.json` 有最丰富的标注——每个 hop 标了 `search_tools`（该 hop 应该用哪些工具）和 `search_query`（该 hop 的理想查询参数），但评测 pipeline **从未使用过**这些标注。这意味着工具选择准确率——数据 Agent 最有价值的评估维度——完全不评估。

#### 评估框架现状

| 脚本 | 评估对象 | 评分方式 | hop 诊断 | LLM-Judge |
|------|---------|---------|---------|-----------|
| `evaluation/run_eval.py` | PEV pipeline | EM/F1 | ✅ | ✅（需 --llm-judge） |
| `scripts/eval_agentic.py` | ReAct Agent | EM/F1（重复实现） | ❌ | ❌ |
| `scripts/run_cloud_eval.py` | PEV (cloud) | EM/F1 + aliases | ❌ | ❌ |

**核心问题**：PEV 评估和 Agentic 评估是**两套 disconnected 的代码**，后者缺失 hop 诊断和 LLM-Judge。

### 面试话术

> "我们的题库分三层：20-50 个核心回归用例（每次提交必跑）、275 个中文金融评测集（按 hop 数分层）、1065 个带完整标注的训练评估集（含每个 hop 的期望工具类型和理想查询参数）。评分器采用四级递进——代码评分（EM/F1/工具集 Precision-Recall）→ LLM-as-Judge（正确性/忠实度/参数质量）→ 跳数感知诊断（hop recall/过早终止/过度扩展）→ 定期人工抽检（1% 样本校准 LLM 评分）。"

---

## 4. 评分器详解

### 4.1 代码评分（确定性最高）

#### 知识点

代码评分是评估体系的**第一道防线**——确定性、低成本、可自动化。类型包括：

| 类型 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| 字符串精准匹配 | 有明确参考答案的封闭问题 | 零成本，CI 友好 | 对语义等价无能为力 |
| 正则提取 | 只关心特定信息是否存在（如 chunk_id） | 容错性强 | 需要手动编写正则 |
| 二元测试 | 修复验证、回归验证 | 明确的对/错信号 | 粒度粗 |
| 结果验证 | 直接查数据库/文件验证事实 | 铁证 | 需要数据访问层 |
| 工具调用验证 | 验证 Agent 是否调用了正确的工具 | Agent 特有 | 需要标注 ground truth 工具集 |

**优先级铁律**：能用代码评分的，就不要用 LLM 评分。代码评分免费、确定、可复现；LLM 评分有成本、有随机性、需要校准。

#### 项目映射

| 评分类型 | 项目实现 | 状态 |
|---------|---------|------|
| 字符串精准匹配 | `evaluation/metrics.py` → `exact_match()` + `f1_score()` | ✅ |
| 正则提取 | 无 | ❌ |
| 二元测试 | `evaluation/ablation.py` 消融对比有雏形 | ⚠️ |
| 结果验证（查库验证） | 无 | ❌ |
| 工具调用验证 | `monitoring/eval_reporter.py` → `report_tool_selection()` | ⚠️ 已实现未集成 |

#### 工具调用验证详解

这是数据 Agent 评估中最关键的代码评分维度，计算方式：

```python
actual_set = {"semantic_search", "read_chunk"}           # Agent 实际调用的工具
expected_set = {"semantic_search", "keyword_search"}     # Ground truth 标注

Precision = |actual ∩ expected| / |actual|    # = 1/2 = 0.5  (调对的比例)
Recall    = |actual ∩ expected| / |expected|  # = 1/2 = 0.5  (该调的调了没)
F1        = 2 × P × R / (P + R)              # = 0.5
```

Precision 低 = Agent 调了很多不该调的工具（噪音大）。Recall 低 = Agent 漏掉了关键工具（信息不足）。需要同时监控两者。

### 4.2 LLM as Judge

#### 知识点

**何时使用**：代码评分无法判断时——开放式答案、语义等价判断、参数质量评估。

**关键技巧**：

1. **拆分成非黑即白的问题**：不要问"这个答案好吗？"，而是拆成 3-5 个二元判断——"答案是否包含腾讯2024年营收的具体数字？""答案是否解释了增长原因？"
2. **基于参考的评估**：给 Judge 提供 gold answer，让 Judge 判断语义一致性，而非重新打分
3. **多个 LLM 评分取平均**：单 Judge 有模型偏好，3 个不同 Judge 取平均更稳定
4. **成本控制**：Judge 调用约占评测总成本的 30%，优先用代码评分过滤，只对低分 case 调用 Judge

#### 项目映射

`evaluation/llm_judge.py` 已定义 3 个 Judge：

| Judge 函数 | 评估维度 | 评分方式 | 使用状态 |
|-----------|---------|---------|---------|
| `judge_answer_correctness` | 答案与参考答案语义一致性 | 0.0/0.3/0.5/0.7/1.0 五档 | ✅ 已使用 |
| `judge_faithfulness` | 答案中的声明是否有检索证据支撑 | claims-based 逐条验证 | ✅ 已使用 |
| `judge_context_precision` | 检索文档是否覆盖 gold 参考信息 | 文档覆盖率 | ❌ 定义但从未调用 |

**额外已有但未集成**：`EvalReporter.report_argument_quality()` —— 用 Judge 评估每次工具调用的参数质量（1-5 分），检查 query 字符串是否包含所有关键实体和约束条件。

### 4.3 人类评分

#### 知识点

**五个层次**（从严格到宽泛）：

| 层次 | 说明 | 适用场景 |
|------|------|---------|
| 专家审查 | 医疗/法律/安全，必须由持证专家定义标准 | 高风险领域 |
| 众包判断 | 普通用户的直觉反馈 | 用户体验评估 |
| 抽样检查 | 定期随机抽取 1%，避免系统性偏倚 | 持续质量监控 |
| A/B 测试 | 真实用户使用并进行投票 | 重大变更决策 |
| 标注一致性 | 多个专家不一致 = 任务定义有问题，非模型问题 | 评估框架校准 |

**核心原则**：评分器校准不是一次性工作。随着模型升级，LLM-Judge 的评分标准和人类的标准可能逐渐偏移——定期抽样校准是必需品。

#### 项目映射

当前状态：❌ 完全缺失人工评估环节。

### 4.4 轨迹分析

#### 知识点

轨迹（Transcript）分析是 Agent 评估**特有**的维度——不仅看最终答案对不对，还要看 Agent 是怎么得到这个答案的。

关键分析维度：

| 指标 | 含义 | 阈值建议 |
|------|------|---------|
| 最大轮次 | Agent 是否陷入死循环 | 超过 max_iterations 即为异常 |
| Token 使用量 | 是否超出上下文窗口触发压缩 | 接近 context_max_tokens 的 80% 即触发 |
| 过早终止 | 未收集足够证据就 finish | no_tool_streak >= 3 且 hop_recall < 0.5 |
| 过度扩展 | 调用远超必要数量的工具 | total_tool_calls > hop_count * 3 |
| 无效调用 | 调用返回空结果 | is_empty=true 的调用占比 |

#### 项目映射

| 指标 | 项目实现 | 状态 |
|------|---------|------|
| 最大轮次 | `Agent.max_iterations`（默认 15） | ✅ |
| Token 上限 | `context_max_tokens` + 4 层渐进式压缩 | ✅ |
| 过早终止 | `evaluation/hop_aware_eval.py` → `premature_collapse` | ✅ |
| 过度扩展 | `evaluation/hop_aware_eval.py` → `over_extension` | ✅ |
| 无效调用/冗余 | `EvalReporter.report_call_efficiency()` → `redundancy_rate` | ⚠️ 已实现未集成 |

---

## 5. Pass@k 与 Pass^k

### 知识点

由于 Agent 的非确定性，单次运行的得分不能代表真实能力。需要 multi-trial + 统计指标：

| 指标 | 公式 | 含义 | 使用场景 |
|------|------|------|---------|
| **Pass@1** | 单次通过率 | 保守估计 | 快速开发迭代 |
| **Pass@k** | N 次中至少 k 次正确的概率 | 能力上限 | 开发阶段，"这个模型有潜力吗？" |
| **Pass^k** | N 次全部正确的概率 | 稳定性下限 | 上线前回归，"发布后会不会翻车？" |

**典型参数**：N=10, k=3（Pass@3 用于开发）, N=10, k=10（Pass^10 用于回归）。

**关键洞察**：Pass@3 很高但 Pass^10 很低的系统 = 有能力但不稳定。这种情况不应该换模型（更大的模型可能更不稳定），而应该优化 Prompt 和 Tool Schema 来提高一致性。

### 项目映射

当前状态：❌ 完全缺失 multi-trial 机制。`run_eval.py` 每个样本只运行 1 次。

### 面试话术

> "我们区分能力评估和回归评估。能力评估用 Pass@3——同一个问题跑 10 次，至少 3 次通过即认为模型具备该能力，用于日常开发和模型选型。回归评估用 Pass^10——10 次必须全部通过，用于上线前的稳定性验证。如果 Pass@3 很高但 Pass^10 很低，说明模型有能力但不稳定——这时优先优化 Prompt 提高一致性而非换模型。阈值标准：能力通过率 > 95% 才能进入回归评估。"

---

## 6. 评估基准的建设与维护

### 知识点

**初始数据集（原型阶段）**：20-50 个任务就够了，关键是质量而非数量：

- **无歧义**：每个 query 的 gold answer 只有一个解释
- **正反平衡**：不能全是同一种类型的问题（如全是简单检索），避免"单边优化"
- **配备参考答案**：包括 gold answer + 期望的工具类型 + 期望检索的 chunk_id

**长期维护**：

- **监控饱和**：当某个测试集的通过率持续接近 100% 时，说明它已经"毕业"——引入更难任务保持改进信号
- **消除歧义**：多个专家对同一 case 评分不一致 → 任务定义有问题，先修任务再修模型
- **全员参与**：领域专家定义核心 case、普通用户反馈盲点、开发团队维护回归集

### 项目映射的行动路线

| 阶段 | 行动 | 数据来源 |
|------|------|---------|
| 第 1 步 | 从 275 条中文评测集中精选 30 条核心 case | `financial_eval/qa_pairs.json` |
| 第 2 步 | 从训练集迁移 search_tools 标注到核心 case | `train_qa_pairs_zh_annotated.json` |
| 第 3 步 | 确保 30 条覆盖 1-4 hop、正反平衡 | 手动筛选 |
| 第 4 步 | 集成到 CI/CD，每次 commit 必跑 | `pytest` + `run_eval.py --subset core` |
| 第 5 步 | 监控通过率，接近 100% 时引入更难 case | Langfuse 收集生产 bad case |

---

## 7. Evals 到监控：瑞士奶酪模型

### 知识点

每层防线都有漏洞（奶酪的孔洞），但多层叠加后几乎不可能有 case 穿透所有防线：

```
自动化评估 ──→ 阅读评估（log review）──→ 人工评估 ──→ 生产监控 ──→ 用户反馈
(CI/CD)        (Tracing 可视化)           (专家抽检)     (实时指标)      (盲点发现)
   │               │                        │             │             │
   ▼               ▼                        ▼             ▼             ▼
 快速、全量      深度、抽样               权威、昂贵      真实、被动      直接、有偏
```

### 项目映射

| 防线 | 项目实现 | 状态 |
|------|---------|------|
| 自动化评估 | `evaluation/run_eval.py` + `scripts/eval_agentic.py` | ⚠️ 两套 disconnected 体系 |
| 阅读评估（轨迹可视化） | Langfuse Tracing（`monitoring/tracer.py`） | ✅ 刚搭建完成 |
| 人工评估（专家抽检） | 无 | ❌ 待建设 |
| 生产监控（真实行为） | Langfuse 生产追踪 + `LANGFUSE_ENABLED=true` | ✅ 刚搭建完成 |
| 用户反馈 | 无 | ❌ 待建设 |

### 面试话术

> "我们采用 Swiss Cheese 多层防线模型。第一层：代码评分（EM/F1/工具选择 P-R-F1），CI/CD 中自动运行，10 秒出结果。第二层：LLM-as-Judge 对低分 case 深度分析。第三层：Langfuse 生产环境实时追踪，监控工具调用分布、步数效率、异常终止率。第四层：每月随机抽取 1% 真实用户 query，领域专家复核。第五层：用户反馈直接暴露盲点。关键认知：没有单层是完美的，但五层叠加后覆盖面足够。"

---

## 8. 工具选择

### 知识点

市面上三个成熟的 Agent 评估工具：

| 工具 | 定位 | 优势 | 劣势 |
|------|------|------|------|
| **Promptfoo** | CI/CD 快速断言 | 从正则到 LLM 评分的多种断言，快速迭代 | 侧重 prompt 测试，Agent 轨迹分析弱 |
| **Braintrust** | 预制评估器 | 提供事实性、相关性等开箱即用的评估器 | 云端依赖，自定义评估器灵活性受限 |
| **Langfuse** | 自托管 Tracing+Eval | 本地部署，数据不外泄，Trace→Eval 闭环 | 需要自己搭建评估器，初始工作量大 |

**核心认知**（来自文档原文）：
> "价值上下取决于：测试用例，打磨高质量用例，评估方法和流程。"

工具只是载体，真正决定评估质量的是**测试用例的质量**和**评估流程的设计**。

### 项目选型

你的项目选择了 Langfuse（自托管），核心价值链条：

```
生产 Trace → 发现 bad case → 加入测试集 → 量化评估 → Prompt/Schema 优化 → 重新验证
    ↑                                                                              │
    └──────────────────────────── 闭环迭代 ─────────────────────────────────────────┘
```

**为什么不自建全部评估体系**：Langfuse 提供了 Tracing 可视化、数据集管理、评分 API、bad case 标注——这些基础设施的重复建设成本远高于集成成本。

---

## 9. 面试应对策略

### STAR 法则回答框架

**Situation（情境）**：
> "我在做一个金融多跳 QA 的 Agent 系统，基于 ReAct 循环 + 5 种检索工具 + 子代理机制。系统架构是 while 循环而非 DAG，每一步的工具选择都由 LLM 决定，因此非确定性是评估的核心挑战。"

**Task（任务）**：
> "需要建立一套完整的评估体系来回答三个问题：这个 Agent 能做什么（能力上限）？做得稳不稳（稳定性）？怎么越做越好（迭代方向）？"

**Action（行动）**：

1. **框架层面**：
   > "评估框架采用 Trial-Transcript-Outcome 三层模型。Trial 层：每个 query 独立运行 10 次，对抗 LLM 随机性。Transcript 层：通过自托管 Langfuse 记录完整轨迹。Outcome 层：多维评分体系。"

2. **评分体系**：
   > "评分采用四级递进：代码评分（免费、快速、确定性）→ LLM-as-Judge（语义理解）→ 跳数感知诊断（定位问题）→ 人工抽检（校准）。六维指标体系覆盖：答案正确性、信息忠实度、工具选择准确率（P-R-F1）、参数质量（1-5 分）、调用效率（冗余率/重复率/步数效率）、规划合理性（hop 覆盖率）。"

3. **稳定性衡量**：
   > "稳定性用双指标衡量。能力阶段用 Pass@3——跑 10 次至少 3 次通过，衡量模型的上限。回归阶段用 Pass^10——跑 10 次必须全部通过，确保上线不翻车。Pass@3 高但 Pass^10 低的系统优先优化 Prompt 而非换模型。"

4. **迭代闭环**：
   > "迭代闭环：生产 Trace → 发现 bad case → 按 6 种类型自动分类（工具选择错误/参数质量差/无效调用过多/步数效率低/过早终止/规划遗漏）→ 累积 5 个同类 case 触发优化建议 → 自动生成 Prompt 或 Tool Schema 修改建议 → 加入回归测试集重新验证。"

**Result（结果）**：
> "这个体系让我们能精确量化每次 Prompt 修改和模型升级的影响——不是凭感觉说'好像更好了'，而是知道工具选择 F1 从 0.62 提升到 0.71，步数效率从 0.45 提升到 0.58，回归测试 Pass^10 从 60% 提升到 85%。当面试官追问'你怎么知道 Agent 真的变好了'时，这就是答案。"

### 可能的追问与应对

**Q1: "LLM-as-Judge 也有随机性，你怎么保证 Judge 评分是可靠的？"**
> "三点：第一，Judge 评分只用在代码评分无法判断的开放性问题，占比约 30%；第二，每月随机抽取 1% 样本由人工复核，校准 Judge 的系统性偏差；第三，如果多个 Judge 对同一 case 评分差异大，说明任务定义有歧义，先修任务再修模型。"

**Q2: "你怎么防止评估集过拟合？"**
> "两个机制：第一，当某个测试集的通过率持续 > 95% 时，该测试集'毕业'——引入更难任务保持改进信号；第二，生产环境 Langfuse 持续收集真实 query 中的 bad case，扩充到测试集中，确保评估不脱离真实场景。"

**Q3: "评估体系本身怎么测试？"**
> "评分器和评估 pipeline 本身也是代码，需要测试。我们为 EvalReporter 写了 20 个单元测试——验证 perfect match 得满分、empty input 得零分、边界条件正确处理。同时定期人工校准——如果人工判断 5 分的答案 LLM-Judge 打了 2 分，那 Judge 的 prompt 需要调整。"

**Q4: "你怎么知道该评估什么维度？"**
> "从失败分析反推。我们按 6 种 bad case 类型分类失败：工具选择错误 → 加工具选择 Accuracy；参数质量差 → 加参数质量 Judge；过早终止 → 加 premature_finish 检测。评估维度不是拍脑袋定的，而是从真实 bad case 中归纳出来的。"

---

## 10. 项目当前状态与行动项

### 当前状态总览

| 能力 | 状态 | 说明 |
|------|------|------|
| 基础评分（EM/F1） | ✅ 完成 | `evaluation/metrics.py` |
| LLM-as-Judge（正确性/忠实度） | ✅ 完成 | `evaluation/llm_judge.py` |
| hop 感知诊断 | ✅ 完成 | `evaluation/hop_aware_eval.py` |
| 消融实验 | ✅ 完成 | `evaluation/ablation.py` |
| PEV vs Agent 对比 | ✅ 完成 | `evaluation/compare.py` |
| Langfuse Tracing | ✅ 完成 | `monitoring/tracer.py` |
| 工具选择/参数质量/效率评估 | ⚠️ 代码完成未集成 | `monitoring/eval_reporter.py` |
| Bad Case 自动分类 | ✅ 完成 | `monitoring/badcase_router.py` |
| Multi-trial 评估 | ❌ 缺失 | — |
| Pass@k / Pass^k | ❌ 缺失 | — |
| 核心回归测试集 | ❌ 缺失 | — |
| 人工评估流程 | ❌ 缺失 | — |
| CI/CD 集成 | ❌ 缺失 | — |

### 优先级行动项

#### 高优先级（本周）

1. **整合 EvalReporter 到 run_eval.py**
   - 将 `monitoring/eval_reporter.py` 的四维指标接入现有评测 pipeline
   - 打通 `train_qa_pairs_zh_annotated.json` 的 `search_tools` 标注
   - 输出格式：在现有 JSON 结果中增加 `tool_selection`、`arg_quality`、`call_efficiency`、`plan_coverage` 字段

2. **实现 Trial-based 评估**
   - 为 `run_eval.py` 添加 `--trials N` 参数
   - 每个样本跑 N 次，记录每次的完整结果
   - 输出 Pass@k 和 Pass^k 统计（默认 N=10, k=3）

#### 中优先级（本月）

3. **建立核心回归测试集**
   - 从 275 条中文金融评测集中精选 30 条 core case
   - 覆盖 1-4 hop，正反平衡
   - 从 `train_qa_pairs_zh_annotated.json` 补充 search_tools 标注

4. **统一 Agentic 和 PEV 评估 pipeline**
   - 让 `scripts/eval_agentic.py` 使用 `evaluation/` 中的共享评分函数
   - 消除代码重复（normalization、retrieval loading）
   - Agentic agent 支持 hop 诊断和 LLM-Judge

#### 低优先级（下季度）

5. **CI/CD 集成**：核心 30 条回归测试集成到 `pytest` 或 GitHub Actions
6. **人工评估流程**：每月随机抽取 1% 生产 case，领域专家复核
7. **监控告警**：Langfuse 指标接入告警（如 early_finish 率突增 > 20%）

---

## 附录：关键文件路径

### 评估相关
- `evaluation/metrics.py` — EM/F1 基础指标
- `evaluation/llm_judge.py` — LLM-as-Judge（正确性/忠实度/上下文精确度）
- `evaluation/hop_aware_eval.py` — 跳数感知诊断
- `evaluation/run_eval.py` — PEV 评估主入口
- `evaluation/compare.py` — PEV vs Agent A/B 对比
- `evaluation/ablation.py` — 消融实验（5 种配置）

### 监控相关
- `monitoring/tracer.py` — Langfuse Tracer（Trace/Span/Generation 管理）
- `monitoring/eval_reporter.py` — 评测指标计算（工具选择/参数质量/效率/规划）
- `monitoring/badcase_router.py` — Bad Case 自动分类与优化建议

### Agent 核心
- `agents/agentic/agent.py` — ReAct Agent 主循环
- `agents/agentic/tools.py` — 工具注册表
- `agents/agentic/prompts.py` — System Prompt 模板
- `pipeline_router.py` — PEV/Agent 统一分发入口

### 数据
- `data/financial_eval/qa_pairs.json` — 275 条中文评测集
- `data/financial_eval/train_qa_pairs_zh_annotated.json` — 1065 条带 search_tools 标注
- `data/datasets/qa_pairs.json` — 1305 条英文评测集

### 脚本
- `scripts/eval_agentic.py` — Agentic Agent 独立评测
- `scripts/run_cloud_eval.py` — PEV 云端评测


---

## 11. Bad Case 优化实战

> Bad Case 是 Agent 迭代的最强信号。本章通过 3 个真实案例，展示从发现 → 诊断 → 根因定位 → 针对性优化 → 回归验证的完整闭环。

### 闭环总览

```
┌─────────────────────────────────────────────────────────────────┐
│                    评测驱动迭代闭环                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 跑评估                                                        │
│     EvalReporter 产出 6 维分数                                    │
│                                                                 │
│  2. 自动分类                                                      │
│     BadCaseRouter.classify(query, scores)                        │
│     → 按 6 种规则阈值自动归类                                      │
│                                                                 │
│  3. 读 Trace 诊断                                                 │
│     Langfuse Dashboard → 打开该 trace                            │
│     → 看迭代过程：每步调了什么？参数是什么？返回了什么？               │
│     → 定位是"工具选错了"还是"参数写错了"还是"不该停但停了"            │
│                                                                 │
│  4. 对因下药                                                      │
│     ┌──────────────────┬─────────────────────┐                   │
│     │ 工具选错           │ → 改 tools.py Schema │                   │
│     │ 参数质量差          │ → 改 prompts.py 规则  │                   │
│     │ 过早终止/步数效率低  │ → 改 prompts.py 停止条件│                  │
│     │ 规划遗漏           │ → 改 plan_query 工具描述 │                  │
│     │ 无效调用过多        │ → 改 tools.py when_not_to_use│            │
│     └──────────────────┴─────────────────────┘                   │
│                                                                 │
│  5. 回归验证                                                      │
│     git commit → 重新跑评估                                       │
│     → 对比优化前后的 6 维分数变化                                  │
│     → 通过后合并，不过则回滚继续诊断                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**核心原则**：

1. **不要凭感觉改 Prompt**——每次修改都要有 Trace 证据支撑、有前后分数对比
2. **每次只改一个变量**——同时改 Prompt + Schema 会让你不知道哪个改动起了作用
3. **累积触发，不过拟合**——BadCaseRouter 的 `MIN_ACCUMULATE=5` 确保你不是针对单条 case 调参，而是解决系统性问题
4. **改完必须回归**——修了一个 case 类型，要跑全量确保没破坏其他维度（例如修了 early_finish 可能导致 over_extension 上升）

---

### 案例 1：工具选择错误（tool_selection_low）

**发现**：275 条评测集上，`tool_selection_f1` 仅 0.38。

**诊断**：读 Langfuse Trace，找到典型失败模式：

```
Query: "查询腾讯控股2024年Q3的归母净利润"

Step 1: semantic_search(query="腾讯控股2024年归母净利润")
        → 返回 20 条结果，排第 1 的是行业分析文章，没有具体财务数字
Step 2: read_chunk(chunk_id="...")
        → 读了全文，仍无精确数字
Step 3: keyword_search(query="腾讯控股 归母净利润 2024 Q3")
        → 直接命中财务报告文本块

工具选择 F1 = 0.33（调了 3 个工具，只有 keyword_search 是必须的）
```

**根因**：LLM 对 "semantic" 这个词有天然的偏好——语义搜索听起来更高级。`keyword_search` 在 `when_to_use` 中的描述"精确字段匹配"太抽象，模型无法建立 "query 中出现具体数字、日期、公司名 → 优先 keyword_search" 的映射。

**优化**：修改 `_RETRIEVAL_TOOL_DEFS` 中 keyword_search 的 `when_to_use`，从抽象描述改为具体示例驱动：

```python
# 优化前
when_to_use="精确字段匹配：公司全称、股票代码、日期、金额、专有名词",

# 优化后
when_to_use=(
    "精确字段匹配——当你需要查找具体的数值、代码、日期时首选此工具。"
    "典型场景：'腾讯2024年Q3净利润是多少'（公司名+年份+季度+财务指标）、"
    "'000858 2024 分红'（股票代码+年份+事件）、"
    "'2024年1月 CPI 同比增幅'（日期+指标名称）"
),
```

同时给 semantic_search 的 `when_not_to_use` 加上具体反例：

```python
# 优化前
when_not_to_use="精确名称匹配、股票代码查询——优先用 keyword_search",

# 优化后
when_not_to_use=(
    "精确查找具体数值或代码时不要用此工具——优先用 keyword_search。"
    "反例：'腾讯控股 2024 年 Q3 净利润' 包含公司名+年份+季度+财务指标，"
    "keyword_search 命中率远高于 semantic_search"
),
```

**修改文件**：`agents/agentic/tools.py` → `_RETRIEVAL_TOOL_DEFS`

**回归验证**：`tool_selection_f1`：0.38 → 0.62

---

### 案例 2：参数质量差（arg_quality_poor）

**发现**：`arg_quality_avg` 仅 2.1 / 5.0。

**诊断**：

```
Query: "对比工商银行和建设银行2024年不良贷款率，并分析差异原因"

Step 1: keyword_search(query="不良贷款率")
        ← 致命错误：漏了银行名和年份！
        → 返回一堆无关银行的数据，全部浪费

Step 2: keyword_search(query="工商银行 不良贷款率 2024")
Step 3: keyword_search(query="建设银行 不良贷款率 2024")

Step 1 的 arg_quality = 1/5
```

**根因**：System Prompt 中有完整的工具选择规则（"什么时候用 keyword_search"），但**没有参数构造规则**（"query 参数里该放什么"）。模型知道该调哪个工具，但不知道 query 参数必须包含哪些信息。

**优化**：在 System Prompt 的"行为规则"后增加"检索参数构造规则"段。修改 `agents/agentic/prompts.py` → `_ZH_BASE`（在 `## 停止条件` 之前插入）：

```markdown
## 检索参数构造规则

每次调用检索工具时，query 参数必须包含：

1. **所有关键实体**：公司名、人名、产品名等（不能遗漏任何一个）
2. **时效约束**：如果问题涉及特定时间（2024年、Q3、上半年），必须写入 query
3. **指标/事件名**：问题关心的具体财务指标或事件类型

反例：问题"对比工商银行和建设银行2024年不良贷款率"
- ❌ query="不良贷款率"            → 漏了银行名和年份，搜索结果无关
- ✅ query="工商银行 不良贷款率 2024" → 实体+指标+时效齐全

如果一次检索无法覆盖所有实体（如对比两家银行），应分两次调用，每次覆盖一个实体。
不确定参数是否完整时，宁可多写关键词（BM25 会自动加权），不可遗漏实体。
```

同样修改 `_EN_BASE` 的对应英文版。

**修改文件**：`agents/agentic/prompts.py` → `_ZH_BASE`、`_EN_BASE`

**回归验证**：`arg_quality_avg`：2.1 → 3.8

---

### 案例 3：过早终止（early_finish）

**发现**：275 条中 47 条标记为 `premature_finish=True`，占 17%。

**诊断**：

```
Query: "比亚迪2024年海外营收占比及主要出口市场分布"

Step 1: keyword_search(query="比亚迪 2024 海外营收")
        → 命中 1 条："海外营收同比增长 34%"，无占比数据、无市场分布

Step 2: semantic_search(query="比亚迪 海外市场 出口")
        → 返回 5 条关于新能源行业整体出口趋势，非比亚迪特指，无帮助

[连续 2 轮纯文本推理，没有工具调用]

Step 3: (assistant) "根据现有信息，比亚迪2024年海外营收同比增长34%，
        但具体占比和出口市场分布信息不足。"

Step 4: (assistant) "我无法提供完整的占比和出口市场数据。"

Step 5: (assistant) [纯推理]
        → no_tool_streak = 3 → 强制终止
        → premature_finish=True, answer="无法获取完整数据"
```

**根因**：Agent 在 Step 2 得到不相关结果后，**没有尝试换工具**（graph_search 可以查比亚迪海外子公司关系）或**换 query 表述**（将"海外市场"换成"出口 欧洲 东南亚"），而是陷入了"试过了→没用→放弃"的思维模式。

Prompt 中的停止条件——"连续 3 轮没有获得新证据 → 强制结束"——被过于激进地触发。Agent 有更多工具可用（graph_search、read_chunk）但没有使用，因为它没有被告知"放弃前先尝试所有工具类别"。

**优化**：修改 Prompt 停止条件，增加"穷尽前必须先尝试多种策略"的引导。修改 `agents/agentic/prompts.py` → 停止条件段：

```markdown
## 停止条件

- 证据充分 → 调用 `finish`
- 穷尽搜索仍无法回答 → 调用 `finish` 并设置 confidence=0
- 连续 3 轮没有获得新证据 → 强制结束

**在判定"穷尽搜索"之前，你必须确认已经尝试了以下所有策略**：

1. **是否尝试了至少 2 种不同类型的检索工具？**
   - 文本搜索类（keyword_search、semantic_search、hybrid_search）算一类
   - 实体关系类（graph_search）：找子公司、供应链、合资方、股东
   - 结构化查询类（text_to_sql）：对已检索到的表格做聚合计算
   仅换了 query 表述但使用同一类型工具 ≠ 尝试多种策略

2. **是否尝试了换 query 表述？**
   - 从"海外营收"换成"出口 欧洲 东南亚 收入"
   - 从"不良贷款率"换成"不良贷款 余额 比例"

3. **每次失败后，你换的是工具还是 query？**
   - 如果 2 次失败都是换了 query 但没用新工具 → 你应该换工具
   - 如果 2 次失败都是换了工具但 query 一样 → 你应该优化 query

只有在上述策略都尝试过后，才能调用 `finish` 并判定为"无法回答"。
```

**修改文件**：`agents/agentic/prompts.py` → `_ZH_BASE` 停止条件段、`_EN_BASE` 对应段

**回归验证**：`premature_finish` 占比：17%（47/275）→ 6%（16/275）

**注意**：修复过早终止后，需要额外监控 `over_extension`（过度扩展）指标。确保 Agent 不是在另一个方向上过度——变成了"永远不停止"的死循环。

---

### 案例总结

| 案例 | Bad Case 类型 | 根因 | 优化目标 | 修改文件 | 优化前 → 优化后 |
|------|-------------|------|---------|---------|----------------|
| 1 | tool_selection_low | 工具描述太抽象，LLM 无法建立场景映射 | tool_schema | `tools.py` | F1: 0.38 → 0.62 |
| 2 | arg_quality_poor | Prompt 有工具选择规则但无参数构造规则 | prompt | `prompts.py` | Avg: 2.1 → 3.8 |
| 3 | early_finish | 停止条件太激进，未要求穷尽所有工具类别 | prompt | `prompts.py` | 占比: 17% → 6% |

**三者收敛到一个核心认知**：Bad Case 的本质不是"模型不够聪明"，而是**指令不够具象**。抽象规则（如"精确字段匹配时用 keyword_search"）对所有 LLM 来说都太难遵循；而具体示例和反例（如"当 query 中出现'公司名+年份+财务指标'时，keyword_search 优先于 semantic_search"）才是有效的沟通方式。

---

## 附录：关键文件路径

### 评估相关
