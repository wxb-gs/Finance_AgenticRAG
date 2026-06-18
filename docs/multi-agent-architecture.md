# 多 Agent 架构：主 Agent + 子 Agent 动态派发

日期: 2026-06-17

## 1. 核心设计：模型驱动的动态编排

本系统的"多 Agent"不是固定的 Agent 协作网络，而是 **一个主 ReAct Agent 按需动态派发子 Agent** 的架构。关键决策点——"是否拆分、拆几个、用什么类型、是否并行"——全部由模型在 ReAct 循环中自主判断，代码层面只提供工具和执行通道。

```
用户查询
  │
  ▼
┌──────────────────────────────────────────────┐
│  PipelineRouter (pipeline_router.py)          │
│  根据 mode 分发: agent / pev / compare        │
└──────────────────────────────────────────────┘
  │ (默认 mode="agent")
  ▼
┌──────────────────────────────────────────────┐
│  Agent (agents/agentic/agent.py)             │
│  ReAct 循环: Think → Act → Observe → ...     │
│                                              │
│  工具集 (tools.py):                           │
│  ┌─ 检索层 (6): semantic/keyword/graph/     │
│  │              hybrid/read_chunk/text_to_sql│
│  ├─ 元工具 (4): dispatch_subagent ──────┐   │
│  │              activate_skill           │   │
│  │              remember / plan_steps    │   │
│  └─ 生命周期 (1): finish                 │   │
│                                              │
│  上下文管理 (context.py): 三层压缩策略        │
│  记忆系统 (memory.py): 跨会话持久化           │
│  技能系统 (skills/): 领域工作流指引           │
└──────────────────────────────────────────────┘
                    │
                    │ dispatch_subagent
                    ▼
┌──────────────────────────────────────────────┐
│  SubAgentManager (sub_agent.py)              │
│                                              │
│  ┌─ retrieval 子Agent (小模型, 5轮上限)      │
│  ├─ comparison 子Agent (大模型, 8轮上限)     │
│  └─ computation 子Agent (小模型, 3轮上限)    │
│                                              │
│  支持 dispatch_parallel() 并行执行            │
└──────────────────────────────────────────────┘
```

## 2. 核心组件

| 组件 | 文件 | 职责 |
|------|------|------|
| `Agent` | `agents/agentic/agent.py:15` | ReAct 主循环，工具驱动，最大 15 轮迭代 |
| `ToolRegistry` | `agents/agentic/tools.py:249` | 三层工具注册表：检索(6) + 元(4) + 生命周期(1) |
| `SubAgentManager` | `agents/agentic/sub_agent.py:37` | 子 Agent 派发、并行执行、结果合并 |
| `SkillManager` | `agents/agentic/skills/loader.py:61` | 扫描 `SKILL.md`，模型自主激活，零 Python 改动 |
| `ContextManager` | `agents/agentic/context.py:29` | 按模型规格的三层压缩 |
| `MemoryManager` | `agents/agentic/memory.py:41` | jieba 分词召回，文件持久化 |

### 2.1 主 Agent 循环 (`agent.py`)

```python
# agent.py:33 — run() 方法核心流程
def run(self, query: str) -> AgentResult:
    # 0. MCP 初始化
    # 1. 召回相关记忆 (MemoryManager.recall)
    # 2. 组装 System Prompt (SkillManager.build_system_prompt)
    # 3. 构建消息列表 + 工具 schema
    # 4. ReAct 循环:
    while state.iterations < self.max_iterations and not state.finished:
        # 上下文压缩检查 (ContextManager)
        # LLM 调用 (OpenAI SDK native tool calling)
        # 工具调用分发:
        #   finish        → 直接结束
        #   dispatch_subagent → 创建子 Agent 实例运行
        #   activate_skill → 加载 SKILL.md 内容
        #   remember      → 写入文件持久化
        #   plan_steps    → 记录步骤计划
        #   检索工具      → 执行检索 + 自动记忆(confidence>0.8)
        # 停止条件: 连续 3 轮无工具调用 或 达到 max_iterations
    # 5. 兜底: _force_answer()
```

每个工具调用的处理逻辑在 `agent.py:99-200` 行。`finish` 和 `dispatch_subagent` 有特殊处理——`finish` 直接终止循环，`dispatch_subagent` 创建新的 Agent 实例来执行子任务。

### 2.2 子 Agent 类型 (`sub_agent.py`)

每种子 Agent 都是一个**受限的独立 Agent 实例**，通过 `agent_factory` 创建（`sub_agent.py:46`），拥有精简的工具集和独立的迭代上限：

| 类型 | 可用工具 | 迭代上限 | 模型规格 | 职责 |
|------|---------|---------|---------|------|
| `retrieval` | semantic_search, keyword_search, graph_search, read_chunk, finish | 5 | small | 聚焦检索，返回结构化结果，不做推理 |
| `comparison` | semantic_search, keyword_search, read_chunk, finish | 8 | large | 多源数据对比，标注矛盾点，输出对比表 |
| `computation` | finish | 3 | small | 纯数值计算，展示推导过程，不检索 |

```python
# sub_agent.py:12 — 子代理类型配置
SUBAGENT_TYPES = {
    "retrieval": SubAgentConfig(
        tools=["semantic_search", "keyword_search", "graph_search", "read_chunk", "finish"],
        max_iterations=5,
        system_prompt_override="你是信息检索专家。快速定位相关信息，返回结构化结果。不做深度分析推理。",
        model_hint="small",
    ),
    "comparison": SubAgentConfig(
        tools=["semantic_search", "keyword_search", "read_chunk", "finish"],
        max_iterations=8,
        system_prompt_override="你是财务分析专家。仔细对比多源数据，标注矛盾点和一致点。输出表格对比。",
        model_hint="large",
    ),
    "computation": SubAgentConfig(
        tools=["finish"],
        max_iterations=3,
        system_prompt_override="你是财务计算专家。精确计算并展示推导过程。只计算，不检索。",
        model_hint="small",
    ),
}
```

关键设计点：
- **computation 子 Agent 只有 `finish` 工具**——它不检索，只靠模型自身的数学推理能力做计算。
- **comparison 子 Agent 用 large 模型**——对比分析需要较强的推理和判断能力。
- **retrieval 子 Agent 用 small 模型**——检索是简单的"搜→读→返回"，不需要大模型。

### 2.3 技能系统 (`skills/`)

三个内置技能，均为文件夹 + `SKILL.md` 结构，模型根据 YAML frontmatter 中的 `description` 自主判断是否激活：

| 技能 | 触发场景 | 核心工作流 |
|------|---------|-----------|
| `financial-statement-analysis` | 财报数据、营收、净利润、ROE、毛利率等 | 识别报表范围 → 交叉验证 → 比率计算 → 趋势判断 |
| `risk-assessment` | 风险、违约、诉讼、ST、退市、处罚等 | 风险分类 → 正反证据 → 对立分析 → 量化评分 |
| `multi-hop-comparison` | 对比、排名、差异、优于/不如等 | 并行获取 → 表格输出 → 差异标注 |

```python
# skills/loader.py:61 — SkillManager
class SkillManager:
    def build_system_prompt(self, language: str = "zh") -> str:
        """组装 System Prompt：基础模板 + 工具描述 + 可用技能列表"""
        prompt = get_system_prompt(self.model_size, language)
        prompt += "\n" + get_tool_descriptions(language)
        listing = self.get_listing_text()  # 扫描 SKILL.md 生成摘要
        if listing:
            prompt += "\n" + listing
        return prompt

    def activate(self, name: str) -> Skill | None:
        """激活技能，返回 SKILL.md body 内容作为 prompt 扩展"""
```

新增技能只需创建文件夹放入 `SKILL.md`，零 Python 代码改动。

### 2.4 上下文压缩 (`context.py`)

三层策略按模型规格自动选择：

| 模型规格 | Token 上限 | 触发阈值 | 压缩策略 | 行为 |
|---------|-----------|---------|---------|------|
| small | 8192 | 80% | aggressive | 保留 system + 最后 4 条，其余摘要 |
| mid | 16384 | 80% | summarize_old | 保留 system + 最后 6 条，其余摘要 |
| large | 32768 | 80% | preserve_recent | 删除旧轮次的 tool 消息，保留文本推理 |

### 2.5 记忆系统 (`memory.py`)

文件持久化的跨会话记忆，采用 Claude Code 同款 `MEMORY.md` 索引结构：

- **存储**：`data/memory/<name>.md`，YAML frontmatter + markdown body
- **召回**：jieba 分词 + 关键词交集评分
- **自动记忆**：检索工具返回 confidence > 0.8 时自动调用 `remember`（`tools.py:189`）
- **手动记忆**：模型发现矛盾点、信息缺口时主动调用 `remember` 工具

## 3. 财报分析调用示例

### 3.1 单公司深度分析

> **查询**：「分析宁德时代 2024Q3 的盈利能力，计算毛利率、净利率、ROE，并与行业平均水平对比」

```
Round 1 ─ Think: 查询涉及"毛利率、净利率、ROE"等财务指标
  Act: activate_skill("financial-statement-analysis")
  Observe: 技能已激活 → 工作流指引: 识别报表范围 → 交叉验证 → 比率计算 → 趋势判断

Round 2 ─ Think: 需要宁德时代数据 + 行业均值，互不依赖，可以并行
  Act: dispatch_subagent × 2 (同轮并行派发)
    ├─ SubAgent-1 (retrieval)
    │    task="搜索宁德时代 2024Q3 利润表：营业收入、营业成本、
    │          净利润、归母净利润、净资产"
    │    → keyword_search("宁德时代 2024Q3 营收")        [第1轮]
    │    → read_chunk("CATL_2024Q3_income")              [第2轮]
    │    → keyword_search("宁德时代 2024Q3 归母净利润")    [第3轮]
    │    → finish(结构化数据)
    │
    └─ SubAgent-2 (retrieval)
         task="搜索动力电池行业 2024Q3 平均毛利率、净利率、ROE"
         → semantic_search("动力电池行业 平均毛利率 2024")  [第1轮]
         → semantic_search("动力电池 ROE 行业均值")         [第2轮]
         → finish(行业均值)

Round 3 ─ Think: 数据齐全，需要计算比率（纯计算任务）
  Act: dispatch_subagent(agent_type="computation",
       task="计算: 毛利率=(营收-营业成本)/营收,
             净利率=净利润/营收,
             ROE=净利润/净资产,
             并对比行业均值计算差值")
  Observe: 计算结果 + 推导过程 + 差值表

Round 4 ─ Think: 证据充分，可以输出
  Act: finish(answer="宁德时代 2024Q3 盈利能力分析：
    毛利率 25.3% (高于行业均值 22.1%，+3.2pp)
    净利率 11.8% (高于行业均值 8.5%，+3.3pp)
    ROE 18.2% (高于行业均值 12.7%，+5.5pp)
    结论：盈利能力显著优于行业平均...",
    confidence=0.88)
```

**时序示意**：

```
时间 ──────────────────────────────────────────────────────>

主 Agent:  [激活技能] ── [派发2子Agent] ── [等待] ── [派发计算] ── [等待] ── [finish]
                           │                            │
SubAgent-1:               [搜→读→搜→finish]              │
SubAgent-2:               [搜→搜→finish]                 │
SubAgent-3:                                             [计算→finish]
                           │                            │
                           ├── 并行执行 ──┤              │
```

### 3.2 跨公司横向对比

> **查询**：「对比比亚迪、特斯拉、宁德时代 2024 年的研发投入和研发占比」

```
Round 1 ─ Think: 涉及对比分析，激活技能获取工作流指引
  Act: activate_skill("multi-hop-comparison")
  Observe: 技能指引 — "不同公司 → 每个公司一个 retrieval 子代理，并行执行"

Round 2 ─ Think: 三家公司数据完全独立，按技能指引并行派发
  Act: dispatch_subagent × 3 (同轮并行)
    ├─ SubAgent-1 (retrieval) task="比亚迪 2024 年研发费用、营业收入"
    ├─ SubAgent-2 (retrieval) task="特斯拉 2024 R&D expense, total revenue"
    └─ SubAgent-3 (retrieval) task="宁德时代 2024 年研发费用、营业收入"
  Observe: 三份数据同时返回

Round 3 ─ Think: 数据齐全，生成对比表
  Act: dispatch_subagent(agent_type="computation",
       task="计算三家研发费用、研发占比(=研发费用/营收)，
             生成 Markdown 对比表格，差异超 10% 加粗标注")
  Observe: 对比表返回

Round 4 ─ finish
```

关键：`multi-hop-comparison` 技能的指引直接影响了模型的编排决策——"不同公司用并行子代理"不是代码逻辑，而是 SKILL.md 中的文字指引，由模型在激活后自主遵守。

### 3.3 风险评估 + 财报分析组合

> **查询**：「恒大集团是否有退市风险？分析其资产负债情况和违约记录」

```
Round 1 ─ Think: "退市"、"违约" → 风险评估领域
  Act: activate_skill("risk-assessment")
  Observe: 工作流 — 风险分类 → 正反证据 → 对立分析 → 量化评分

Round 2 ─ Think: 按技能指引，先做结构化任务分解
  Act: plan_steps(steps=[
    {id:"1", desc:"检索恒大资产负债数据"},
    {id:"2", desc:"检索恒大违约/诉讼/处罚记录"},
    {id:"3", desc:"检索退市规则适用性"},
    {id:"4", desc:"风险评估量化评分"},
  ])

Round 3 ─ Think: 步骤 1-3 互不依赖，全部并行
  Act: dispatch_subagent × 3 (同轮)
    ├─ SubAgent-1 (retrieval)
    │    task="恒大集团最新财报：总资产、总负债、资产负债率、
    │          流动负债、短期借款、现金及等价物"
    │    → 资产负债率 132%, 现金仅 15 亿
    │
    ├─ SubAgent-2 (retrieval)
    │    task="恒大集团违约记录：债券违约、银行贷款违约、
    │          供应商欠款、诉讼公告、证监会处罚"
    │    → keyword_search → graph_search("恒大"→"诉讼")
    │    → 境内债券违约 23 只, 供应商诉讼 400+ 件
    │
    └─ SubAgent-3 (retrieval)
         task="A股/港股退市规则：连续亏损、净资产为负、
               股价低于面值、重大违法等退市条件"
         → 净资产为负触发退市警告

Round 4 ─ Think: 发现重大风险信号，存入记忆。按技能要求做对立分析
  Act: remember(content="恒大资产负债率 132%，净资产为负，满足退市条件",
                type="evidence")
  Act: dispatch_subagent(agent_type="comparison",
       task="对立分析：恒大是否存在重组/救助可能（支持存活的证据）
             vs 退市信号强度（支持退市的证据）")
  Observe: 救助信号弱（无白名单、无战投进场），退市信号强

Round 5 ─ Think: 按技能要求做量化评分
  Act: finish(answer="恒大集团退市风险评估 (综合评分 4.5/5)：
    财务风险 5/5: 资产负债率 132%, 净资产为负...
    合规风险 5/5: 23 只债券违约, 400+ 诉讼...
    经营风险 4/5: 销售额连降, 大面积停工...
    市场风险 4/5: 股价长期低于 1 港元...
    结论：退市风险极高",
    confidence=0.92)
```

**完整时序**：

```
主Agent:  [激活技能] [plan] [派发3子Agent] [等待] [remember] [派发对比] [等待] [finish]
                              │                        │            │
Sub-1(retrieval):            [搜→读→搜→finish]          │            │
Sub-2(retrieval):            [搜→图→finish]             │            │
Sub-3(retrieval):            [搜→finish]                │            │
                              │                        │            │
Sub-4(comparison):                                     [搜→对比→finish]
                              │                        │            │
                              ├── 第1波并行 ──┤         ├─ 第2波 ──┤
```

## 4. 工具层级体系

三层的设计意图：**检索层是手和脚，元工具层是大脑皮层，生命周期层是终止信号**。

```
┌─────────────────────────────────────────────┐
│  生命周期层 (1)                               │
│  finish — 唯一出口，强制带上 confidence       │
├─────────────────────────────────────────────┤
│  元工具层 (4)                                 │
│  dispatch_subagent — 分解与并行               │
│  activate_skill    — 加载领域专业知识          │
│  plan_steps        — 复杂任务结构化            │
│  remember          — 关键发现持久化            │
├─────────────────────────────────────────────┤
│  检索工具层 (6)                               │
│  semantic_search   — FAISS 稠密向量 + BGE 重排│
│  keyword_search    — BM25 + jieba 分词        │
│  graph_search      — NetworkX 知识图谱遍历     │
│  hybrid_search     — RRF 融合 + CrossEncoder   │
│  read_chunk        — 按 ID 获取完整文本        │
│  text_to_sql       — NL→SQL→SQLite MCP 执行    │
└─────────────────────────────────────────────┘
```

工具选择内置于 System Prompt 的规则表中（`prompts.py:170-195`），模型在每轮 Think 阶段自行对照规则选择工具。

## 5. 与传统 Multi-Agent 框架的对比

| 特性 | 本系统 | LangGraph / CrewAI |
|------|--------|-------------------|
| Agent 编排 | 模型自主决策（ReAct 循环） | 预定义图结构或顺序管线 |
| 子 Agent 派发 | 模型在运行时判断"可拆分"才调用 | 预先配置固定 Agent 角色 |
| 并行执行 | `dispatch_subagent` 同轮并行 | 需显式定义并行边 |
| 技能激活 | 模型读 description 自主判断 | 关键词匹配或硬编码路由 |
| 新增能力 | 创建 SKILL.md 文件夹 | 需写 Python 代码 + 注册 |
| 状态持久化 | MemoryManager（跨会话） + AgentResult（单次） | LangGraph checkpointer / 无 |
| 小模型支持 | 自动降级：小模型不暴露 dispatch_subagent | 需手动调整图结构 |

核心哲学差异：**本系统把"何时用谁"的决策权完全交给 LLM**，代码只提供工具和执行通道，不写死编排规则。这也是为什么 System Prompt（`prompts.py`）和 Skill 指引（`SKILL.md`）对系统行为有关键影响——它们是模型决策的唯一依据。

## 6. 当前限制与后续方向

- **无 checkpoint**：Agent 崩溃后无法恢复。PEV 可通过 LangGraph checkpointer 接入，Agentic 需要自建。
- **子 Agent 无嵌套**：子 Agent 不能再派发孙 Agent（`enable_subagents=False`，`agent.py:308`），避免递归爆炸。
- **并行由模型判断**：模型可能判断失误，把有依赖的任务当并行派发。对比 skill 中写"不同公司→并行"是软约束，模型不一定遵守。
- **子 Agent 通信单向**：子 Agent 只向主 Agent 返回结果，子 Agent 之间不直接通信。结果合并由主 Agent 完成。
