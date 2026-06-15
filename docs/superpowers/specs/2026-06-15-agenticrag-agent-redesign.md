# AgenticRAG → Agent 架构改造设计

日期: 2026-06-15
状态: 已确认，待实施

## 1. 动机与目标

将现有 PEV (Planner-Executor-Verifier) 架构的 AgenticRAG 改造为 Claude Code 风格的 ReAct Agent，核心驱动力：

- **灵活性**：PEV 固定流程（预设计划 → 顺序执行 → 统一验证）太僵化，Agent 应能自主决定何时检索、推理、回答
- **参考 Claude Code**：借鉴其 Agent 框架的核心设计模式

### 约束

- **增量添加**：PEV 架构完全保留不动，Agent 作为新增模块并存
- **混合模型**：生产环境混合使用本地 7B-14B、32B+、云端 API
- **共享底层**：retrieval/ 和 llm/ 层两个系统共用

### 非目标（一期）

- MCP 协议客户端（预留接口，不做实现）
- Skills 语义匹配（用关键词匹配替代）
- Memory 语义检索（用关键词得分替代）
- 基于路由模型的智能分发

## 2. 核心设计原则

**Agent = System Prompt + Tools + 简单循环**

所有行为由 System Prompt 定义，工具由注册表统一管理，编排逻辑是一个 while 循环而非图节点。这是 Claude Code 的核心设计，也是本次改造的基础原则。

## 3. 架构总览

```
                     ┌─────────────────────────┐
                     │     api/server.py        │
                     │   /query?mode=pev|agent  │
                     └───────────┬─────────────┘
                                 │
               ┌─────────────────┴─────────────────┐
               │         PipelineRouter            │
               │  根据 mode 参数路由到对应系统       │
               └─────────────────┬─────────────────┘
                                 │
           ┌─────────────────────┼─────────────────────┐
           │                     │                     │
   ┌───────┴───────┐   ┌────────┴────────┐   ┌────────┴────────┐
   │  PEV Pipeline │   │  ReAct Agent    │   │  对比模式       │
   │  (保留不变)    │   │  (新增)         │   │  run_both()    │
   └───────┬───────┘   └────────┬────────┘   └────────┬────────┘
           │                     │                     │
           └─────────────────────┼─────────────────────┘
                                 │
               ┌─────────────────┴─────────────────┐
               │          共享层                    │
               │  retrieval/  │  llm/  │  data/    │
               └───────────────────────────────────┘
```

## 4. 模块设计

### 4.1 目录结构

```
agents/
  ├── agentic/                    # 新增：Claude Code 风格 Agent
  │   ├── __init__.py
  │   ├── agent.py                # ReAct 主循环 + Agent 类
  │   ├── tools.py                # 工具注册表（内置 + MCP接口）
  │   ├── sub_agent.py            # 子代理类型 + 派发管理
  │   ├── skills.py               # Skills 注册 + 匹配 + Prompt注入
  │   ├── context.py              # 上下文压缩（三层策略）
  │   ├── memory.py               # 记忆持久化
  │   ├── prompts.py              # System Prompt 模板
  │   └── types.py                # 共享类型定义
  ├── planner.py                  # PEV - 保留
  ├── executor.py                 # PEV - 保留
  ├── verifier.py                 # PEV - 保留
  ├── synthesizer.py              # PEV - 保留
  ├── graph.py                    # PEV - 保留
  └── ...

pipeline_router.py                # 新增：统一入口分发
evaluation/
  └── compare.py                  # 新增：PEV vs Agent 对比评估
config.py                         # 修改：新增 AGENT_CONFIG
api/server.py                     # 修改：新增路由
```

### 4.2 Agent 主循环 (`agent.py`)

ReAct 循环：工具驱动的 while 循环，模型自主决定每步行为。

```
初始化：System Prompt + 召回 Memory + 匹配 Skills
  ↓
while iteration < max_iterations:
  ├── 检查上下文 → 超过 80% 水位线触发压缩
  ├── LLM chat(messages, tools=schemas)
  ├── 响应是工具调用？
  │   ├── 元工具 → 特殊处理（dispatch_subagent, remember, plan_steps）
  │   ├── 检索工具 → 并行执行
  │   └── MCP 工具 → 通过 MCP 协议调用
  └── 响应是文本？
      ├── 纯推理 → 追加到消息历史，继续循环
      └── finish 调用 → 退出循环
```

关键状态字段：
- `iterations`: 当前循环轮次
- `tool_calls`: 所有工具调用记录
- `trace`: 执行轨迹（ReAct 格式，可用于 GRPO 训练）
- `skills_activated`: 本次激活的 Skills
- `subagents_dispatched`: 派发的子代理数量
- `compression_events`: 压缩事件记录

停止条件：
- 模型主动调用 `finish` → 正常退出
- `iterations >= max_iterations` → 汇总已有证据，标注缺失
- 连续 3 轮无工具调用且无新内容 → 强制退出

### 4.3 工具系统 (`tools.py`)

工具按三层分层，每层有不同的权限和优先级：

**第一层：检索工具（retrieval）**
无副作用，只读，可直接并行执行。

| 工具 | 来源 | when_to_use |
|------|------|-------------|
| semantic_search | retrieval/semantic_search.py | 模糊语义查询、概念解释 |
| keyword_search | retrieval/keyword_search.py | 精确字段匹配：公司名、代码、日期 |
| graph_search | retrieval/graph_search.py | 实体关系查询 |
| hybrid_search | retrieval/hybrid_search.py | 多模态召回融合 |
| read_chunk | retrieval/read_chunk.py | 按 ID 读取文本块 |

每个检索工具附 `when_to_use` 和 `when_not_to_use` 描述，让 LLM 更准确选择。

**第二层：元工具（meta）**
控制 Agent 行为本身，需要特殊处理逻辑。

| 工具 | 用途 | 对应 Claude Code |
|------|------|-----------------|
| dispatch_subagent | 派生子代理独立执行子任务 | Agent 工具 |
| remember | 保存关键证据/矛盾到持久记忆 | Memory 系统 |
| plan_steps | 创建结构化任务追踪 | TaskCreate |

**第三层：生命周期工具（lifecycle）**

| 工具 | 用途 |
|------|------|
| finish | 输出最终答案，附带置信度和证据摘要 |

**MCP 集成接口（预留）：**
- `ToolRegistry.discover_mcp_tools(servers)`: 启动时扫描 MCP 服务器
- MCP 工具带 `mcp__` 前缀，命名空间隔离
- 对内外部工具透明统一，LLM 感知不到差异

### 4.4 子代理机制 (`sub_agent.py`)

`dispatch_subagent` 是一个工具，但当 LLM 调用它时，系统创建独立的 Agent 实例执行。无依赖的子代理自动并行。

**子代理类型：**

| 类型 | 工具集 | max_iterations | 用途 |
|------|--------|---------------|------|
| retrieval | 搜索+读取+finish | 5 | 聚焦的信息检索 |
| comparison | 搜索+读取+finish | 8 | 多源数据对比分析 |
| computation | finish | 3 | 精确数值计算 |

子代理设计原则：
- 独立上下文（不共享消息历史）
- 受限工具集（只给必要工具）
- 独立 System Prompt（针对子任务的精确指导）
- 支持 background（无依赖的子代理并行执行）

**小模型降级：** 小模型 (7B-14B) 模式下 dispatch_subagent 不可用，改为串行推理。

### 4.5 Skills 系统 (`skills.py`)

Skill = 一段注入到 System Prompt 的专业指令 + 可选的受限工具列表 + 触发条件。

**一期 Skills：**

| Skill | 触发关键词 | 核心行为 |
|-------|-----------|----------|
| financial-statement-analysis | 财报、营收、净利润、ROE、资产负债... | 交叉验证、比率计算、趋势判断 |
| risk-assessment | 风险、违约、担保、诉讼、ST、退市... | 风险分类、对立分析、量化评分 |
| multi-hop-comparison | 对比、比较、差异、排名、优于、不如 | 并行子代理获取、表格输出 |

**匹配机制：** 关键词匹配（一期）；语义匹配（后续迭代）。

**System Prompt 组装：** 基础模板 + 匹配的 Skills 扩展指令。小模型最多激活 1 个 Skill。

### 4.6 上下文压缩 (`context.py`)

当 token 使用率超过 80% 时触发压缩，按模型尺寸分三种策略：

| 模型规格 | 策略 | 保留内容 |
|---------|------|---------|
| small (7B-14B) | 激进压缩 | System Prompt + 结构化摘要 + 最近 2 轮 |
| mid (32B-70B) | 摘要旧轮次 | System Prompt + 每轮摘要 + 最近 3 轮 |
| large (70B+) | 保留近期 | 移除最早工具结果，保留推理链 |

**压缩摘要格式：**
```
[数据]
- 营收2024Q1: 123亿 | source:chunk_45
[矛盾]
- chunk_12 与 chunk_89 对管理层变动时间描述不一致
[缺口]
- 未找到2024Q2的现金流数据
```

### 4.7 Memory 持久化 (`memory.py`)

文件系统存储，每条记忆独立文件，前缀头元数据，`MEMORY.md` 索引。

**记忆类型：**

| 类型 | 何时保存 | 如何使用 |
|------|---------|---------|
| evidence | 经 2+ 来源交叉验证的数据点 | 回答相关查询时优先引用 |
| contradiction | 发现矛盾且无法当场解决 | 后续查询标注不确定性 |
| gap | 确认缺失的信息 | 避免重复搜索已知缺失 |
| pattern | 同一类行为出现 3 次以上 | 缩短后续同类查询推理路径 |

**自动记忆：** `remember` 工具调用时保存；检索结果置信度 >0.8 或发现矛盾时自动触发。

**过期管理：** 矛盾解决后合并为 evidence；访问频次衰减清理。

### 4.8 System Prompt 设计 (`prompts.py`)

System Prompt 是 Agent 的唯一行为来源，包含以下段落结构：

1. **Role** - 角色定义
2. **Tools** - 工具使用说明（含 when_to_use/when_not_to_use）
3. **Behavior Rules** - 行为规则
4. **State Management** - 状态管理指导
5. **Stop Conditions** - 停止条件
6. **Model-specific instructions** - 按模型规格的条件指令

按模型尺寸和语言分层：`small` / `large` × `zh` / `en`。

## 5. 与 PEV 并行共存

### 5.1 PipelineRouter (`pipeline_router.py`)

统一入口，根据 `mode` 参数分发：

- `mode=pev` → PEV pipeline（完全不变）
- `mode=agent` → ReAct Agent（默认）
- `mode=compare` → 同时跑两边，返回对比结果

### 5.2 API 扩展 (`api/server.py`)

增量修改，原有路由完全保留：

- `POST /query` → 统一入口，新增 `mode` 字段
- `POST /query/pev` → 显式 PEV 模式（向后兼容）
- `POST /query/agent` → 显式 Agent 模式
- `POST /query/compare` → 对比模式
- `GET /health` → 增强，返回 PEV/Agent 可用性和 MCP/Skills 状态

### 5.3 评估对比 (`evaluation/compare.py`)

同一个 judge 模型评估两边，确保公平：

- 指标：correctness, faithfulness, latency, tool_calls, iterations
- 按查询难度分组统计
- 生成 winner/loser/tie 汇总

### 5.4 渐进策略

- Phase 1: Agent 为默认，PEV 保留（随时可切回）
- Phase 2: 10% 流量用 compare 模式收集数据
- Phase 3: 根据数据决定保留、废弃或智能路由

## 6. 与 Claude Code 的设计对应

| Claude Code | RAG Agent |
|-------------|-----------|
| System Prompt (行为宪法) | prompts.py + Skills 注入 |
| Tool Layer (Bash/Read/Write) | tools.py (检索/元/生命周期) |
| Agent tool (子代理派发) | dispatch_subagent + SubAgentType |
| MCP protocol (外部工具) | mcp_client.py 接口 (预留) |
| Skills (领域行为扩展) | skills.py + 关键词匹配 |
| Memory (持久记忆) | memory.py (四类型) |
| Context auto-compression | context.py (三层策略) |
| TaskCreate/TaskList | plan_steps 元工具 |

## 7. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 小模型无法有效使用子代理 | 中 | 高 | 小模型禁用 dispatch_subagent，串行推理 |
| ReAct 循环无限不自停 | 低 | 中 | max_iterations 硬上限 + 连续 3 轮无工具调用强制 finish |
| 上下文爆炸 | 中 | 中 | 80% 水位线触发压缩，小模型激进策略 |
| MCP 工具描述质量差 | 中 | 中 | 启动时校验，缺失描述自动生成 |
| 评估尺度不一致 | 低 | 中 | 同一 judge 模型 + 同一 prompt |
| Memory 膨胀 | 低 | 低 | 过期清理 + 矛盾合并 + 频次衰减 |
| Skills 关键词匹配粗糙 | 中 | 低 | 二期用语义匹配替代 |

## 8. 文件清单

**新增 (11 个):**
- `agents/agentic/__init__.py`
- `agents/agentic/agent.py`
- `agents/agentic/tools.py`
- `agents/agentic/sub_agent.py`
- `agents/agentic/skills.py`
- `agents/agentic/context.py`
- `agents/agentic/memory.py`
- `agents/agentic/types.py`
- `agents/agentic/prompts.py`
- `pipeline_router.py`
- `evaluation/compare.py`

**修改 (2 个):**
- `api/server.py` — 新增路由
- `config.py` — 新增 AGENT_CONFIG

**不受影响:**
- `agents/` 下除 `agentic/` 外的所有文件
- `retrieval/` 全部
- `llm/` 全部
- `training/` 全部
- `data/` 全部
