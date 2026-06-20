项目描述：针对中文金融财报中“跨章节、多指标对比”的复杂多跳（Multi-hop）问答痛点，通过迁移 AgenticRAGTracer 评测方法论，自主设计并实现具备长链推理能力的财报分析 Agent，显著提升了复杂金融场景下的问答精度。

1. 知识库构建：收集多家上市公司金融年报 PDF，实现复杂财报的多模态提取，采用递归分块并通过规则清洗和 LLM 精炼提升文本块质量，构建向量 + BM25 + 知识图谱三路检索索引，并通过 RRF 融合和 Reranker 重排优化检索结果，将单跳 QA 相关上下文的检索召回率（Recall@5）提升了 23%。
2. 评测基准：设计多跳 QA 合成 pipeline（种子 QA 提取、多跳组合、四重验证、LLM Judge 评分），优化多跳组合提高产出率，成功从 4k 种子数据中精炼出覆盖 2-4 跳推理场景的高质量数据集 983 条。
3. Agent架构：设计中心化 Multi-Agent 协作架构，集成 MCP 协议和 Skills，封装多路检索、SQL 引擎及特定子任务 Sub-Agent 工具链；设计 15 轮执行硬上限 + 3 轮连续报错自动退避熔断机制，配合异常兜底策略保障系统稳定性。
4. 上下文压缩：设计实现三级渐进式上下文压缩机制，优先裁剪旧工具调用链路，其次用结构化增量摘要替代历史消息，最后使用 AI 摘要和截断兜底 ，尽量保证长链路稳定性同时提升 Prompt Cache 收益。
5. 监控与评测：基于 Langfuse 搭建 Agent 链路执行轨迹持久化与可视化追踪系统，量化评估 Agent 规划合理性与工具调用准确率，根据 bad Case 驱动 Prompt 迭代与 Tool Schema 优化。


用户查询: "计算宁德时代 2024Q3 的 DuPont ROE 分解，并对比行业均值"
```markdown
┌─────────────────────────────────────────────────────┐
│                    Agent 主循环                       │
│                                                      │
│  ① PlanPhase: LLM 生成结构化 Plan                    │
│  ┌──────────────────────────────────────────────┐    │
│  │ steps:                                       │    │
│  │  [1] 获取宁德时代利润表 (retrieval) []        │    │
│  │  [2] 获取宁德时代资产负债表 (retrieval) []    │    │
│  │  [3] 获取行业平均数据 (retrieval) []          │    │
│  │  [4] DuPont ROE 分解计算 (computation) [1,2]  │    │
│  │  [5] 行业对比计算 (computation) [3,4]         │    │
│  │  [6] 结论综合 (comparison) [4,5]              │    │
│  └──────────────────────────────────────────────┘    │
│                                                      │
│  ② ExecutePhase: 按依赖调度                          │
│  ┌──────────────────────────────────────────────┐    │
│  │ Wave 1 (并行): [1] [2] [3]                   │    │
│  │   ├─ SubAgent(retrieval) → keyword_search    │    │
│  │   ├─ SubAgent(retrieval) → semantic_search   │    │
│  │   └─ SubAgent(retrieval) → hybrid_search     │    │
│  │                                              │    │
│  │ Wave 2: [4] 依赖 [1,2] 完成                  │    │
│  │   └─ SubAgent(computation) → execute_python  │    │
│  │        │                                      │    │
│  │        ▼  Python MCP Server (独立进程)        │    │
│  │        ┌──────────────────────────────┐      │    │
│  │        │ import pandas as pd          │      │    │
│  │        │ import numpy as np           │      │    │
│  │        │                              │      │    │
│  │        │ # DuPont 分解                │      │    │
│  │        │ ni = 36.0  # 净利润(亿)       │      │    │
│  │        │ rev = 292.0  # 营收           │      │    │
│  │        │ assets = 320.0  # 总资产       │      │    │
│  │        │ equity = 198.0  # 净资产       │      │    │
│  │        │                              │      │    │
│  │        │ npm = ni / rev               │      │    │
│  │        │ tat = rev / assets           │      │    │
│  │        │ em = assets / equity         │      │    │
│  │        │ roe = npm * tat * em         │      │    │
│  │        │                              │      │    │
│  │        │ print(f"ROE={roe:.4f}")      │      │    │
│  │        │ print(f"NPM={npm:.4f}")      │      │    │
│  │        │ print(f"TAT={tat:.4f}")      │      │    │
│  │        │ print(f"EM={em:.4f}")        │      │    │
│  │        └──────────────────────────────┘      │    │
│  │        → 返回: stdout + 结构化结果            │    │
│  │                                              │    │
│  │ Wave 3: [5] 依赖 [3,4] 完成                  │    │
│  │   └─ SubAgent(computation) → execute_python  │    │
│  │        → 行业对比计算                         │    │
│  │                                              │    │
│  │ Wave 4: [6] → SubAgent(comparison)           │    │
│  │   → 综合分析 + 结论生成                       │    │
│  └──────────────────────────────────────────────┘    │
│                                                      │
│  ③ Finish: 输出最终答案 + Plan trace                 │
└─────────────────────────────────────────────────────┘
```



