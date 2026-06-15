"""Skills 技能系统 — 按需加载的领域能力模块"""
from dataclasses import dataclass, field


@dataclass
class Skill:
    """技能：领域知识 + 行为约束的模块"""
    name: str
    description: str
    trigger_keywords: list[str]
    prompt_extension: str
    restricted_tools: list[str] | None = None
    override_max_iterations: int | None = None

    def matches(self, query: str) -> bool:
        return any(kw in query for kw in self.trigger_keywords)


# ══════════════════════════════════════════════════════════════════
# 技能注册表
# ══════════════════════════════════════════════════════════════════

SKILL_REGISTRY: dict[str, Skill] = {
    "financial-statement-analysis": Skill(
        name="financial-statement-analysis",
        description="深度解析财报三大表，计算关键财务比率",
        trigger_keywords=["财报", "营收", "净利润", "ROE", "资产负债", "现金流",
                          "毛利润", "净利率", "流动比率", "速动比率", "应收账款",
                          "存货周转", "营业成本", "归母净利润"],
        prompt_extension="""
## 激活技能：财务报表分析

当前查询涉及财务报表分析，遵循以下工作流：

1. **识别报表范围**：确认需要哪些报表期间、哪些科目
2. **交叉验证**：同一数据如果多个来源有出入，标注差异而非选择其一
3. **比率计算要点**：
   - 盈利能力：毛利率、净利率、ROE、ROA
   - 偿债能力：流动比率、速动比率、资产负债率
   - 运营效率：存货周转率、应收账款周转率
4. **趋势判断**：对比多期数据时给出方向性结论
5. **置信度要求**：财报分析要求更高置信度 (>= 0.8)
""",
    ),
    "risk-assessment": Skill(
        name="risk-assessment",
        description="评估公司的财务风险、经营风险和合规风险",
        trigger_keywords=["风险", "违约", "担保", "诉讼", "ST", "退市",
                          "处罚", "监管", "立案", "警示函", "问询函"],
        prompt_extension="""
## 激活技能：风险评估

当前查询涉及风险评估，遵循：

1. **风险分类**：先将风险归类（财务/经营/合规/市场）
2. **证据收集**：不仅找正面证据，必须主动搜索负面信号
3. **对立分析**：对每个风险点，同时搜索支持和反对的证据
4. **量化评分**：给每类风险打分（1-5），附带依据
""",
    ),
    "multi-hop-comparison": Skill(
        name="multi-hop-comparison",
        description="跨公司、跨时间段的对比分析",
        trigger_keywords=["对比", "比较", "差异", "排名", "优于", "不如",
                          "最高", "最低", "大于", "小于", "超过"],
        prompt_extension="""
## 激活技能：多跳对比

当前查询涉及对比分析，遵循：

1. **并行获取**：所有可并行的信息获取通过 `dispatch_subagent` 并行执行
   - 不同公司 → 每个公司一个 retrieval 子代理
   - 不同指标 → 每个指标一个 computation 子代理
2. **输出格式**：表格优先，每列一家公司/时间段，每行一个指标
3. **差异标注**：关键差异高亮，附带来源引用
""",
    ),
}


class SkillManager:
    """技能管理器 — 匹配与注入"""

    def __init__(self, model_size: str = "large"):
        self.model_size = model_size
        self.loaded: list[Skill] = []

    def match(self, query: str) -> list[Skill]:
        """根据查询匹配应激活的 Skills"""
        matched = []
        for name, skill in SKILL_REGISTRY.items():
            if skill.matches(query):
                matched.append(skill)

        # 小模型限制
        if self.model_size == "small" and len(matched) > 1:
            matched.sort(
                key=lambda s: sum(1 for kw in s.trigger_keywords if kw in query),
                reverse=True,
            )
            matched = matched[:1]

        self.loaded = matched
        return matched

    def build_system_prompt(self, query: str, language: str = "zh") -> str:
        """组装完整 System Prompt：基础 + Skills 扩展"""
        from agents.agentic.prompts import get_system_prompt, get_tool_descriptions

        prompt = get_system_prompt(self.model_size, language)
        prompt += "\n" + get_tool_descriptions(language)

        matched = self.match(query)
        for skill in matched:
            prompt += f"\n\n{skill.prompt_extension}"

        if self.model_size == "small" and len(matched) > 0:
            prompt += "\n\n注意：小模型模式下只激活一个技能。"

        return prompt

    def get_active_skill_names(self) -> list[str]:
        return [s.name for s in self.loaded]
