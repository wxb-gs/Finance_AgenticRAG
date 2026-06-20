"""BadCaseRouter — Bad Case 识别 → 分类 → Prompt/Schema 优化建议"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class BadCase:
    """单个 Bad Case 记录"""
    query: str
    category: str          # tool_selection_low / arg_quality_poor / ...
    scores: dict            # 触发该分类的分数快照
    trace_id: str = ""
    optimize_target: str = ""  # "prompt" | "tool_schema"


# 分类规则定义
BADCASE_RULES = [
    {
        "name": "tool_selection_low",
        "category": "工具选择错误",
        "condition": lambda s: s.get("tool_selection_f1", 1.0) < 0.5,
        "target": "tool_schema",
        "description": "Agent 调用的工具类型与预期严重不一致",
    },
    {
        "name": "arg_quality_poor",
        "category": "参数质量差",
        "condition": lambda s: s.get("arg_quality_avg", 5.0) < 3.0,
        "target": "prompt",
        "description": "工具调用参数缺少关键实体或约束条件",
    },
    {
        "name": "high_redundancy",
        "category": "无效调用过多",
        "condition": lambda s: s.get("redundancy_rate", 0.0) > 0.4,
        "target": "tool_schema",
        "description": "超过 40% 的工具调用返回空结果或无效应答",
    },
    {
        "name": "step_inefficient",
        "category": "步数效率低",
        "condition": lambda s: s.get("step_efficiency", 1.0) < 0.5,
        "target": "prompt",
        "description": "Agent 用了远超必要数量的工具调用才完成任务",
    },
    {
        "name": "early_finish",
        "category": "过早终止",
        "condition": lambda s: s.get("premature_finish", False),
        "target": "prompt",
        "description": "Agent 在未收集足够证据时提前结束循环",
    },
    {
        "name": "plan_mismatch",
        "category": "规划遗漏",
        "condition": lambda s: s.get("plan_hop_recall", 1.0) < 0.5,
        "target": "prompt",
        "description": "Agent 的计划未能覆盖 ground truth 的关键 hop",
    },
]

# 触发建议的最小同类累积数
MIN_ACCUMULATE = 5


class BadCaseRouter:
    """根据评测分数自动分类 Bad Case 并生成优化建议"""

    def __init__(self):
        self._accumulator: dict[str, list[BadCase]] = {}

    def classify(self, query: str, scores: dict, trace_id: str = "") -> list[BadCase]:
        """根据分数快照自动分类，返回匹配的 BadCase 列表"""
        matched = []
        for rule in BADCASE_RULES:
            try:
                if rule["condition"](scores):
                    bc = BadCase(
                        query=query,
                        category=rule["name"],
                        scores=scores,
                        trace_id=trace_id,
                        optimize_target=rule["target"],
                    )
                    matched.append(bc)
                    self._accumulate(bc)
            except Exception as e:
                logger.warning(f"BadCase rule '{rule['name']}' failed: {e}")
        return matched

    def _accumulate(self, bc: BadCase):
        """累积同类 Bad Case"""
        key = bc.category
        if key not in self._accumulator:
            self._accumulator[key] = []
        self._accumulator[key].append(bc)

    def get_pending_suggestions(self) -> list[dict]:
        """返回所有达到阈值的优化建议"""
        suggestions = []
        for category, cases in self._accumulator.items():
            if len(cases) >= MIN_ACCUMULATE:
                rule = next((r for r in BADCASE_RULES if r["name"] == category), None)
                if rule:
                    suggestion = self._build_suggestion(rule, cases)
                    suggestions.append(suggestion)
                    # 重置，避免重复触发
                    self._accumulator[category] = []
        return suggestions

    def _build_suggestion(self, rule: dict, cases: list[BadCase]) -> dict:
        """基于累积的 Bad Case 生成优化建议"""
        query_samples = [c.query[:120] for c in cases[:5]]
        score_summary = self._aggregate_scores(cases)

        if rule["target"] == "prompt":
            suggestion_text = self._build_prompt_suggestion(
                rule, query_samples, score_summary
            )
        else:
            suggestion_text = self._build_schema_suggestion(
                rule, query_samples, score_summary
            )

        return {
            "category": rule["name"],
            "target": rule["target"],
            "case_count": len(cases),
            "query_samples": query_samples,
            "score_summary": score_summary,
            "suggestion": suggestion_text,
        }

    def _aggregate_scores(self, cases: list[BadCase]) -> dict:
        """聚合多个 Bad Case 的分数统计"""
        if not cases:
            return {}
        keys = cases[0].scores.keys()
        result = {}
        for k in keys:
            values = [c.scores[k] for c in cases if k in c.scores]
            if values:
                result[k] = {
                    "min": round(min(values), 4),
                    "max": round(max(values), 4),
                    "avg": round(sum(values) / len(values), 4),
                }
        return result

    def _build_prompt_suggestion(
        self, rule: dict, samples: list[str], scores: dict,
    ) -> str:
        """生成 Prompt 优化建议"""
        suggestions = {
            "arg_quality_poor": (
                "建议在 System Prompt 中增加工具调用参数规范：\n"
                "1. 要求每个检索 query 必须包含所有关键实体名称和时效约束\n"
                "2. 增加 few-shot 示例展示高质量参数格式\n"
                "3. 在工具描述中增加参数反例说明"
            ),
            "step_inefficient": (
                "建议在 System Prompt 中强化工具调用效率意识：\n"
                "1. 要求优先使用 hybrid_search 覆盖多维度，减少单工具多次调用\n"
                "2. 增加\"调用前先思考是否已有足够信息\"的提醒\n"
                "3. 设置每轮工具调用上限提示"
            ),
            "early_finish": (
                "建议在 System Prompt 中强化完备性检查：\n"
                "1. 要求 finish 前必须确认所有 plan 步骤已完成\n"
                "2. 增加\"在信息不足时继续搜索而非提前终止\"的规则\n"
                "3. 提高连续无工具调用的终止阈值（当前 3）或移除该规则"
            ),
            "plan_mismatch": (
                "建议优化 Plan 生成 Prompt：\n"
                "1. 在 plan_query 工具描述中增加 hop 分解示例\n"
                "2. 要求 plan 步骤显式标注每个步骤的目标信息类型\n"
                "3. 增加多跳问题 plan 模板"
            ),
        }
        base = suggestions.get(rule["name"], "建议审查并优化相关 Prompt 模板。")
        return (
            f"## Bad Case 类型：{rule['category']}\n"
            f"涉及 {len(samples)} 个案例，例如：{samples[0] if samples else 'N/A'}\n\n"
            f"{base}\n\n"
            f"修改文件：agents/agentic/prompts.py"
        )

    def _build_schema_suggestion(
        self, rule: dict, samples: list[str], scores: dict,
    ) -> str:
        """生成 Tool Schema 优化建议"""
        suggestions = {
            "tool_selection_low": (
                "建议调整工具描述和优先级：\n"
                "1. 检查 when_to_use / when_not_to_use 描述是否与实际使用场景匹配\n"
                "2. 考虑调整工具 priority 值以影响模型选择偏好\n"
                "3. 在工具描述中增加典型查询示例"
            ),
            "high_redundancy": (
                "建议优化工具 Schema 减少无效调用：\n"
                "1. 检查工具 description 中是否过度承诺了不存在的功能\n"
                "2. 为参数增加更严格的约束（如 min_length、enum 等）\n"
                "3. 在 when_not_to_use 中增加会导致空结果的典型场景"
            ),
        }
        base = suggestions.get(rule["name"], "建议审查并优化相关工具 Schema 定义。")
        return (
            f"## Bad Case 类型：{rule['category']}\n"
            f"涉及 {len(samples)} 个案例，例如：{samples[0] if samples else 'N/A'}\n\n"
            f"{base}\n\n"
            f"修改文件：agents/agentic/tools.py (_RETRIEVAL_TOOL_DEFS)"
        )

    def reset(self):
        """清空累积的 Bad Case"""
        self._accumulator.clear()
