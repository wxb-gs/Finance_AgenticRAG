"""EvalReporter — 评测指标计算与 Langfuse Score 上报"""

import logging
from collections import Counter

logger = logging.getLogger(__name__)


class EvalReporter:
    """为单次 Agent 执行计算评测指标并上报到 Langfuse Trace"""

    def __init__(self, tracer=None):
        self.tracer = tracer

    # ── 1. 工具选择 Accuracy ──

    def report_tool_selection(
        self,
        actual_tools: list[str],
        expected_tools: list[str],
    ) -> dict:
        """工具选择 Precision/Recall/F1

        Args:
            actual_tools: Agent 实际调用的工具名列表（去重）
            expected_tools: Ground truth 标注的工具集
        """
        actual_set = set(actual_tools)
        expected_set = set(expected_tools)

        tp = len(actual_set & expected_set)
        precision = tp / len(actual_set) if actual_set else 0.0
        recall = tp / len(expected_set) if expected_set else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        scores = {
            "tool_selection_precision": round(precision, 4),
            "tool_selection_recall": round(recall, 4),
            "tool_selection_f1": round(f1, 4),
        }

        if self.tracer:
            self.tracer.score_many(scores)

        return scores

    # ── 2. 参数质量（LLM-as-Judge）──

    def report_argument_quality(
        self,
        tool_calls: list[dict],
        query: str,
        judge_model: str = "gpt-oss-120b",
    ) -> dict:
        """对每次工具调用的参数质量打分 1-5

        Args:
            tool_calls: [{name, args, result_preview}, ...]
            query: 原始用户查询
            judge_model: 评分用的 Judge 模型名
        """
        if not tool_calls:
            scores = {"arg_quality_avg": 0.0, "arg_quality_min": 0.0}
            if self.tracer:
                self.tracer.score_many(scores)
            return scores

        ratings = []
        from llm.client import judge_chat_json

        for tc in tool_calls:
            name = tc.get("name", "unknown")
            args = tc.get("args", {})
            result_preview = tc.get("result_preview", "")

            prompt = f"""评估以下工具调用的参数质量。评分标准：
- 5: 参数精准覆盖了用户查询中的所有关键实体、约束条件
- 4: 参数覆盖了大部分关键要素，有轻微遗漏
- 3: 参数部分覆盖，缺少关键实体或时限约束
- 2: 参数与查询意图部分偏离，较大缺陷
- 1: 参数与查询无关或完全偏离

用户查询：{query}
工具名称：{name}
调用参数：{args}
返回结果摘要：{result_preview}

请返回JSON格式：{{"score": <1-5>, "reason": "<简要理由>"}}"""

            try:
                result = judge_chat_json(prompt, model=judge_model)
                if result and "score" in result:
                    ratings.append(min(5, max(1, int(result["score"]))))
                else:
                    ratings.append(3)  # 默认分数
            except Exception as e:
                logger.warning(f"Argument quality judge failed: {e}")
                ratings.append(3)

        avg_quality = sum(ratings) / len(ratings)
        min_quality = min(ratings)

        scores = {
            "arg_quality_avg": round(avg_quality, 2),
            "arg_quality_min": float(min_quality),
        }

        if self.tracer:
            self.tracer.score_many(scores)

        return scores

    # ── 3. 调用效率（自动）──

    def report_call_efficiency(
        self,
        tool_calls: list[dict],
        expected_tool_count: int = 0,
    ) -> dict:
        """冗余率 / 重复率 / 步数效率

        Args:
            tool_calls: [{"name": ..., "is_empty": bool, "args": {...}}, ...]
            expected_tool_count: ground truth 期望的工具类型数
        """
        total = len(tool_calls)
        if total == 0:
            scores = {
                "redundancy_rate": 0.0,
                "repetition_count": 0,
                "step_efficiency": 0.0,
            }
            if self.tracer:
                self.tracer.score_many(scores)
            return scores

        # 冗余率：空结果或 success=false 的调用占比
        empty_count = sum(
            1 for tc in tool_calls
            if tc.get("is_empty") or not tc.get("success", True)
        )
        redundancy_rate = empty_count / total

        # 重复率：相同 (name, query_string) 的重复调用
        call_sigs = [
            (tc.get("name", ""), str(tc.get("args", {})))
            for tc in tool_calls
        ]
        sig_counts = Counter(call_sigs)
        repetition_count = sum(c - 1 for c in sig_counts.values())

        # 步数效率：理论最小 / 实际
        min_steps = max(1, expected_tool_count + 1)  # +1 for finish
        step_efficiency = min_steps / total if total > 0 else 1.0

        scores = {
            "redundancy_rate": round(redundancy_rate, 4),
            "repetition_count": repetition_count,
            "step_efficiency": round(min(1.0, step_efficiency), 4),
        }

        if self.tracer:
            self.tracer.score_many(scores)

        return scores

    # ── 4. 规划合理性 ──

    def report_planning(
        self,
        plan_steps_desc: list[str],
        ground_truth_hops: list[str],
    ) -> dict:
        """计算计划步骤对 ground truth hop 的覆盖率

        Args:
            plan_steps_desc: Agent plan 中各步骤的 description 文本列表
            ground_truth_hops: Ground truth 的 hop 描述列表
        """
        if not ground_truth_hops:
            scores = {"plan_hop_precision": 0.0, "plan_hop_recall": 0.0}
            if self.tracer:
                self.tracer.score_many(scores)
            return scores

        if not plan_steps_desc:
            scores = {"plan_hop_precision": 0.0, "plan_hop_recall": 0.0}
            if self.tracer:
                self.tracer.score_many(scores)
            return scores

        # 简单 token 重叠匹配：每个 plan step 匹配最佳 hop
        import jieba

        def _tokenize(text: str) -> set:
            return set(jieba.lcut(text.lower()))

        plan_tokens_list = [_tokenize(s) for s in plan_steps_desc]
        hop_tokens_list = [_tokenize(h) for h in ground_truth_hops]

        # 每个 hop 找最佳匹配 plan step
        matched_hops = 0
        used_plans = set()
        for hop_tokens in hop_tokens_list:
            best_overlap = 0
            best_idx = -1
            for i, plan_tokens in enumerate(plan_tokens_list):
                if i in used_plans:
                    continue
                overlap = len(hop_tokens & plan_tokens) / max(1, len(hop_tokens))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = i
            if best_overlap >= 0.2:  # 至少 20% token 重叠
                matched_hops += 1
                used_plans.add(best_idx)

        plan_recall = matched_hops / len(ground_truth_hops)
        plan_precision = matched_hops / len(plan_steps_desc) if plan_steps_desc else 0.0

        scores = {
            "plan_hop_precision": round(plan_precision, 4),
            "plan_hop_recall": round(plan_recall, 4),
        }

        if self.tracer:
            self.tracer.score_many(scores)

        return scores
