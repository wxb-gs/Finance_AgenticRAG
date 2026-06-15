"""PEV vs Agent A/B 对比评估"""
import time
from dataclasses import dataclass, field


@dataclass
class SingleEval:
    correctness: float
    faithfulness: float
    latency_ms: float
    tool_calls: int
    iterations: int


@dataclass
class ComparisonReport:
    total_queries: int = 0
    pev_scores: list[SingleEval] = field(default_factory=list)
    agent_scores: list[SingleEval] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)

    def avg_pev_correctness(self) -> float:
        if not self.pev_scores: return 0
        return sum(s.correctness for s in self.pev_scores) / len(self.pev_scores)

    def avg_agent_correctness(self) -> float:
        if not self.agent_scores: return 0
        return sum(s.correctness for s in self.agent_scores) / len(self.agent_scores)

    def winner_stats(self) -> dict:
        agent_wins = pev_wins = ties = 0
        for a, p in zip(self.agent_scores, self.pev_scores):
            if a.correctness > p.correctness:
                agent_wins += 1
            elif p.correctness > a.correctness:
                pev_wins += 1
            else:
                ties += 1
        return {"agent_wins": agent_wins, "pev_wins": pev_wins, "ties": ties}

    def summary(self) -> str:
        win = self.winner_stats()
        return (
            f"=== PEV vs Agent Comparison ({self.total_queries} queries) ===\n"
            f"Correctness — PEV: {self.avg_pev_correctness():.3f}  "
            f"Agent: {self.avg_agent_correctness():.3f}\n"
            f"Wins — Agent: {win['agent_wins']}  PEV: {win['pev_wins']}  "
            f"Ties: {win['ties']}"
        )


class PipelineComparator:
    """PEV vs Agent 对比评估器"""

    def __init__(self, router, judge_model_config=None):
        self.router = router
        self.judge_config = judge_model_config

    def evaluate(self, queries: list[str],
                 ground_truths: list[str] | None = None) -> ComparisonReport:
        report = ComparisonReport(total_queries=len(queries))
        ground_truths = ground_truths or [""] * len(queries)

        for i, (query, gt) in enumerate(zip(queries, ground_truths)):
            pev = self.router._run_pev(query)
            agent = self.router._run_agent(query)

            pev_correct = self._judge_correctness(pev["answer"], gt, query)
            agent_correct = self._judge_correctness(agent["answer"], gt, query)
            pev_faith = self._judge_faithfulness(pev["answer"], pev["trace"], query)
            agent_faith = self._judge_faithfulness(agent["answer"], agent["trace"], query)

            report.pev_scores.append(SingleEval(
                correctness=pev_correct, faithfulness=pev_faith,
                latency_ms=pev["latency_ms"], tool_calls=pev["total_tool_calls"],
                iterations=pev["iterations"],
            ))
            report.agent_scores.append(SingleEval(
                correctness=agent_correct, faithfulness=agent_faith,
                latency_ms=agent["latency_ms"], tool_calls=agent["total_tool_calls"],
                iterations=agent["iterations"],
            ))
            report.queries.append(query)

        return report

    def _judge_correctness(self, answer: str, ground_truth: str, query: str) -> float:
        if not ground_truth:
            return 0.5
        from llm.client import judge_chat_json
        prompt = (
            f"Score the answer's correctness against the ground truth.\n"
            f"Query: {query}\n"
            f"Ground truth: {ground_truth}\n"
            f"Answer: {answer}\n"
            f"Return JSON: {{\"score\": <0-1 float>, \"reason\": \"<brief>\"}}"
        )
        try:
            result = judge_chat_json(prompt)
            return float(result.get("score", 0.5))
        except Exception:
            return 0.5

    def _judge_faithfulness(self, answer: str, trace: list[dict], query: str) -> float:
        if not trace:
            return 0.5
        from llm.client import judge_chat_json
        evidence_text = "\n".join(
            t.get("result_summary", "") for t in trace[-5:]
        )
        prompt = (
            f"Score whether the answer is fully supported by the evidence.\n"
            f"Query: {query}\n"
            f"Evidence: {evidence_text[:2000]}\n"
            f"Answer: {answer}\n"
            f"Return JSON: {{\"score\": <0-1 float>, \"reason\": \"<brief>\"}}"
        )
        try:
            result = judge_chat_json(prompt)
            return float(result.get("score", 0.5))
        except Exception:
            return 0.5
