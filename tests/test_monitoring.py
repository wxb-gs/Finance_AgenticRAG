"""监控模块单元测试"""
import pytest
from unittest.mock import patch, MagicMock


class TestTracer:
    """Tracer 单元测试"""

    def test_noop_mode_returns_without_error(self):
        from monitoring.tracer import Tracer
        tracer = Tracer.noop()
        assert tracer.enabled is False
        # All methods should return without error
        assert tracer.start_trace("test query") is None
        tracer.end_trace()
        tracer.start_iteration(1)
        tracer.end_iteration()
        tracer.log_generation("test-model", 10, 2, 150.0)
        tracer.log_tool_call("search", {"q": "test"}, "result", True, 0.9, 50.0)
        tracer.log_compression("session_memory", 10000, 5000)
        tracer.log_subagent("retrieval", "find X", 3)
        tracer.log_recall(2)
        tracer.score("test_score", 0.8)
        tracer.flush()  # Should not raise

    def test_enabled_tracer_creates_trace(self):
        from monitoring.tracer import Tracer
        tracer = Tracer(enabled=True)
        tracer._client = MagicMock()
        tracer._sample = True
        trace_id = tracer.start_trace("test query", mode="agent", model="Qwen3-32B")
        tracer._client.trace.assert_called_once()
        assert trace_id is not None

    def test_disabled_tracer_skips_all(self):
        from monitoring.tracer import Tracer
        tracer = Tracer(enabled=False)
        tracer._client = MagicMock()
        assert tracer.start_trace("q") is None
        tracer._client.trace.assert_not_called()


class TestEvalReporter:
    """EvalReporter 单元测试"""

    def test_tool_selection_perfect(self):
        from monitoring.eval_reporter import EvalReporter
        reporter = EvalReporter(tracer=None)
        scores = reporter.report_tool_selection(
            actual_tools=["semantic_search", "keyword_search", "read_chunk"],
            expected_tools=["semantic_search", "keyword_search", "read_chunk"],
        )
        assert scores["tool_selection_precision"] == 1.0
        assert scores["tool_selection_recall"] == 1.0
        assert scores["tool_selection_f1"] == 1.0

    def test_tool_selection_miss(self):
        from monitoring.eval_reporter import EvalReporter
        reporter = EvalReporter(tracer=None)
        scores = reporter.report_tool_selection(
            actual_tools=["semantic_search"],
            expected_tools=["semantic_search", "keyword_search"],
        )
        assert scores["tool_selection_precision"] == 1.0
        assert scores["tool_selection_recall"] == 0.5

    def test_tool_selection_empty(self):
        from monitoring.eval_reporter import EvalReporter
        reporter = EvalReporter(tracer=None)
        scores = reporter.report_tool_selection(
            actual_tools=[],
            expected_tools=["semantic_search"],
        )
        assert scores["tool_selection_precision"] == 0.0
        assert scores["tool_selection_recall"] == 0.0

    def test_tool_selection_zero_expected(self):
        from monitoring.eval_reporter import EvalReporter
        reporter = EvalReporter(tracer=None)
        scores = reporter.report_tool_selection(
            actual_tools=["semantic_search"],
            expected_tools=[],
        )
        # When expected is empty and actual is non-empty:
        # precision = 0/1 = 0, recall = 0/0 = 0 (guard)
        assert scores["tool_selection_f1"] == 0.0

    def test_call_efficiency_perfect(self):
        from monitoring.eval_reporter import EvalReporter
        reporter = EvalReporter(tracer=None)
        scores = reporter.report_call_efficiency(
            tool_calls=[
                {"name": "semantic_search", "args": {"q": "x"}, "success": True, "is_empty": False},
                {"name": "read_chunk", "args": {"id": "1"}, "success": True, "is_empty": False},
            ],
            expected_tool_count=2,
        )
        assert scores["redundancy_rate"] == 0.0
        assert scores["repetition_count"] == 0

    def test_call_efficiency_with_redundancy(self):
        from monitoring.eval_reporter import EvalReporter
        reporter = EvalReporter(tracer=None)
        scores = reporter.report_call_efficiency(
            tool_calls=[
                {"name": "search", "args": {"q": "x"}, "success": True, "is_empty": True},
                {"name": "search", "args": {"q": "x"}, "success": True, "is_empty": False},
                {"name": "search", "args": {"q": "y"}, "success": True, "is_empty": False},
            ],
            expected_tool_count=1,
        )
        assert scores["redundancy_rate"] == round(1 / 3, 4)
        assert scores["repetition_count"] == 1  # search+x appears twice

    def test_call_efficiency_empty(self):
        from monitoring.eval_reporter import EvalReporter
        reporter = EvalReporter(tracer=None)
        scores = reporter.report_call_efficiency(
            tool_calls=[],
            expected_tool_count=0,
        )
        assert scores["redundancy_rate"] == 0.0
        assert scores["repetition_count"] == 0

    def test_planning_perfect_match(self):
        from monitoring.eval_reporter import EvalReporter
        reporter = EvalReporter(tracer=None)
        scores = reporter.report_planning(
            plan_steps_desc=["查找腾讯2024年营收", "分析营收增长原因"],
            ground_truth_hops=["腾讯2024年营收", "营收增长原因分析"],
        )
        assert scores["plan_hop_recall"] == 1.0

    def test_planning_no_steps(self):
        from monitoring.eval_reporter import EvalReporter
        reporter = EvalReporter(tracer=None)
        scores = reporter.report_planning(
            plan_steps_desc=[],
            ground_truth_hops=["hop1", "hop2"],
        )
        assert scores["plan_hop_precision"] == 0.0
        assert scores["plan_hop_recall"] == 0.0

    def test_planning_no_hops(self):
        from monitoring.eval_reporter import EvalReporter
        reporter = EvalReporter(tracer=None)
        scores = reporter.report_planning(
            plan_steps_desc=["step1"],
            ground_truth_hops=[],
        )
        assert scores["plan_hop_precision"] == 0.0
        assert scores["plan_hop_recall"] == 0.0


class TestBadCaseRouter:
    """BadCaseRouter 单元测试"""

    def test_classify_tool_selection_low(self):
        from monitoring.badcase_router import BadCaseRouter
        router = BadCaseRouter()
        bad = router.classify(
            query="test query",
            scores={"tool_selection_f1": 0.3},
        )
        assert len(bad) == 1
        assert bad[0].category == "tool_selection_low"
        assert bad[0].optimize_target == "tool_schema"

    def test_classify_multiple_matches(self):
        from monitoring.badcase_router import BadCaseRouter
        router = BadCaseRouter()
        bad = router.classify(
            query="test query",
            scores={
                "tool_selection_f1": 0.3,
                "arg_quality_avg": 2.0,
                "step_efficiency": 0.3,
            },
        )
        assert len(bad) == 3
        categories = {b.category for b in bad}
        assert "tool_selection_low" in categories
        assert "arg_quality_poor" in categories
        assert "step_inefficient" in categories

    def test_classify_no_match_when_scores_good(self):
        from monitoring.badcase_router import BadCaseRouter
        router = BadCaseRouter()
        bad = router.classify(
            query="test query",
            scores={
                "tool_selection_f1": 0.9,
                "arg_quality_avg": 4.5,
                "redundancy_rate": 0.1,
                "step_efficiency": 0.8,
                "premature_finish": False,
                "plan_hop_recall": 0.9,
            },
        )
        assert len(bad) == 0

    def test_early_finish_detected(self):
        from monitoring.badcase_router import BadCaseRouter
        router = BadCaseRouter()
        bad = router.classify(
            query="test query",
            scores={"premature_finish": True},
        )
        assert len(bad) == 1
        assert bad[0].category == "early_finish"

    def test_accumulate_and_trigger_suggestion(self):
        from monitoring.badcase_router import BadCaseRouter
        router = BadCaseRouter()
        for i in range(5):
            router.classify(
                query=f"query_{i}",
                scores={"tool_selection_f1": 0.3},
            )
        suggestions = router.get_pending_suggestions()
        assert len(suggestions) == 1
        assert suggestions[0]["case_count"] == 5
        assert "tool_schema" in suggestions[0]["target"]

    def test_no_suggestion_below_threshold(self):
        from monitoring.badcase_router import BadCaseRouter
        router = BadCaseRouter()
        for i in range(3):
            router.classify(
                query=f"query_{i}",
                scores={"tool_selection_f1": 0.3},
            )
        suggestions = router.get_pending_suggestions()
        assert len(suggestions) == 0

    def test_reset_clears_accumulator(self):
        from monitoring.badcase_router import BadCaseRouter
        router = BadCaseRouter()
        for i in range(5):
            router.classify(
                query=f"query_{i}",
                scores={"tool_selection_f1": 0.3},
            )
        router.reset()
        suggestions = router.get_pending_suggestions()
        assert len(suggestions) == 0
