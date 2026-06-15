"""FastAPI 演示接口"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="AgenticRAG Demo", version="1.0")


class QueryRequest(BaseModel):
    question: str
    verbose: bool = False
    mode: str = "agent"


class QueryResponse(BaseModel):
    answer: str
    query_type: str
    iteration_count: int
    total_tool_calls: int
    evidence_summary: list[dict]
    trace: list[dict]
    latency: float
    metadata: dict = {}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    from agents.pev.graph import run_query
    from llm.client import stats

    stats.reset()
    t0 = time.time()
    state = run_query(req.question)
    latency = time.time() - t0

    # 简化 evidence
    evidence_summary = []
    for e in state.get("evidence", []):
        evidence_summary.append({
            "step_id": e.get("step_id"),
            "sub_query": e.get("sub_query"),
            "tool": e.get("tool"),
            "num_results": len(e.get("results", [])),
            "top_chunk_ids": [r.get("chunk_id", "") for r in e.get("results", [])[:3]],
        })

    return QueryResponse(
        answer=state.get("final_answer", ""),
        query_type=state.get("query_type", ""),
        iteration_count=state.get("iteration_count", 0),
        total_tool_calls=state.get("total_tool_calls", 0),
        evidence_summary=evidence_summary,
        trace=state.get("trace", []),
        latency=round(latency, 2),
    )


@app.get("/health")
def health():
    from config import AGENT_CONFIG
    return {
        "status": "ok",
        "pev_available": True,
        "agent_available": True,
        "mode": AGENT_CONFIG.get("default_mode", "agent"),
        "agent_model_size": AGENT_CONFIG.get("agent_model_size", "large"),
    }


@app.post("/query/agent", response_model=QueryResponse)
def query_agent_endpoint(req: QueryRequest):
    """显式 Agent 模式"""
    from pipeline_router import PipelineRouter
    from config import AGENT_CONFIG
    from llm.client import stats

    stats.reset()
    router = PipelineRouter(AGENT_CONFIG)
    result = router.run(query=req.question, mode="agent")

    return QueryResponse(
        answer=result["answer"],
        query_type="agent",
        iteration_count=result["iterations"],
        total_tool_calls=result["total_tool_calls"],
        evidence_summary=result.get("metadata", {}).get("evidence_summary", []),
        trace=result["trace"],
        latency=result["latency_ms"] / 1000,
    )


@app.post("/query/compare")
def query_compare_endpoint(req: QueryRequest):
    """对比模式：同时跑 PEV 和 Agent"""
    from pipeline_router import PipelineRouter
    from config import AGENT_CONFIG

    router = PipelineRouter(AGENT_CONFIG)
    return router.run(query=req.question, mode="compare")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
