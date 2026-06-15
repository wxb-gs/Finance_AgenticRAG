# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

AgenticRAG is a financial multi-hop QA system with two coexisting Agent architectures, plus SFT+GRPO training pipelines. The shared layers (`retrieval/`, `llm/`, `data/`) are used by both architectures and the training pipeline — never break them.

## Commands

```bash
# Run all agent tests (the only test suite so far)
pytest tests/test_agentic.py -v

# Run a single test
pytest tests/test_agentic.py::TestSkills::test_activate_valid_skill -v

# PEV pipeline evaluation
python scripts/run_cloud_eval.py --model Qwen3-4B --workers 1

# Agent mode evaluation
python scripts/eval_agentic.py --model Qwen3-4B --max-samples 50

# Start API server
python api/server.py          # uvicorn on port 8000
```

## Architecture

### Two pipelines, one router

Both pipelines live under `agents/` and share `retrieval/` + `llm/`. The `PipelineRouter` (`pipeline_router.py`) dispatches based on `mode`:

| Mode | Architecture | Entry |
|------|-------------|-------|
| `pev` | LangGraph state machine: Planner → Executor → Verifier → Synthesizer | `agents/pev/graph.py` |
| `agent` (default) | ReAct while-loop: Think → Act → Observe | `agents/agentic/agent.py` |
| `compare` | Both side-by-side, same query | `PipelineRouter._run_both()` |

### Agentic Agent (`agents/agentic/`)

Claude Code-style design: **Agent = System Prompt + Tools + Simple Loop**.

- `agent.py` — ReAct loop. No graph nodes. Model decides every step via tool calls.
- `tools.py` — `ToolRegistry` with 3 tiers: retrieval (5 tools), meta (`dispatch_subagent`, `activate_skill`, `remember`, `plan_steps`), lifecycle (`finish`). Small model (`model_size="small"`) hides `dispatch_subagent`.
- `skills/` — Each skill is a folder with `SKILL.md` (YAML frontmatter + body). The model reads skill descriptions from the system prompt and calls `activate_skill` to load one. No keyword matching — model-driven activation.
- `sub_agent.py` — Sub-agent types (retrieval/comparison/computation) with restricted tools and iteration caps.
- `context.py` — 3-tier compression (aggressive/summarize_old/preserve_recent), triggers at 80% token usage.
- `memory.py` — File-based persistent memory with MEMORY.md index, jieba tokenizer for recall scoring.
- `prompts.py` — System prompt templates stratified by model size (`small`/`mid`/`large`) × language (`zh`/`en`).

### PEV Pipeline (`agents/pev/`)

LangGraph StateGraph with 5 nodes: `router` → `planner` → `executor` → `verifier` → `synthesizer`, with conditional edges for verification loops.

### LLM layer (`llm/client.py`)

All models accessed through a single OpenAI-compatible interface. Model configs are registered in the `MODEL_CONFIGS` dict — local vLLM or cloud APIs, no difference to callers. Key functions:
- `agent_chat()` / `agent_chat_json()` — Agent reasoning
- `judge_chat()` / `judge_chat_json()` — Evaluation / synthesis
- `get_from_llm()` — Low-level, any registered model

### Retrieval layer (`retrieval/`)

Four search methods + chunk reader: `semantic_search` (FAISS+BGE-M3), `keyword_search` (BM25+jieba), `graph_search` (NetworkX), `hybrid_search` (RRF fusion + CrossEncoder rerank), `read_chunk` (by ID).

## Configuration

All config lives in `config.py`, driven by environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `VLLM_BASE_URL` | `http://localhost:9097/v1` | Agent model endpoint |
| `JUDGE_BASE_URL` | `http://localhost:8086/v1` | Judge model endpoint |
| `AGENT_LLM_MODEL` | `Qwen3-32B` | Default agent model |
| `JUDGE_LLM_MODEL` | `gpt-oss-120b` | Default judge model |
| `AGENT_MODEL_SIZE` | `large` | Agent tier: `small`/`mid`/`large` |
| `PROMPT_LANG` | `zh` | Prompt language: `zh`/`en` |

Add new models by adding entries to `MODEL_CONFIGS` in `llm/client.py`.

## Key design rules

- **PEV is frozen.** New agent features go in `agents/agentic/`. PEV in `agents/pev/` is preserved for backward compatibility and comparison evaluation.
- **Skills are folder-based.** Adding a new skill means creating a folder with `SKILL.md` — zero Python changes. The model reads descriptions and decides when to activate.
- **No model/vendor coupling.** All LLM calls go through `llm/client.py`. Never hardcode API keys or URLs in other modules.
- **Tests live in `tests/` at repo root.** Current test coverage is for the agentic module only (37 tests).
- **Chinese-first.** Default prompt language is `zh`. English translations exist but financial domain terms are primarily Chinese.
