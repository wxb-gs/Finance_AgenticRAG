"""全局配置 — 所有路径支持环境变量覆盖"""
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 模型路径（环境变量覆盖）
MODEL_HUB = os.environ.get("MODEL_HUB", os.path.join(PROJECT_ROOT, "models"))
BGE_M3_PATH = os.environ.get("BGE_M3_PATH", os.path.join(MODEL_HUB, "bge-m3"))
BGE_RERANKER_PATH = os.environ.get("BGE_RERANKER_PATH", os.path.join(MODEL_HUB, "bge-reranker-v2-m3"))

# 数据和索引路径
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "datasets")
INDEX_DIR = os.path.join(PROJECT_ROOT, "data", "indexes")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
TMP_DIR = os.path.join(PROJECT_ROOT, "tmp")

# Agent 推理模型（环境变量覆盖）
AGENT_LLM_MODEL = os.environ.get("AGENT_LLM_MODEL", "Qwen3-32B")
# Prompt 语言：en / zh
PROMPT_LANG = os.environ.get("PROMPT_LANG", "zh")
# 评测 Judge 用
JUDGE_LLM_MODEL = os.environ.get("JUDGE_LLM_MODEL", "gpt-oss-120b")
# 数据合成用
SYNTH_LLM_MODEL = os.environ.get("SYNTH_LLM_MODEL", "gpt-oss-120b")

# 检索参数
RERANK_TOP_K = 5
SEMANTIC_TOP_K = 20
BM25_TOP_K = 20

# 金融语料配置（环境变量覆盖）
NEWS_CORPUS_DIR = os.environ.get("NEWS_CORPUS_DIR", "")
NEWS_INDEX_DIR = os.environ.get("NEWS_INDEX_DIR", "")

# 实际使用的路径（优先金融语料）
ACTIVE_DATA_DIR = NEWS_CORPUS_DIR or DATA_DIR
ACTIVE_INDEX_DIR = NEWS_INDEX_DIR or INDEX_DIR

# 确保目录存在
for d in [DATA_DIR, INDEX_DIR, RESULTS_DIR, TMP_DIR, ACTIVE_DATA_DIR, ACTIVE_INDEX_DIR]:
    if d:
        os.makedirs(d, exist_ok=True)

# ── Agentic Agent 配置 ──
AGENT_CONFIG = {
    "default_mode": "agent",         # pev | agent | compare
    "agent_model": os.environ.get("AGENT_LLM_MODEL", "Qwen3-32B"),
    "agent_model_size": os.environ.get("AGENT_MODEL_SIZE", "large"),
    "agent_language": os.environ.get("PROMPT_LANG", "zh"),
    "agent_max_iterations": int(os.environ.get("AGENT_MAX_ITERATIONS", "15")),
    "agent_enable_subagents": os.environ.get("AGENT_ENABLE_SUBAGENTS", "true").lower() == "true",
    "pev_enable_verifier": True,
    "pev_enabled_tools": None,

    # ── 上下文压缩配置 ──
    "context_max_tokens": int(os.environ.get("CONTEXT_MAX_TOKENS", "32768")),
    "context_compress_threshold": float(os.environ.get("CONTEXT_COMPRESS_THRESHOLD", "0.75")),
    "context_output_reserve": int(os.environ.get("CONTEXT_OUTPUT_RESERVE", "4096")),
    "context_snip_enabled": os.environ.get("CONTEXT_SNIP_ENABLED", "false").lower() == "true",
    "context_micro_keep_recent": int(os.environ.get("CONTEXT_MICRO_KEEP_RECENT", "5")),
    "context_micro_trigger_threshold": int(os.environ.get("CONTEXT_MICRO_TRIGGER_THRESHOLD", "10")),
    "context_micro_idle_minutes": int(os.environ.get("CONTEXT_MICRO_IDLE_MINUTES", "60")),
    "context_sm_min_tokens": int(os.environ.get("CONTEXT_SM_MIN_TOKENS", "10000")),
    "context_sm_step_tokens": int(os.environ.get("CONTEXT_SM_STEP_TOKENS", "5000")),
    "context_sm_keep_recent": int(os.environ.get("CONTEXT_SM_KEEP_RECENT", "8")),
    "context_summary_max_tokens": int(os.environ.get("CONTEXT_SUMMARY_MAX_TOKENS", "4000")),
    "context_circuit_breaker": int(os.environ.get("CONTEXT_CIRCUIT_BREAKER", "3")),
}

# ── MCP Client 配置 ──
MCP_SERVERS = [
    {
        "name": "sqlite_default",
        "transport": "stdio",
        "command": ["python", "-m", "mcp.servers.sqlite_server"],
        "args": ["--db", os.path.join(DATA_DIR, "sqlite", "default.db")],
    },
    {
        "name": "python_default",
        "transport": "stdio",
        "command": ["python", "-m", "mcp.servers.python_server"],
    },
]

# 环境变量覆盖 MCP 配置
_mcp_override = os.environ.get("MCP_SERVERS_CONFIG")
if _mcp_override:
    import json as _json
    if _mcp_override.endswith(".json"):
        with open(_mcp_override) as _f:
            MCP_SERVERS = _json.load(_f)
    else:
        MCP_SERVERS = _json.loads(_mcp_override)

# ── Langfuse 监控配置 ──
LANGFUSE_ENABLED = os.environ.get("LANGFUSE_ENABLED", "false").lower() == "true"
LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")
LANGFUSE_FLUSH_INTERVAL = int(os.environ.get("LANGFUSE_FLUSH_INTERVAL", "5"))
LANGFUSE_SAMPLE_RATE = float(os.environ.get("LANGFUSE_SAMPLE_RATE", "1.0"))
