"""上下文压缩 — 三层策略按模型规格"""
import tiktoken
from typing import Literal

from agents.agentic.types import CompressionEvent


def count_tokens(messages: list[dict], model: str = "gpt-4") -> int:
    """估算消息列表的 token 数"""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    total = 0
    for msg in messages:
        for key in ("content", "tool_calls", "tool_call_id"):
            if key in msg and msg[key]:
                if isinstance(msg[key], str):
                    total += len(enc.encode(msg[key]))
                elif isinstance(msg[key], list):
                    for item in msg[key]:
                        if isinstance(item, dict) and "function" in item:
                            total += len(enc.encode(str(item["function"])))
        if "role" in msg:
            total += 4
    return total


class ContextManager:
    """三层上下文压缩策略"""

    def __init__(self, model_size: Literal["small", "mid", "large"],
                 max_tokens: int | None = None):
        self.model_size = model_size
        if max_tokens is not None:
            self.max_tokens = max_tokens
        elif model_size == "small":
            self.max_tokens = 8192
        elif model_size == "mid":
            self.max_tokens = 16384
        else:
            self.max_tokens = 32768

    def should_compress(self, messages: list[dict]) -> bool:
        current = count_tokens(messages)
        return current > self.max_tokens * 0.8

    def compress(self, messages: list[dict]) -> tuple[list[dict], CompressionEvent]:
        before = count_tokens(messages)

        if self.model_size == "small":
            result, strategy = self._aggressive(messages), "aggressive"
        elif self.model_size == "mid":
            result, strategy = self._summarize_old(messages), "summarize_old"
        else:
            result, strategy = self._preserve_recent(messages), "preserve_recent"

        after = count_tokens(result)
        return result, CompressionEvent(
            before_tokens=before,
            after_tokens=after,
            strategy=strategy,
        )

    def _aggressive(self, messages: list[dict]) -> list[dict]:
        if len(messages) <= 5:
            return messages
        system_msgs = [m for m in messages if m.get("role") == "system"]
        rest = [m for m in messages if m.get("role") != "system"]
        old = rest[1:-4]
        summary = self._summarize_tool_results(old)
        return [
            *system_msgs,
            {"role": "system", "content": f"[前期检索摘要]\n{summary}"},
            *rest[-4:],
        ]

    def _summarize_old(self, messages: list[dict]) -> list[dict]:
        if len(messages) <= 8:
            return messages
        system_msgs = [m for m in messages if m.get("role") == "system"]
        rest = [m for m in messages if m.get("role") != "system"]
        old = rest[1:-6]
        summary = self._summarize_tool_results(old)
        return [
            *system_msgs,
            {"role": "system", "content": f"[历史检索摘要]\n{summary}"},
            *rest[-6:],
        ]

    def _preserve_recent(self, messages: list[dict]) -> list[dict]:
        kept = []
        removed_tool = 0
        for i, msg in enumerate(messages):
            is_old_tool = (msg.get("role") == "tool" and
                           i < len(messages) - 12 and
                           removed_tool < 20)
            if is_old_tool:
                removed_tool += 1
                continue
            kept.append(msg)
        return kept

    def _summarize_tool_results(self, messages: list[dict]) -> str:
        data_points = []
        seen_chunks = set()
        for msg in messages:
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                for line in content.split("\n"):
                    if line.startswith("[") and "chunk_id=" in line:
                        chunk_id = line.split("chunk_id=")[1].split()[0]
                        if chunk_id not in seen_chunks:
                            seen_chunks.add(chunk_id)
                            text = line.split("\n    ")[-1][:150] if "\n    " in line else ""
                            data_points.append(f"- {text} | source:{chunk_id}")
        if not data_points:
            return "（早期轮次无有效检索结果）"
        return "[数据]\n" + "\n".join(data_points[:20])
