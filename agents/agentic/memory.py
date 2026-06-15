"""Memory 持久化系统 — 跨会话的知识记忆"""
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import jieba

MEMORY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "memory"
)


@dataclass
class Memory:
    name: str
    type: str                      # evidence | contradiction | gap | pattern
    description: str
    content: str
    source_query: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    access_count: int = 0

    def to_markdown(self) -> str:
        return f"""---
name: {self.name}
description: {self.description}
metadata:
  type: {self.type}
  source_query: {self.source_query}
  created_at: {self.created_at}
  access_count: {self.access_count}
---

{self.content}
"""


class MemoryManager:
    def __init__(self, base_dir: str = MEMORY_DIR):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index: dict[str, Memory] = {}
        self._load_index()

    def _load_index(self):
        index_file = self.base_dir / "MEMORY.md"
        if not index_file.exists():
            return
        for line in index_file.read_text(encoding="utf-8").split("\n"):
            match = re.match(r"- \[(.*?)\]\((.*?)\) — (.*)", line)
            if match:
                name = match.group(1)
                desc = match.group(3)
                mem_file = self.base_dir / f"{match.group(2)}"
                if mem_file.exists():
                    self.index[name] = Memory(
                        name=name, type="evidence", description=desc,
                        content="", source_query="",
                    )

    def _update_index_file(self):
        lines = []
        for mem in self.index.values():
            filename = f"{mem.name}.md"
            lines.append(f"- [{mem.name}]({filename}) — {mem.description}")
        (self.base_dir / "MEMORY.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )

    def save(self, content: str, mem_type: str, query: str) -> Memory:
        """保存记忆"""
        name = self._generate_name(content)
        description = content[:150].replace("\n", " ")

        mem = Memory(
            name=name,
            type=mem_type,
            description=description,
            content=content,
            source_query=query,
        )
        self.index[name] = mem
        filepath = self.base_dir / f"{name}.md"
        filepath.write_text(mem.to_markdown(), encoding="utf-8")
        self._update_index_file()
        return mem

    def recall(self, query: str, top_k: int = 5) -> list[Memory]:
        """关键词召回相关记忆"""
        scored = []
        query_tokens = set(jieba.cut(query))
        for mem in self.index.values():
            score = sum(1 for t in query_tokens
                       if t in mem.description or t in mem.content)
            if score > 0:
                scored.append((mem, score))
        scored.sort(key=lambda x: (x[1], x[0].access_count), reverse=True)
        return [m for m, _ in scored[:top_k]]

    def forget(self, name: str):
        filepath = self.base_dir / f"{name}.md"
        if filepath.exists():
            filepath.unlink()
        if name in self.index:
            del self.index[name]
        self._update_index_file()

    def _generate_name(self, content: str) -> str:
        """生成 kebab-case 标识符"""
        name = content[:60].strip().lower()
        name = re.sub(r'[^\w\-]', '-', name)
        name = re.sub(r'-+', '-', name).strip('-')
        base = name[:40] or "untitled"
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{base}-{ts}"
