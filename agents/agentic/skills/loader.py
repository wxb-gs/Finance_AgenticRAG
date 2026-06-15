"""技能加载器 — 扫描 skills/ 子目录，每个文件夹一个技能

Claude Code 风格：每技能一个文件夹，内含 SKILL.md。
模型根据 description 自主判断激活，不做关键词匹配。
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Skill:
    name: str
    description: str
    content: str          # SKILL.md body (prompt extension)
    path: Path            # skill folder path

    def to_listing(self) -> str:
        """生成系统提示中的技能摘要条目"""
        return f"- **{self.name}**: {self.description}"


def load_skills(skills_dir: Path | None = None) -> dict[str, Skill]:
    """扫描目录：每个子文件夹包含 SKILL.md 即为一个技能"""
    if skills_dir is None:
        skills_dir = Path(__file__).parent
    registry: dict[str, Skill] = {}
    for skill_dir in sorted(skills_dir.iterdir()):
        if not skill_dir.is_dir() or skill_dir.name.startswith("_"):
            continue
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            continue
        skill = _parse_skill_file(skill_file, skill_dir)
        if skill:
            registry[skill.name] = skill
    return registry


def _parse_skill_file(filepath: Path, skill_dir: Path) -> Skill | None:
    """解析 SKILL.md，提取 YAML frontmatter + body"""
    text = filepath.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
    if not match:
        return None
    try:
        meta = yaml.safe_load(match.group(1))
        body = match.group(2).strip()
        return Skill(
            name=meta["name"],
            description=meta.get("description", "").strip(),
            content=body,
            path=skill_dir,
        )
    except (yaml.YAMLError, KeyError):
        return None


class SkillManager:
    """技能管理器

    模型根据 description 自主判断激活哪个技能——
    不做关键词匹配，完全交给模型决策。
    """

    def __init__(self, model_size: str = "large", skills_dir: Path | None = None):
        self.model_size = model_size
        self.skills_dir = skills_dir or Path(__file__).parent
        self.registry = load_skills(self.skills_dir)
        self.active: dict[str, Skill] = {}

    def get_listing_text(self) -> str:
        """生成技能摘要列表，注入 System Prompt"""
        if not self.registry:
            return ""
        lines = ["## 可用技能", ""]
        for skill in self.registry.values():
            lines.append(skill.to_listing())
        lines.append("")
        lines.append(
            "当查询匹配某个技能的描述时，调用 `activate_skill` 激活它。"
            "激活后你将获得该技能的完整工作流指引。"
            "如不需技能即可回答，则不必激活。"
        )
        return "\n".join(lines)

    def activate(self, name: str) -> Skill | None:
        """激活技能，返回其完整内容"""
        skill = self.registry.get(name)
        if skill is None:
            # 模糊匹配：尝试部分匹配
            for sname, s in self.registry.items():
                if name in sname or sname in name:
                    skill = s
                    break
        if skill:
            self.active[name] = skill
        return skill

    def get_active_skill_names(self) -> list[str]:
        return list(self.active.keys())

    def build_system_prompt(self, language: str = "zh") -> str:
        """组装 System Prompt：基础 + 工具描述 + 技能列表"""
        from agents.agentic.prompts import get_system_prompt, get_tool_descriptions

        prompt = get_system_prompt(self.model_size, language)
        prompt += "\n" + get_tool_descriptions(language)

        listing = self.get_listing_text()
        if listing:
            prompt += "\n" + listing

        return prompt
