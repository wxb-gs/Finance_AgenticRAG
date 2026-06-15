"""Skills 子包 — 从 skills/ 目录按需加载领域技能"""
from agents.agentic.skills.loader import Skill, SkillManager, load_skills

__all__ = ["Skill", "SkillManager", "load_skills"]
