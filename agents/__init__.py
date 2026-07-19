"""Abstracciones propias para agentes especializados."""

from agents.base import (
    AgentContext,
    AgentExecutionError,
    AgentInput,
    BaseAgent,
    StubAgent,
)
from agents.explorer import ExplorerAgent, ExplorerReport, build_explorer_registry
from agents.repository_detection import (
    BuildSystemDetector,
    LanguageDetector,
    RepositoryDetector,
    TechnologyDetector,
)

__all__ = [
    "AgentContext",
    "AgentExecutionError",
    "AgentInput",
    "BaseAgent",
    "BuildSystemDetector",
    "ExplorerAgent",
    "ExplorerReport",
    "LanguageDetector",
    "RepositoryDetector",
    "StubAgent",
    "TechnologyDetector",
    "build_explorer_registry",
]
