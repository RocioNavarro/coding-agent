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
from agents.researcher import (
    EvidenceFragment,
    EvidenceSufficiencyEvaluator,
    KnowledgeRetriever,
    ProjectMemoryProvider,
    ResearcherAgent,
    ResearcherResult,
    WebSearchProvider,
)

__all__ = [
    "AgentContext",
    "AgentExecutionError",
    "AgentInput",
    "BaseAgent",
    "BuildSystemDetector",
    "ExplorerAgent",
    "ExplorerReport",
    "EvidenceFragment",
    "EvidenceSufficiencyEvaluator",
    "KnowledgeRetriever",
    "LanguageDetector",
    "RepositoryDetector",
    "ProjectMemoryProvider",
    "ResearcherAgent",
    "ResearcherResult",
    "StubAgent",
    "TechnologyDetector",
    "WebSearchProvider",
    "build_explorer_registry",
]
