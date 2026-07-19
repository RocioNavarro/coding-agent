"""Abstracciones propias para agentes especializados."""

from agents.base import (
    AgentContext,
    AgentExecutionError,
    AgentInput,
    BaseAgent,
    StubAgent,
)
from agents.explorer import ExplorerAgent, ExplorerReport, build_explorer_registry
from agents.context_manager import ContextManager, StateContextManager
from agents.implementer import (
    ImplementerAgent,
    ImplementerBlockedError,
    ImplementerResult,
    ScopedWritePolicy,
    WritePolicy,
)
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
    "ContextManager",
    "ExplorerAgent",
    "ExplorerReport",
    "EvidenceFragment",
    "EvidenceSufficiencyEvaluator",
    "KnowledgeRetriever",
    "ImplementerAgent",
    "ImplementerBlockedError",
    "ImplementerResult",
    "LanguageDetector",
    "RepositoryDetector",
    "ProjectMemoryProvider",
    "ResearcherAgent",
    "ResearcherResult",
    "StubAgent",
    "StateContextManager",
    "ScopedWritePolicy",
    "TechnologyDetector",
    "WebSearchProvider",
    "WritePolicy",
    "build_explorer_registry",
]
