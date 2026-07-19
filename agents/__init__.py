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
from agents.tester import (
    StaticCommandProvider,
    SubprocessValidationExecutor,
    TaskStateCommandProvider,
    TesterAgent,
    TesterLimits,
    TesterResult,
    ValidationCommand,
    ValidationCommandProvider,
    ValidationExecutor,
    ValidationSafetyPolicy,
)
from agents.reviewer import (
    DiffProvider,
    DiffSnapshot,
    GitDiffProvider,
    ReviewIssue,
    ReviewerAgent,
    ReviewerResult,
)

__all__ = [
    "AgentContext",
    "AgentExecutionError",
    "AgentInput",
    "BaseAgent",
    "BuildSystemDetector",
    "ContextManager",
    "DiffProvider",
    "DiffSnapshot",
    "ExplorerAgent",
    "ExplorerReport",
    "EvidenceFragment",
    "EvidenceSufficiencyEvaluator",
    "KnowledgeRetriever",
    "GitDiffProvider",
    "ImplementerAgent",
    "ImplementerBlockedError",
    "ImplementerResult",
    "LanguageDetector",
    "RepositoryDetector",
    "ProjectMemoryProvider",
    "ResearcherAgent",
    "ResearcherResult",
    "ReviewIssue",
    "ReviewerAgent",
    "ReviewerResult",
    "StubAgent",
    "StateContextManager",
    "StaticCommandProvider",
    "SubprocessValidationExecutor",
    "TaskStateCommandProvider",
    "ScopedWritePolicy",
    "TechnologyDetector",
    "TesterAgent",
    "TesterLimits",
    "TesterResult",
    "ValidationCommand",
    "ValidationCommandProvider",
    "ValidationExecutor",
    "ValidationSafetyPolicy",
    "WebSearchProvider",
    "WritePolicy",
    "build_explorer_registry",
]
