"""Composición legible de agentes reales para escenarios integrales."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from agents.explorer import ExplorerAgent
from agents.implementer import ImplementerAgent, ScopedWritePolicy
from agents.researcher import EvidenceFragment, ResearcherAgent
from agents.tester import (
    StaticCommandProvider, TesterAgent, TesterLimits, ValidationCommand,
)
from core.planned_operations import PolicyPreflight
from core.settings import AgentSettings
from security.policy_engine import (
    AgentToolPermissions, PolicyContext, PolicyEngine,
)
from tests.integration.fakes import ScriptedLLM


def agent_payload(
    summary: str,
    *,
    files: Sequence[str] = (),
    sources: Sequence[dict[str, str]] = (),
    findings: Sequence[str] = (),
    confidence: float = 0.9,
) -> dict[str, object]:
    return {
        "summary": summary,
        "findings": list(findings),
        "recommendations": ["Keep the result scoped and validated."],
        "sources": list(sources),
        "files_relevant": list(files),
        "blockers": [],
        "confidence": confidence,
    }


def explorer_for(
    root: Path,
    files: Sequence[str],
    *,
    memory=None,
    observability=None,
) -> ExplorerAgent:
    sources = [
        {"origin": "repository", "reference": path, "summary": "repository evidence"}
        for path in files
    ]
    return ExplorerAgent(
        repository_root=root,
        llm_client=ScriptedLLM((agent_payload(
            "Repository structure and relevant files were inspected.",
            files=files,
            sources=sources,
            findings=(
                "Architecture conventions are evidenced by the repository structure.",
                "The requested change has localized impact on the selected file.",
            ),
        ),)),
        project_memory=memory,
        observability=observability,
    )


class StructuredCommandExplorer:
    """Decora Explorer con un comando explícitamente estructurado para el escenario."""

    def __init__(self, explorer: ExplorerAgent, command: str) -> None:
        self.explorer = explorer
        self.command = command
        self.calls = 0

    def run(self, instruction, state, *args, **kwargs):
        self.calls += 1
        result = self.explorer.run(instruction, state, *args, **kwargs)
        state.add_observation(f"Structured command: {self.command}")
        return result


def researcher_for(memory, rag, web=None) -> ResearcherAgent:
    from tests.integration.fakes import FixedSufficiencyEvaluator
    return ResearcherAgent(
        llm_client=ScriptedLLM((agent_payload(
            "Technical evidence is sufficient and traceable.",
            findings=("Recovered evidence supports the requested behavior.",),
        ),)),
        project_memory=memory,
        knowledge_retriever=rag,
        web_search=web,
        sufficiency_evaluator=FixedSufficiencyEvaluator(sufficient=True),
    )


def implementer_for(root: Path, path: str, old: str, new: str) -> ImplementerAgent:
    response = {
        "summary": "Localized change applied.",
        "proposed_change": "Replace the exact selected value.",
        "conventions_check": ["The localized repository convention is preserved."],
        "changes": [{
            "path": path,
            "old_text": old,
            "new_text": new,
            "explanation": "Exact replacement within the authorized fragment.",
        }],
    }
    return ImplementerAgent(
        llm_client=ScriptedLLM((response,)),
        write_policy=ScopedWritePolicy(root),
    )


def build_tester(
    root: Path,
    command: str,
    runner,
    *,
    observability=None,
    progress_monitor=None,
    max_retries: int = 0,
) -> TesterAgent:
    return TesterAgent(
        llm_client=ScriptedLLM(()),
        repository_root=root,
        providers=(StaticCommandProvider((ValidationCommand(
            command, "configuration", "structured integration command", "validation"
        ),)),),
        executor=runner,
        limits=TesterLimits(max_retries=max_retries),
        observability=observability,
        progress_monitor=progress_monitor,
    )


def preflight_for(
    root: Path,
    *,
    allowed_tools: frozenset[str] | None = None,
    approval_tools: frozenset[str] = frozenset(),
) -> PolicyPreflight:
    return PolicyPreflight(
        PolicyEngine(),
        PolicyContext(
            agent="main",
            workspace=root,
            permissions=AgentToolPermissions(allowed_tools, approval_tools),
            settings=AgentSettings(supervision_enabled=False),
        ),
    )
