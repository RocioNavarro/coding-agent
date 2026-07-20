"""Integración conductual, neutral y trazable de ProjectProfile."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agents.explorer import ExplorerAgent
from agents.researcher import ResearcherAgent
from agents.tester import (
    CommandOutcome, StaticCommandProvider, TesterAgent as AgentTester, ValidationCommand,
    ValidationExecutor,
)
from core.models import LLMResponse, LLMUsage, Message
from core.profiles import ProjectProfile
from core.task_state import TaskState
from core.settings import AgentSettings
from security.policy_engine import PolicyContext, PolicyEngine


class ExplorerLLM:
    def complete(self, messages, tools=()):
        payload = {
            "summary": "Exploración completada.", "findings": [],
            "recommendations": [], "sources": [], "files_relevant": [],
            "blockers": [], "confidence": 0.9,
        }
        text = json.dumps(payload)
        return LLMResponse(Message("assistant", text), text, [], "fake", LLMUsage(1, 1, 2), 1.0)


class UnusedLLM:
    def complete(self, messages, tools=()):
        raise AssertionError("No se esperaba una llamada al LLM.")


class EmptyMemory:
    def search(self, query: str, *, limit: int = 5):
        return ()


class EmptyRetriever:
    def retrieve_filtered(self, query: str, *, filters=None, limit: int = 5):
        return ()

    def retrieval_audit(self):
        return None


class FakeExecutor(ValidationExecutor):
    def __init__(self, commands: Sequence[str]) -> None:
        self.commands = set(commands)
        self.calls: list[str] = []

    def execute(self, command: str, *, timeout_seconds: float) -> CommandOutcome:
        self.calls.append(command)
        assert command in self.commands
        return CommandOutcome(0, 1.0, "passed")


class RecordingPolicy(PolicyEngine):
    def __init__(self) -> None:
        super().__init__()
        self.commands: list[str] = []

    def evaluate(self, tool, parameters, context, *, modifies_system=None):
        self.commands.append(str(parameters.get("command")))
        return super().evaluate(
            tool, parameters, context, modifies_system=modifies_system
        )


def repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    (root / "src").mkdir(parents=True)
    (root / "README.md").write_text("docs", encoding="utf-8")
    (root / "src/app.py").write_text("VALUE = 1", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname='sample'\n", encoding="utf-8")
    return root


def test_explorer_prioritizes_important_files(tmp_path: Path) -> None:
    root = repository(tmp_path)
    profile = ProjectProfile(important_files=("README.md",))

    report = ExplorerAgent(
        repository_root=root, llm_client=ExplorerLLM(), profile=profile
    ).explore("Cambiar app")

    assert report.relevant_files[0] == "README.md"


def test_explorer_does_not_confirm_technology_without_evidence_and_records_discrepancy(
    tmp_path: Path,
) -> None:
    root = repository(tmp_path)
    profile = ProjectProfile(
        name="hypotheses", expected_technologies=("Unobserved runtime",),
        search_tags=("service",), important_files=("README.md",),
    )
    state = TaskState.create("Explorar")
    explorer = ExplorerAgent(
        repository_root=root, llm_client=ExplorerLLM(), profile=profile
    )

    explorer.run("Explorar", state)

    assert not any(
        "confirmada: Unobserved runtime" in item for item in state.observations
    )
    assert any("no confirmada: Unobserved runtime" in item for item in state.warnings)
    assert any("Archivos importantes utilizados: README.md" in item for item in state.observations)
    assert any("Tags de búsqueda usados: service" in item for item in state.observations)


def test_researcher_uses_confirmed_technology_and_profile_tags() -> None:
    profile = ProjectProfile(
        expected_technologies=("Secondary hypothesis",), search_tags=("official",)
    )
    researcher = ResearcherAgent(
        llm_client=UnusedLLM(), project_memory=EmptyMemory(),
        knowledge_retriever=EmptyRetriever(), profile=profile,
    )
    state = TaskState.create("Investigar")
    state.add_repository_finding("language=Confirmed runtime; evidencia: manifest.")

    query = researcher.build_research_query("Buscar contrato", state)
    filters = researcher._build_rag_filters(state, None)

    assert "Tecnologías confirmadas por Explorer: Confirmed runtime" in query
    assert "Hipótesis secundarias del perfil: Secondary hypothesis" in query
    assert filters["tags"] == ("official",)
    assert any("Tags usados en consultas: official" in item for item in state.observations)


def changed_state() -> TaskState:
    state = TaskState.create("Validar")
    state.record_file_modified("src/app.py")
    state.add_repository_finding("language=runtime; evidencia: manifest.")
    return state


def test_tester_considers_profile_command_and_passes_it_through_policy(
    tmp_path: Path,
) -> None:
    root = repository(tmp_path)
    profile = ProjectProfile(name="generic", suggested_commands={"test": "safe-test"})
    executor = FakeExecutor(("safe-test",))
    policy = RecordingPolicy()
    state = changed_state()
    tester = AgentTester(
        llm_client=UnusedLLM(), repository_root=root, providers=(),
        executor=executor, profile=profile, policy_engine=policy,
    )

    result = tester.run("Validar", state)

    assert result.records[0].origin == "project_profile"
    assert policy.commands == ["safe-test"]
    assert executor.calls == ["safe-test"]
    assert any("sugerido considerado" in item for item in state.observations)
    assert any("sugerido ejecutado" in item for item in state.observations)


def test_tester_prefers_discovered_command_on_contradiction(tmp_path: Path) -> None:
    root = repository(tmp_path)
    discovered = ValidationCommand(
        "repository-test", "configuration", "manifest", check_type="test", priority=10
    )
    profile = ProjectProfile(suggested_commands={"test": "profile-test"})
    executor = FakeExecutor(("repository-test",))
    state = changed_state()
    tester = AgentTester(
        llm_client=UnusedLLM(), repository_root=root,
        providers=(StaticCommandProvider((discovered,)),), executor=executor,
        profile=profile,
    )

    tester.run("Validar", state)

    assert executor.calls == ["repository-test"]
    assert any("desplazado" in warning for warning in state.warnings)


def test_profile_policy_adds_restrictions(tmp_path: Path) -> None:
    root = repository(tmp_path)
    profile = ProjectProfile(additional_policies={"denied_commands": ["safe-test"]})
    context = PolicyContext(agent="tester", workspace=root, profile=profile)

    decision = PolicyEngine().evaluate(
        "run_command", {"command": "safe-test"}, context, modifies_system=False
    )

    assert decision.outcome == "deny"
    assert "project_profile" in decision.reason


def test_profile_cannot_relax_base_command_restrictions(tmp_path: Path) -> None:
    root = repository(tmp_path)
    profile = ProjectProfile(additional_policies={"allowed_tools": ["run_command"]})
    context = PolicyContext(agent="tester", workspace=root, profile=profile)

    decision = PolicyEngine().evaluate(
        "run_command", {"command": "rm -rf ."}, context, modifies_system=False
    )

    assert decision.outcome == "deny"
    assert "destructivo" in decision.reason


def test_default_empty_profile_preserves_explorer_behavior(tmp_path: Path) -> None:
    root = repository(tmp_path)
    without = ExplorerAgent(repository_root=root, llm_client=ExplorerLLM())
    empty = ExplorerAgent(
        repository_root=root, llm_client=ExplorerLLM(), profile=ProjectProfile()
    )

    assert without.explore("Cambiar app") == empty.explore("Cambiar app")


def test_researcher_from_settings_uses_profile_priority_domains(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "not-used-by-test")
    profile = ProjectProfile(priority_web_domains=("priority.example.com",))

    researcher = ResearcherAgent.from_settings(
        llm_client=UnusedLLM(), project_memory=EmptyMemory(),
        knowledge_retriever=EmptyRetriever(),
        settings=AgentSettings(web_search_enabled=True, web_search_config={}),
        profile=profile,
    )

    assert researcher.web_search is not None
    assert researcher.web_search.config.priority_domains == ("priority.example.com",)
