"""Escenario integral: persistencia real y exploración incremental."""

from agents.orchestrator import MainAgent
from agents.project_memory import ProjectMemory
from core.models import PlanReview
from tests.integration.fakes import (
    FakeObservability, FakeRAG, ScriptedPlanner, StaticTaskAnalyzer,
)
from tests.integration.scenario_support import explorer_for, researcher_for


def run_memory_analysis(repository, storage, task_id):
    observed = FakeObservability()
    memory = ProjectMemory(
        repository.root,
        identifier="persistent-scenario",
        storage_root=storage,
        observability=observed,
    )
    agent = MainAgent(
        task_analyzer=StaticTaskAnalyzer("analysis", research_required=True),
        plan_generator=ScriptedPlanner(("1. Report current architecture evidence",)),
        explorer=explorer_for(
            repository.root, ("src/component.py",), memory=memory,
            observability=observed,
        ),
        researcher=researcher_for(memory, FakeRAG()),
        implementer=None,
        tester=None,
        reviewer=None,
        project_memory=memory,
        observability=observed,
    )
    result = agent.run(
        "Analyze component architecture and validation",
        lambda _plan: PlanReview("approve"),
        task_id=task_id,
    )
    return result, memory, observed


def test_second_execution_reloads_memory_and_reduces_exploration(
    persistent_memory_repository,
) -> None:
    storage = persistent_memory_repository.memory_root
    assert storage is not None

    first, first_memory, _ = run_memory_analysis(
        persistent_memory_repository, storage, "memory-first"
    )
    second, second_memory, observed = run_memory_analysis(
        persistent_memory_repository, storage, "memory-second"
    )

    assert first.status == second.status == "completed"
    assert first_memory is not second_memory
    assert first_memory.path == second_memory.path
    persisted = ProjectMemory(
        persistent_memory_repository.root,
        identifier="persistent-scenario",
        storage_root=storage,
    ).load().data
    assert persisted["architecture"]
    assert "src/component.py" in persisted["important_files"]
    assert persisted["technologies"]
    assert persisted["known_commands"]
    assert persisted["file_fingerprints"]
    assert len(persisted["previous_tasks"]) == 2

    first_state = first.task_state
    second_state = second.task_state
    assert any("Estrategia de exploración: full" in item for item in first_state.observations)
    assert any("Estrategia de exploración: incremental" in item for item in second_state.observations)
    assert len(second_state.files_read) < len(first_state.files_read)
    assert any("Archivos evitados:" in item and "ninguno" not in item
               for item in second_state.observations)
    assert any("Archivos revalidados:" in item and "ninguno" not in item
               for item in second_state.observations)
    assert any("Memoria reutilizada como pista" in item
               for item in second_state.observations)
    assert any(source.origin == "project_memory" for source in second_state.sources)
    strategy_event = next(
        event for event in observed.events if event.name == "explorer-strategy"
    )
    assert strategy_event.payload["strategy"] == "incremental"
    assert strategy_event.payload["files_avoided"] > 0
