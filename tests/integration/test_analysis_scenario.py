"""Escenario integral: análisis aprobado sin modificaciones."""

from agents.orchestrator import MainAgent
from core.models import PlanReview
from tests.integration.fakes import (
    FakeObservability, ForbiddenRunner, ScriptedPlanner, ScriptedReviewer,
    StaticTaskAnalyzer,
)
from tests.integration.scenario_support import explorer_for


def test_analysis_only_runs_explorer_plan_and_optional_reviewer(
    analysis_repository,
) -> None:
    observed = FakeObservability()
    implementer = ForbiddenRunner("implementer")
    tester = ForbiddenRunner("tester")
    reviewer = ScriptedReviewer(("approved",))
    agent = MainAgent(
        task_analyzer=StaticTaskAnalyzer("analysis"),
        plan_generator=ScriptedPlanner(("1. Inspect structure\n2. Report evidence",)),
        explorer=explorer_for(
            analysis_repository.root, ("src/component.py",),
            observability=observed,
        ),
        researcher=None,
        implementer=implementer,
        tester=tester,
        reviewer=reviewer,
        review_analysis_tasks=True,
        observability=observed,
    )

    result = agent.run(
        "Analyze the component without changing files",
        lambda _plan: PlanReview("approve"),
        task_id="analysis-scenario",
    )

    state = result.task_state
    assert result.status == "completed"
    assert result.selected_agents == ("explorer", "reviewer")
    assert state.task_id == "analysis-scenario"
    assert state.approved_plan is not None
    assert "Revisar la estructura" in state.approved_plan
    assert "Generar el informe técnico final" in state.approved_plan
    assert "modo de lectura" in state.approved_plan
    assert any("Estrategia de exploración: full" in item for item in state.observations)
    assert reviewer.calls == ["Revisar el resultado del análisis"]
    assert implementer.calls == tester.calls == 0
    assert state.files_modified == ()
    assert state.commands_executed == ()
    assert analysis_repository.modified_files() == ()
    names = [event.name for event in observed.events]
    assert names[0] == "orchestrated-task"
    assert "explorer" in names
    assert "reviewer" in names
    assert names[-1] == "orchestrated-task-finished"
    assert all(
        event.parent_event_id == "task:analysis-scenario"
        for event in observed.events
        if event.name in {"explorer", "reviewer", "orchestrated-task-finished"}
    )
