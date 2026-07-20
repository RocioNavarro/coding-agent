"""Escenario integral: cambio localizado con evidencia y validación."""

from agents.orchestrator import MainAgent
from agents.researcher import EvidenceFragment
from agents.tester import CommandOutcome
from core.models import PlanReview
from tests.integration.fakes import (
    FakeMemory, FakeObservability, FakeRAG, ScriptedCommandRunner,
    ScriptedPlanner, ScriptedReviewer, StaticTaskAnalyzer,
)
from tests.integration.scenario_support import (
    StructuredCommandExplorer, explorer_for, implementer_for, preflight_for,
    build_tester, researcher_for,
)


def test_simple_change_runs_complete_guarded_pipeline(
    simple_change_repository,
) -> None:
    observed = FakeObservability()
    command = "project-check"
    runner = ScriptedCommandRunner({
        command: (CommandOutcome(0, 2.0, stdout="validation passed"),)
    })
    rag = FakeRAG((EvidenceFragment(
        "rag", "docs://localized-contract", "Localized value contract.", 0.95
    ),), observability=observed)
    explorer = StructuredCommandExplorer(
        explorer_for(
            simple_change_repository.root, ("src/value.py",),
            observability=observed,
        ),
        command,
    )
    implementer = implementer_for(
        simple_change_repository.root,
        "src/value.py",
        "VALUE = 'before'",
        "VALUE = 'after'",
    )
    agent = MainAgent(
        task_analyzer=StaticTaskAnalyzer("change"),
        plan_generator=ScriptedPlanner((
            "1. Change the selected value\n2. Run the structured validation",
        )),
        explorer=explorer,
        researcher=researcher_for(FakeMemory(observability=observed), rag),
        implementer=implementer,
        tester=build_tester(
            simple_change_repository.root, command, runner,
            observability=observed,
        ),
        reviewer=ScriptedReviewer(("approved",)),
        policy_preflight=preflight_for(simple_change_repository.root),
        observability=observed,
    )

    result = agent.run(
        "Change the selected value from before to after",
        lambda _plan: PlanReview("approve"),
        task_id="simple-change",
    )

    state = result.task_state
    assert result.status == "completed"
    assert result.selected_agents == (
        "explorer", "researcher", "implementer", "tester", "reviewer"
    )
    assert state.evidence_assessment is not None
    assert state.evidence_assessment.status == "sufficient"
    assert {item["operation_type"] for item in state.planned_operations} == {
        "modify_file", "run_command"
    }
    assert all(item["outcome"] == "allow" for item in state.policy_preflight)
    assert state.files_modified == ("src/value.py",)
    assert "src/value.py" in state.files_read
    assert state.commands_executed == (command,)
    assert simple_change_repository.read("src/value.py") == "VALUE = 'after'\n"
    assert simple_change_repository.modified_files() == ("src/value.py",)
    assert runner.calls == [(command, 60.0)]
    assert any("Implementer modo=apply_changes" in item for item in state.observations)
    root = "task:simple-change"
    children = [event for event in observed.events if event.parent_event_id == root]
    assert {event.name for event in children} >= {
        "explorer", "researcher", "policy-preflight",
        "evidence-sufficiency-policy", "implementer", "tester", "reviewer",
    }
