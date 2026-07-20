"""Escenario integral: validación repetida falla y produce estrategia controlada."""

from agents.orchestrator import MainAgent
from agents.researcher import EvidenceFragment
from agents.tester import CommandOutcome
from core.models import PlanReview
from core.progress import ProgressLimits, ProgressMonitor
from tests.integration.fakes import (
    FakeMemory, FakeObservability, FakeRAG, ScriptedCommandRunner,
    ScriptedPlanner, ScriptedReviewer, StateRecordingImplementer,
    StaticTaskAnalyzer,
)
from tests.integration.scenario_support import (
    StructuredCommandExplorer, explorer_for, preflight_for, researcher_for,
    build_tester,
)


def test_repeated_command_failure_reaches_orchestrator_without_infinite_retry(
    failed_command_repository,
) -> None:
    observed = FakeObservability()
    command = "project-check"
    failure = CommandOutcome(1, 3.0, stderr="deterministic failure")
    runner = ScriptedCommandRunner({command: (failure, failure)})
    progress = ProgressMonitor(
        ProgressLimits(command_error_repeats=2), observability=observed
    )
    planner = ScriptedPlanner(("1. Apply scoped state\n2. Run structured check",))
    implementer = StateRecordingImplementer("src/component.py")
    agent = MainAgent(
        task_analyzer=StaticTaskAnalyzer("change"),
        plan_generator=planner,
        explorer=StructuredCommandExplorer(
            explorer_for(
                failed_command_repository.root, ("src/component.py",),
                observability=observed,
            ),
            command,
        ),
        researcher=researcher_for(
            FakeMemory(),
            FakeRAG((EvidenceFragment(
                "rag", "docs://validation", "Validation must be deterministic.", 0.9
            ),)),
        ),
        implementer=implementer,
        tester=build_tester(
            failed_command_repository.root, command, runner,
            observability=observed, progress_monitor=progress, max_retries=1,
        ),
        reviewer=ScriptedReviewer(("changes_requested",)),
        policy_preflight=preflight_for(failed_command_repository.root),
        max_iterations=1,
        observability=observed,
    )

    result = agent.run(
        "Apply the scoped change and validate it",
        lambda _plan: PlanReview("approve"),
        task_id="failed-command",
    )

    state = result.task_state
    assert result.status == "max_iterations"
    assert result.iterations == 1
    assert runner.calls == [(command, 60.0), (command, 60.0)]
    assert state.commands_executed == (command,)
    tester_result = next(
        item for item in state.subagent_results if item.subagent_id == "tester"
    )
    assert tester_result.status == "failed"
    assert state.tool_calls[-1].result["attempts"] == 2
    assert any(
        "ProgressMonitor recomendó retry_with_new_strategy" in item
        for item in state.observations
    )
    assert planner.calls == [()]
    assert implementer.modes == ["apply_changes"]
    assert "Validación failed" in result.final_response
    progress_events = [
        event for event in observed.events if event.name == "progress-assessment"
    ]
    assert any(event.payload["repetition_detected"] is True for event in progress_events)
    assert observed.events[-1].payload["status"] == "max_iterations"
