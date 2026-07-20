"""Escenario integral: preflight deniega y detiene antes de cualquier escritura."""

import json

from agents.orchestrator import MainAgent
from agents.researcher import EvidenceFragment
from core.models import PlanReview
from tests.integration.fakes import (
    FakeMemory, FakeObservability, FakeRAG, ForbiddenRunner, ScriptedPlanner,
    StaticTaskAnalyzer,
)
from tests.integration.scenario_support import (
    explorer_for, preflight_for, researcher_for,
)


def test_denied_operation_safe_stops_before_apply_changes(
    blocked_operation_repository,
) -> None:
    observed = FakeObservability()
    implementer = ForbiddenRunner("implementer apply_changes")
    tester = ForbiddenRunner("tester")
    reviewer = ForbiddenRunner("reviewer")
    original = blocked_operation_repository.read("protected/locked.txt")
    agent = MainAgent(
        task_analyzer=StaticTaskAnalyzer("change"),
        plan_generator=ScriptedPlanner(("1. Apply the explicitly scoped operation",)),
        explorer=explorer_for(
            blocked_operation_repository.root, ("protected/locked.txt",),
            observability=observed,
        ),
        researcher=researcher_for(
            FakeMemory(),
            FakeRAG((EvidenceFragment(
                "rag", "policy://protected", "The selected path is policy controlled.", 0.9
            ),)),
        ),
        implementer=implementer,
        tester=tester,
        reviewer=reviewer,
        policy_preflight=preflight_for(
            blocked_operation_repository.root, allowed_tools=frozenset()
        ),
        observability=observed,
    )

    result = agent.run(
        "Change the explicitly selected protected file",
        lambda _plan: PlanReview("approve"),
        task_id="blocked-operation",
    )

    state = result.task_state
    assert result.status == "blocked"
    assert implementer.calls == tester.calls == reviewer.calls == 0
    assert state.files_modified == ()
    assert state.commands_executed == ()
    assert state.tool_calls == ()
    assert blocked_operation_repository.read("protected/locked.txt") == original
    assert blocked_operation_repository.modified_files() == ()
    assert len(state.planned_operations) == 1
    operation = state.planned_operations[0]
    assert operation["operation_type"] == "modify_file"
    assert operation["fingerprint"]
    assert state.policy_preflight[0]["outcome"] == "deny"
    assert state.policy_preflight[0]["reason"]
    payload = json.loads(state.final_result or "{}")
    assert payload["status"] == "denied"
    assert payload["recommended_action"] == "stop"
    names = [event.name for event in observed.events]
    assert "policy-preflight" in names
    assert "tester-policy-decision" not in names
    assert not any(
        event.payload.get("phase") == "policy_execution"
        for event in observed.events
    )
