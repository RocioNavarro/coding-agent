"""Operaciones estructuradas y preflight previo a escritura."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from core.planned_operations import (
    PlannedOperation,
    PlannedOperationResult,
    PolicyPreflight,
    StructuredPlannedOperationProvider,
)
from core.settings import AgentSettings
from core.task_state import SubagentResult, TaskState
from security.policy_engine import AgentToolPermissions, PolicyContext, PolicyEngine


def operation(**overrides) -> PlannedOperation:
    values = {
        "operation_id": "operation-1",
        "operation_type": "modify_file",
        "source": "explorer_result",
        "target": "src/value.py",
        "parameters": {"path": "src/value.py", "options": ["localized"]},
        "plan_version": "plan-v1",
        "metadata": {"evidence": {"kind": "structured"}},
    }
    values.update(overrides)
    return PlannedOperation(**values)


def test_planned_operation_is_strict_immutable_serializable_and_stable() -> None:
    planned = operation()

    with pytest.raises(FrozenInstanceError):
        planned.target = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        planned.parameters["path"] = "other"  # type: ignore[index]
    assert planned.fingerprint == operation().fingerprint
    assert planned.to_dict()["parameters"]["options"] == ["localized"]


@pytest.mark.parametrize("invalid", ["", "copy_file", "deploy"])
def test_rejects_invalid_operation_type(invalid: str) -> None:
    with pytest.raises(ValueError, match="operation_type"):
        operation(operation_type=invalid)


def test_fingerprint_changes_with_parameters_plan_or_source() -> None:
    baseline = operation().fingerprint

    assert operation(parameters={"path": "src/value.py", "mode": "other"}).fingerprint != baseline
    assert operation(plan_version="plan-v2").fingerprint != baseline
    assert operation(source="implementer_propose_only").fingerprint != baseline


def test_provider_uses_only_structured_files_commands_and_proposal() -> None:
    state = TaskState.create("change", task_id="provider")
    state.add_observation("Structured command: project-check")
    explorer = SubagentResult(
        "explorer", "inspect", "completed", files_relevant=("src/value.py",)
    )
    proposal = SimpleNamespace(
        changes=(SimpleNamespace(
            path="src/value.py", old_text="before", new_text="after",
            explanation="localized replacement",
        ),)
    )

    result = StructuredPlannedOperationProvider().provide(
        "Free-form plan remains opaque", state, (explorer,), proposal
    )

    assert [item.operation_type for item in result.operations] == [
        "modify_file", "run_command"
    ]
    change = result.operations[0]
    assert change.source == "implementer_propose_only"
    assert change.parameters["old_text"] == "before"
    assert result.operations[1].target == "project-check"


def test_free_text_without_structured_evidence_creates_no_operation() -> None:
    state = TaskState.create("change", task_id="opaque")

    result = StructuredPlannedOperationProvider().provide(
        "Texto libre que no se interpreta", state, (), None
    )

    assert result.operations == ()
    assert result.missing_information


def test_sensitive_marker_without_structured_intent_is_controlled() -> None:
    state = TaskState.create("change", task_id="sensitive")
    state.add_observation(
        "Sensitive operation without structured intent: external component marked ambiguity"
    )

    result = StructuredPlannedOperationProvider().provide("opaque", state, (), None)

    assert result.sensitive_unstructured == (
        "external component marked ambiguity",
    )
    assert result.confidence == 0.0


def policy_context(tmp_path, *, approval: bool = False, protected=()) -> PolicyContext:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    from core.profiles import ProjectProfile
    return PolicyContext(
        agent="main", workspace=workspace,
        permissions=AgentToolPermissions(
            approval_tools=frozenset({"write_file"}) if approval else frozenset()
        ),
        settings=AgentSettings(supervision_enabled=False),
        profile=ProjectProfile(
            additional_policies={"protected_paths": list(protected)}
        ),
    )


def test_preflight_allow_deny_require_approval_and_insufficient(tmp_path) -> None:
    intent = PlannedOperationResult((operation(),), confidence=1.0)

    allowed = PolicyPreflight(PolicyEngine(), policy_context(tmp_path)).evaluate(intent)
    denied = PolicyPreflight(
        PolicyEngine(), policy_context(tmp_path, protected=("src/*",))
    ).evaluate(intent)
    approval = PolicyPreflight(
        PolicyEngine(), policy_context(tmp_path, approval=True)
    ).evaluate(intent)
    insufficient = PolicyPreflight(PolicyEngine(), policy_context(tmp_path)).evaluate(
        PlannedOperationResult(missing_information=("structured intent",))
    )

    assert allowed.outcome == "allow"
    assert denied.outcome == "deny"
    assert approval.outcome == "require_approval"
    assert insufficient.outcome == "insufficient_structured_intent"


def test_exact_approval_is_invalidated_by_plan_or_parameter_change(tmp_path) -> None:
    preflight = PolicyPreflight(
        PolicyEngine(), policy_context(tmp_path, approval=True)
    )
    original = operation()
    approved = preflight.evaluate(
        PlannedOperationResult((original,), confidence=1.0),
        approved_fingerprints=(original.fingerprint,),
    )
    changed_plan = operation(plan_version="plan-v2")
    changed_parameters = operation(parameters={"path": "src/value.py", "mode": "other"})

    assert approved.outcome == "allow"
    assert preflight.evaluate(
        PlannedOperationResult((changed_plan,), confidence=1.0),
        approved_fingerprints=(original.fingerprint,),
    ).outcome == "require_approval"
    assert preflight.evaluate(
        PlannedOperationResult((changed_parameters,), confidence=1.0),
        approved_fingerprints=(original.fingerprint,),
    ).outcome == "require_approval"


def test_structured_prohibited_command_is_denied(tmp_path) -> None:
    command = PlannedOperation(
        "command-1", "run_command", "structured_task_state", "unsafe-command",
        {"command": "rm -rf ."}, "plan-v1",
    )

    result = PolicyPreflight(
        PolicyEngine(), policy_context(tmp_path)
    ).evaluate(PlannedOperationResult((command,), confidence=1.0))

    assert result.outcome == "deny"
    assert result.decisions[0].policy == "PolicyEngine"


def test_fake_provider_is_injectable() -> None:
    expected = PlannedOperationResult((operation(),), confidence=1.0)

    class FakeProvider:
        def provide(self, approved_plan, state, explorer_results, proposal=None):
            return expected

    assert FakeProvider().provide("plan", TaskState.create("x"), ()) is expected
