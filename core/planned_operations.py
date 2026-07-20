"""Intención estructurada y preflight de políticas previo a modificaciones."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping, Protocol, Sequence

from core.task_state import TaskState
from security.policy_engine import PolicyContext, PolicyEngine


OperationType = Literal[
    "read_file", "write_file", "modify_file", "delete_file", "run_command",
    "install_dependency", "git_operation", "unknown_sensitive_operation",
]
PreflightOutcome = Literal[
    "allow", "deny", "require_approval", "insufficient_structured_intent"
]
_OPERATION_TYPES = frozenset(
    {"read_file", "write_file", "modify_file", "delete_file", "run_command",
     "install_dependency", "git_operation", "unknown_sensitive_operation"}
)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class PlannedOperation:
    operation_id: str
    operation_type: OperationType
    source: str
    target: str
    parameters: Mapping[str, Any]
    plan_version: str
    requires_structured_intent: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("operation_id", "source", "target", "plan_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} no puede estar vacío.")
            object.__setattr__(self, name, value.strip())
        if self.operation_type not in _OPERATION_TYPES:
            raise ValueError("operation_type inválido.")
        if not isinstance(self.requires_structured_intent, bool):
            raise ValueError("requires_structured_intent debe ser booleano.")
        for name in ("parameters", "metadata"):
            value = getattr(self, name)
            if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
                raise ValueError(f"{name} debe ser un mapping con claves de texto.")
            object.__setattr__(self, name, _freeze(value))

    @property
    def fingerprint(self) -> str:
        payload = {
            "operation_type": self.operation_type,
            "target": self.target,
            "parameters": _plain(self.parameters),
            "plan_version": self.plan_version,
            "source": self.source,
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "source": self.source,
            "target": self.target,
            "parameters": _plain(self.parameters),
            "plan_version": self.plan_version,
            "requires_structured_intent": self.requires_structured_intent,
            "metadata": _plain(self.metadata),
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class PlannedOperationResult:
    operations: tuple[PlannedOperation, ...] = ()
    missing_information: tuple[str, ...] = ()
    sensitive_unstructured: tuple[str, ...] = ()
    confidence: float = 0.0
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not all(isinstance(item, PlannedOperation) for item in self.operations):
            raise ValueError("operations debe contener PlannedOperation.")
        for name in ("missing_information", "sensitive_unstructured", "provenance"):
            values = getattr(self, name)
            if not all(isinstance(item, str) and item.strip() for item in values):
                raise ValueError(f"{name} debe contener textos no vacíos.")
            object.__setattr__(self, name, tuple(item.strip() for item in values))
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence debe estar entre 0 y 1.")
        object.__setattr__(self, "operations", tuple(self.operations))
        object.__setattr__(self, "confidence", float(self.confidence))


class PlannedOperationProvider(Protocol):
    def provide(
        self,
        approved_plan: str,
        state: TaskState,
        explorer_results: Sequence[Any],
        proposal: Any | None = None,
    ) -> PlannedOperationResult:
        ...


class StructuredPlannedOperationProvider:
    """Extrae sólo operaciones respaldadas por campos ya estructurados."""

    def provide(self, approved_plan, state, explorer_results, proposal=None) -> PlannedOperationResult:
        version = hashlib.sha256(approved_plan.encode("utf-8")).hexdigest()[:16]
        operations: dict[tuple[str, str], PlannedOperation] = {}
        provenance: list[str] = []
        for result in explorer_results:
            for path in getattr(result, "files_relevant", ()):
                key = ("modify_file", path)
                operations[key] = PlannedOperation(
                    f"target:{len(operations) + 1}", "modify_file", "explorer_result",
                    path, {"path": path}, version, metadata={"evidence": "files_relevant"},
                )
                provenance.append(f"explorer_result:{path}")
        if proposal is not None:
            for change in getattr(proposal, "changes", ()):
                path = change.path
                key = ("modify_file", path)
                operations[key] = PlannedOperation(
                    f"proposal:{len(operations) + 1}", "modify_file",
                    "implementer_propose_only", path,
                    {"path": path, "old_text": change.old_text,
                     "new_text": change.new_text, "explanation": change.explanation},
                    version, metadata={"structured_proposal": True},
                )
                provenance.append(f"implementer_propose_only:{path}")
        for observation in state.observations:
            prefix = "Structured command: "
            if observation.startswith(prefix):
                command = observation.removeprefix(prefix).strip()
                key = ("run_command", command)
                operations[key] = PlannedOperation(
                    f"command:{len(operations) + 1}", "run_command",
                    "structured_task_state", command, {"command": command}, version,
                )
                provenance.append("task_state:structured_command")
        sensitive = tuple(
            item.removeprefix("Sensitive operation without structured intent: ").strip()
            for item in state.observations
            if item.startswith("Sensitive operation without structured intent: ")
        )
        missing = (
            ("Falta intención estructurada para una operación sensible.",)
            if sensitive else
            ("No existen operaciones estructuradas para la tarea de modificación.",)
            if not operations else ()
        )
        return PlannedOperationResult(
            tuple(operations.values()), missing, sensitive,
            1.0 if operations and not sensitive else 0.0,
            tuple(dict.fromkeys(provenance)),
        )


@dataclass(frozen=True)
class PreflightDecision:
    operation_id: str
    outcome: PreflightOutcome
    reason: str
    policy: str
    source: str
    approval_required: bool
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class PolicyPreflightResult:
    outcome: PreflightOutcome
    decisions: tuple[PreflightDecision, ...]
    missing_information: tuple[str, ...] = ()
    recommended_action: str = "proceed"


class PolicyPreflight:
    def __init__(self, engine: PolicyEngine, context: PolicyContext) -> None:
        self.engine = engine
        self.context = context

    def evaluate(
        self,
        intent: PlannedOperationResult,
        *,
        approved_fingerprints: Sequence[str] = (),
    ) -> PolicyPreflightResult:
        if intent.sensitive_unstructured or not intent.operations:
            return PolicyPreflightResult(
                "insufficient_structured_intent", (), intent.missing_information,
                "request_structured_intent",
            )
        approvals = frozenset(approved_fingerprints)
        decisions: list[PreflightDecision] = []
        for operation in intent.operations:
            tool, parameters = self._policy_input(operation)
            decision = self.engine.evaluate(tool, parameters, self.context)
            outcome: PreflightOutcome = decision.outcome
            if outcome == "require_approval" and operation.fingerprint in approvals:
                outcome = "allow"
                reason = "Aprobación exacta vigente para la operación."
            else:
                reason = decision.reason
            decisions.append(
                PreflightDecision(
                    operation.operation_id, outcome, reason, "PolicyEngine",
                    operation.source, outcome == "require_approval", operation.fingerprint,
                )
            )
        overall: PreflightOutcome = (
            "deny" if any(item.outcome == "deny" for item in decisions) else
            "require_approval" if any(item.outcome == "require_approval" for item in decisions) else
            "allow"
        )
        action = {"allow": "proceed", "deny": "stop", "require_approval": "request_approval"}[overall]
        return PolicyPreflightResult(overall, tuple(decisions), (), action)

    @staticmethod
    def _policy_input(operation: PlannedOperation) -> tuple[str, dict[str, Any]]:
        if operation.operation_type == "run_command":
            return "run_command", {"command": operation.parameters.get("command", operation.target)}
        if operation.operation_type == "read_file":
            return "read_file", {"path": operation.target}
        if operation.operation_type in {"write_file", "modify_file", "delete_file"}:
            return "write_file", {"path": operation.target, "content": operation.parameters.get("new_text", "")}
        return operation.operation_type, dict(operation.parameters)
