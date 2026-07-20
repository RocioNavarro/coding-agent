"""Estado compartido, tipado y serializable de una tarea del coding agent."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Mapping, Sequence
from uuid import uuid4

from core.models import EvidenceAssessment, ToolCall


SourceOrigin = Literal["repository", "project_memory", "rag", "web", "inference"]
SOURCE_ORIGINS: frozenset[str] = frozenset(
    {"repository", "project_memory", "rag", "web", "inference"}
)


def _required_text(value: object, field_name: str) -> str:
    """Valida y normaliza un texto requerido."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} no puede estar vacío.")
    return value.strip()


def _optional_text(value: object, field_name: str) -> str | None:
    """Valida un texto opcional sin convertir valores de otros tipos."""
    if value is None:
        return None
    return _required_text(value, field_name)


def _require_mapping(value: object, field_name: str) -> dict[str, Any]:
    """Devuelve una copia defensiva de un mapping con claves de texto."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} debe ser un objeto.")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"Las claves de {field_name} deben ser strings.")
    return deepcopy(dict(value))


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} debe ser booleano.")
    return value


@dataclass(frozen=True)
class SubagentResult:
    """Resultado normalizado de una tarea delegada a un subagente."""

    subagent_id: str
    task: str
    status: str
    result: str | None = None
    error: str | None = None
    summary: str | None = None
    findings: tuple[str, ...] = ()
    recommendations: tuple[str, ...] = ()
    requested_tool_calls: tuple[ToolCall, ...] = ()
    sources: tuple[SourceReference, ...] = ()
    files_relevant: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    confidence: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "subagent_id", _required_text(self.subagent_id, "subagent_id"))
        object.__setattr__(self, "task", _required_text(self.task, "task"))
        object.__setattr__(self, "status", _required_text(self.status, "status"))
        object.__setattr__(self, "result", _optional_text(self.result, "result"))
        object.__setattr__(self, "error", _optional_text(self.error, "error"))
        object.__setattr__(self, "summary", _optional_text(self.summary, "summary"))
        for field_name in (
            "findings", "recommendations", "files_relevant", "blockers"
        ):
            value = getattr(self, field_name)
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"{field_name} debe ser una lista o tupla.")
            object.__setattr__(
                self,
                field_name,
                tuple(_required_text(item, field_name) for item in value),
            )
        if not isinstance(self.requested_tool_calls, (list, tuple)) or not all(
            isinstance(item, ToolCall) for item in self.requested_tool_calls
        ):
            raise ValueError("requested_tool_calls debe contener instancias de ToolCall.")
        object.__setattr__(
            self, "requested_tool_calls", tuple(self.requested_tool_calls)
        )
        if not isinstance(self.sources, (list, tuple)) or not all(
            isinstance(item, SourceReference) for item in self.sources
        ):
            raise ValueError("sources debe contener instancias de SourceReference.")
        object.__setattr__(self, "sources", tuple(self.sources))
        if self.confidence is not None:
            if (
                isinstance(self.confidence, bool)
                or not isinstance(self.confidence, (int, float))
                or not 0 <= self.confidence <= 1
            ):
                raise ValueError("confidence debe estar entre 0 y 1.")
            object.__setattr__(self, "confidence", float(self.confidence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "subagent_id": self.subagent_id,
            "task": self.task,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "summary": self.summary,
            "findings": list(self.findings),
            "recommendations": list(self.recommendations),
            "requested_tool_calls": [
                {"id": call.id, "name": call.name, "arguments": deepcopy(call.arguments)}
                for call in self.requested_tool_calls
            ],
            "sources": [source.to_dict() for source in self.sources],
            "files_relevant": list(self.files_relevant),
            "blockers": list(self.blockers),
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SubagentResult:
        values = _require_mapping(data, "subagent_result")
        try:
            return cls(
                subagent_id=values["subagent_id"],
                task=values["task"],
                status=values["status"],
                result=values.get("result"),
                error=values.get("error"),
                summary=values.get("summary"),
                findings=tuple(values.get("findings", [])),
                recommendations=tuple(values.get("recommendations", [])),
                requested_tool_calls=tuple(
                    ToolCall(
                        id=call["id"],
                        name=call["name"],
                        arguments=_require_mapping(call["arguments"], "arguments"),
                    )
                    for call in values.get("requested_tool_calls", [])
                ),
                sources=tuple(
                    SourceReference.from_dict(source)
                    for source in values.get("sources", [])
                ),
                files_relevant=tuple(values.get("files_relevant", [])),
                blockers=tuple(values.get("blockers", [])),
                confidence=values.get("confidence"),
            )
        except KeyError as error:
            raise ValueError(f"Falta el campo requerido: {error.args[0]}.") from error


@dataclass(frozen=True)
class SourceReference:
    """Fuente consultada y procedencia de la información obtenida."""

    origin: SourceOrigin
    reference: str
    summary: str | None = None

    def __post_init__(self) -> None:
        if self.origin not in SOURCE_ORIGINS:
            allowed = ", ".join(sorted(SOURCE_ORIGINS))
            raise ValueError(f"Origen de fuente inválido. Valores permitidos: {allowed}.")
        object.__setattr__(self, "reference", _required_text(self.reference, "reference"))
        object.__setattr__(self, "summary", _optional_text(self.summary, "summary"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin": self.origin,
            "reference": self.reference,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SourceReference:
        values = _require_mapping(data, "source")
        try:
            return cls(
                origin=values["origin"],
                reference=values["reference"],
                summary=values.get("summary"),
            )
        except KeyError as error:
            raise ValueError(f"Falta el campo requerido: {error.args[0]}.") from error


@dataclass(frozen=True)
class ToolExecutionRecord:
    """Registro auditable de una tool call y su resultado controlado."""

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    success: bool
    result: Any = None
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_call_id", _required_text(self.tool_call_id, "tool_call_id"))
        object.__setattr__(self, "tool_name", _required_text(self.tool_name, "tool_name"))
        object.__setattr__(self, "arguments", _require_mapping(self.arguments, "arguments"))
        object.__setattr__(self, "success", _require_bool(self.success, "success"))
        object.__setattr__(self, "result", deepcopy(self.result))
        object.__setattr__(self, "error", _optional_text(self.error, "error"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "arguments": deepcopy(self.arguments),
            "success": self.success,
            "result": deepcopy(self.result),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ToolExecutionRecord:
        values = _require_mapping(data, "tool_call")
        try:
            return cls(
                tool_call_id=values["tool_call_id"],
                tool_name=values["tool_name"],
                arguments=values["arguments"],
                success=values["success"],
                result=values.get("result"),
                error=values.get("error"),
            )
        except KeyError as error:
            raise ValueError(f"Falta el campo requerido: {error.args[0]}.") from error


@dataclass(frozen=True)
class ErrorRecord:
    """Error asociado a una fase o componente de la tarea."""

    message: str
    phase: str
    component: str | None = None
    recoverable: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "message", _required_text(self.message, "message"))
        object.__setattr__(self, "phase", _required_text(self.phase, "phase"))
        object.__setattr__(self, "component", _optional_text(self.component, "component"))
        object.__setattr__(self, "recoverable", _require_bool(self.recoverable, "recoverable"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "phase": self.phase,
            "component": self.component,
            "recoverable": self.recoverable,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ErrorRecord:
        values = _require_mapping(data, "error")
        try:
            return cls(
                message=values["message"],
                phase=values["phase"],
                component=values.get("component"),
                recoverable=values.get("recoverable", True),
            )
        except KeyError as error:
            raise ValueError(f"Falta el campo requerido: {error.args[0]}.") from error


@dataclass
class TaskState:
    """Estado compartido de una tarea, actualizado mediante operaciones explícitas."""

    task_id: str
    original_request: str
    current_status: str = "pending"
    current_phase: str = "intake"
    proposed_plan: str | None = None
    approved_plan: str | None = None
    final_result: str | None = None
    _subagent_results: list[SubagentResult] = field(default_factory=list, repr=False)
    _sources: list[SourceReference] = field(default_factory=list, repr=False)
    _repository_findings: list[str] = field(default_factory=list, repr=False)
    _files_read: list[str] = field(default_factory=list, repr=False)
    _files_modified: list[str] = field(default_factory=list, repr=False)
    _commands_executed: list[str] = field(default_factory=list, repr=False)
    _tool_calls: list[ToolExecutionRecord] = field(default_factory=list, repr=False)
    _errors: list[ErrorRecord] = field(default_factory=list, repr=False)
    _warnings: list[str] = field(default_factory=list, repr=False)
    _observations: list[str] = field(default_factory=list, repr=False)
    _evidence_assessment: EvidenceAssessment | None = field(default=None, repr=False)
    _evidence_assessment_plan: str | None = field(default=None, repr=False)
    _planned_operations: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _policy_preflight: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _policy_approvals: list[str] = field(default_factory=list, repr=False)

    DEFAULT_STATUS: ClassVar[str] = "pending"
    DEFAULT_PHASE: ClassVar[str] = "intake"

    def __post_init__(self) -> None:
        self.task_id = _required_text(self.task_id, "task_id")
        self.original_request = _required_text(self.original_request, "original_request")
        self.current_status = _required_text(self.current_status, "current_status")
        self.current_phase = _required_text(self.current_phase, "current_phase")
        self.proposed_plan = _optional_text(self.proposed_plan, "proposed_plan")
        self.approved_plan = _optional_text(self.approved_plan, "approved_plan")
        self.final_result = _optional_text(self.final_result, "final_result")

    @classmethod
    def create(cls, original_request: str, *, task_id: str | None = None) -> TaskState:
        """Crea una tarea con identificador provisto o generado localmente."""
        return cls(task_id=task_id or str(uuid4()), original_request=original_request)

    @property
    def subagent_results(self) -> tuple[SubagentResult, ...]:
        return tuple(self._subagent_results)

    @property
    def sources(self) -> tuple[SourceReference, ...]:
        return tuple(self._sources)

    @property
    def repository_findings(self) -> tuple[str, ...]:
        return tuple(self._repository_findings)

    @property
    def files_read(self) -> tuple[str, ...]:
        return tuple(self._files_read)

    @property
    def files_modified(self) -> tuple[str, ...]:
        return tuple(self._files_modified)

    @property
    def commands_executed(self) -> tuple[str, ...]:
        return tuple(self._commands_executed)

    @property
    def tool_calls(self) -> tuple[ToolExecutionRecord, ...]:
        return tuple(self._tool_calls)

    @property
    def errors(self) -> tuple[ErrorRecord, ...]:
        return tuple(self._errors)

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(self._warnings)

    @property
    def observations(self) -> tuple[str, ...]:
        return tuple(self._observations)

    @property
    def evidence_assessment(self) -> EvidenceAssessment | None:
        return self._evidence_assessment

    @property
    def planned_operations(self) -> tuple[dict[str, Any], ...]:
        return tuple(deepcopy(self._planned_operations))

    @property
    def policy_preflight(self) -> tuple[dict[str, Any], ...]:
        return tuple(deepcopy(self._policy_preflight))

    @property
    def policy_approvals(self) -> tuple[str, ...]:
        return tuple(self._policy_approvals)

    @property
    def has_current_sufficient_evidence(self) -> bool:
        return (
            self._evidence_assessment is not None
            and self._evidence_assessment.status == "sufficient"
            and self._evidence_assessment_plan == self.approved_plan
        )

    def set_status(self, status: str) -> None:
        self.current_status = _required_text(status, "current_status")

    def set_phase(self, phase: str) -> None:
        self.current_phase = _required_text(phase, "current_phase")

    def propose_plan(self, plan: str) -> None:
        self.proposed_plan = _required_text(plan, "proposed_plan")
        self._invalidate_evidence_assessment()

    def approve_plan(self, plan: str | None = None) -> None:
        """Aprueba el plan indicado o, si se omite, el último plan propuesto."""
        selected_plan = plan if plan is not None else self.proposed_plan
        if selected_plan is None:
            raise ValueError("No hay un plan propuesto para aprobar.")
        self.approved_plan = _required_text(selected_plan, "approved_plan")
        self._invalidate_evidence_assessment()

    def record_evidence_assessment(self, assessment: EvidenceAssessment) -> None:
        """Registra la evaluación y el plan exacto para el cual sigue vigente."""
        if not isinstance(assessment, EvidenceAssessment):
            raise TypeError("assessment debe ser EvidenceAssessment.")
        if self.approved_plan is None:
            raise ValueError("No se puede evaluar evidencia sin un plan aprobado.")
        self._evidence_assessment = assessment
        self._evidence_assessment_plan = self.approved_plan

    def record_planned_operations(self, operations: Sequence[Mapping[str, Any]]) -> None:
        self._planned_operations = [
            _require_mapping(item, "planned_operation") for item in operations
        ]

    def record_policy_preflight(self, decisions: Sequence[Mapping[str, Any]]) -> None:
        self._policy_preflight = [
            _require_mapping(item, "policy_preflight") for item in decisions
        ]

    def record_policy_approval(self, fingerprint: str) -> None:
        self._append_unique(self._policy_approvals, fingerprint, "fingerprint")

    def _invalidate_evidence_assessment(self) -> None:
        self._evidence_assessment = None
        self._evidence_assessment_plan = None

    def add_subagent_result(self, result: SubagentResult) -> None:
        if not isinstance(result, SubagentResult):
            raise TypeError("result debe ser una instancia de SubagentResult.")
        self._subagent_results.append(result)

    def add_source(self, source: SourceReference) -> None:
        if not isinstance(source, SourceReference):
            raise TypeError("source debe ser una instancia de SourceReference.")
        self._sources.append(source)

    def add_repository_finding(self, finding: str) -> None:
        self._repository_findings.append(_required_text(finding, "finding"))

    def record_file_read(self, path: str) -> None:
        self._append_unique(self._files_read, path, "path")

    def record_file_modified(self, path: str) -> None:
        self._append_unique(self._files_modified, path, "path")

    def record_command(self, command: str) -> None:
        self._commands_executed.append(_required_text(command, "command"))

    def record_tool_call(self, record: ToolExecutionRecord) -> None:
        if not isinstance(record, ToolExecutionRecord):
            raise TypeError("record debe ser una instancia de ToolExecutionRecord.")
        self._tool_calls.append(record)

    def record_error(self, error: ErrorRecord) -> None:
        if not isinstance(error, ErrorRecord):
            raise TypeError("error debe ser una instancia de ErrorRecord.")
        self._errors.append(error)

    def add_warning(self, warning: str) -> None:
        self._warnings.append(_required_text(warning, "warning"))

    def add_observation(self, observation: str) -> None:
        self._observations.append(_required_text(observation, "observation"))

    def set_final_result(self, result: str) -> None:
        self.final_result = _required_text(result, "final_result")

    @staticmethod
    def _append_unique(items: list[str], value: str, field_name: str) -> None:
        normalized = _required_text(value, field_name)
        if normalized not in items:
            items.append(normalized)

    def to_dict(self) -> dict[str, Any]:
        """Genera una representación desacoplada y apta para JSON."""
        return {
            "task_id": self.task_id,
            "original_request": self.original_request,
            "current_status": self.current_status,
            "current_phase": self.current_phase,
            "proposed_plan": self.proposed_plan,
            "approved_plan": self.approved_plan,
            "subagent_results": [item.to_dict() for item in self._subagent_results],
            "sources": [item.to_dict() for item in self._sources],
            "repository_findings": list(self._repository_findings),
            "files_read": list(self._files_read),
            "files_modified": list(self._files_modified),
            "commands_executed": list(self._commands_executed),
            "tool_calls": [item.to_dict() for item in self._tool_calls],
            "errors": [item.to_dict() for item in self._errors],
            "warnings": list(self._warnings),
            "observations": list(self._observations),
            "evidence_assessment": (
                self._evidence_assessment.to_dict()
                if self._evidence_assessment is not None else None
            ),
            "evidence_assessment_plan": self._evidence_assessment_plan,
            "planned_operations": deepcopy(self._planned_operations),
            "policy_preflight": deepcopy(self._policy_preflight),
            "policy_approvals": list(self._policy_approvals),
            "final_result": self.final_result,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serializa el estado completo a JSON UTF-8 legible."""
        try:
            return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
        except (TypeError, ValueError) as error:
            raise ValueError(f"El estado contiene datos no serializables: {error}") from error

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TaskState:
        """Reconstruye un estado y valida sus registros anidados."""
        values = _require_mapping(data, "task_state")
        try:
            state = cls(
                task_id=values["task_id"],
                original_request=values["original_request"],
                current_status=values["current_status"],
                current_phase=values["current_phase"],
                proposed_plan=values.get("proposed_plan"),
                approved_plan=values.get("approved_plan"),
                final_result=values.get("final_result"),
            )
        except KeyError as error:
            raise ValueError(f"Falta el campo requerido: {error.args[0]}.") from error

        state._subagent_results.extend(
            SubagentResult.from_dict(item)
            for item in cls._required_list(values, "subagent_results")
        )
        state._sources.extend(
            SourceReference.from_dict(item)
            for item in cls._required_list(values, "sources")
        )
        state._repository_findings.extend(
            cls._required_text_list(values, "repository_findings")
        )
        state._files_read.extend(cls._required_text_list(values, "files_read"))
        state._files_modified.extend(
            cls._required_text_list(values, "files_modified")
        )
        state._commands_executed.extend(
            cls._required_text_list(values, "commands_executed")
        )
        state._tool_calls.extend(
            ToolExecutionRecord.from_dict(item)
            for item in cls._required_list(values, "tool_calls")
        )
        state._errors.extend(
            ErrorRecord.from_dict(item)
            for item in cls._required_list(values, "errors")
        )
        state._warnings.extend(cls._required_text_list(values, "warnings"))
        state._observations.extend(cls._required_text_list(values, "observations"))
        state._planned_operations.extend(
            _require_mapping(item, "planned_operation")
            for item in cls._required_list(values, "planned_operations")
        )
        state._policy_preflight.extend(
            _require_mapping(item, "policy_preflight")
            for item in cls._required_list(values, "policy_preflight")
        )
        state._policy_approvals.extend(
            cls._required_text_list(values, "policy_approvals")
        )
        raw_assessment = values.get("evidence_assessment")
        if raw_assessment is not None:
            if not isinstance(raw_assessment, dict):
                raise ValueError("evidence_assessment debe ser un objeto.")
            state._evidence_assessment = EvidenceAssessment.from_dict(raw_assessment)
            state._evidence_assessment_plan = _optional_text(
                values.get("evidence_assessment_plan"), "evidence_assessment_plan"
            )
        return state

    @classmethod
    def from_json(cls, payload: str) -> TaskState:
        """Deserializa JSON y rechaza una raíz que no sea un objeto."""
        try:
            data = json.loads(payload)
        except (json.JSONDecodeError, TypeError) as error:
            raise ValueError(f"JSON de estado inválido: {error}") from error
        if not isinstance(data, dict):
            raise ValueError("El JSON de estado debe contener un objeto en la raíz.")
        return cls.from_dict(data)

    @staticmethod
    def _required_list(values: Mapping[str, Any], field_name: str) -> list[Any]:
        value = values.get(field_name, [])
        if not isinstance(value, list):
            raise ValueError(f"{field_name} debe ser una lista.")
        return value

    @classmethod
    def _required_text_list(
        cls, values: Mapping[str, Any], field_name: str
    ) -> list[str]:
        return [
            _required_text(item, field_name)
            for item in cls._required_list(values, field_name)
        ]
