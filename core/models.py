"""Modelos de datos internos, independientes de cualquier proveedor LLM."""

from dataclasses import dataclass, field
from typing import Any, Literal


MessageRole = Literal["system", "developer", "user", "assistant", "tool"]
EvidenceStatus = Literal["sufficient", "partial", "insufficient"]
EvidenceAction = Literal["proceed", "gather_more_evidence", "request_help", "stop"]


@dataclass(frozen=True)
class ToolCall:
    """Solicitud normalizada para ejecutar una tool local."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Message:
    """Mensaje intercambiado con el modelo, incluidas llamadas y salidas de tools."""

    role: MessageRole
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None


@dataclass(frozen=True)
class LLMUsage:
    """Consumo de tokens informado por el proveedor."""

    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class LLMResponse:
    """Respuesta normalizada que el resto del agente puede consumir."""

    assistant_message: Message
    text: str
    tool_calls: list[ToolCall]
    model: str
    usage: LLMUsage
    latency_ms: float


@dataclass(frozen=True)
class InternalLoopResult:
    """Respuesta final y cantidad de iteraciones consumidas por el loop interno."""

    response: LLMResponse
    iterations: int


PlanReviewAction = Literal["approve", "reject", "modify"]


@dataclass(frozen=True)
class PlanReview:
    """Decisión humana sobre un plan generado por el modelo."""

    action: PlanReviewAction
    modification: str | None = None


@dataclass(frozen=True)
class PlanningResult:
    """Resultado de la fase de planificación previa a las tools."""

    approved: bool
    plan: str | None = None


@dataclass(frozen=True)
class EvidenceAssessment:
    """Resultado explícito de evaluar evidencia antes de modificar artefactos."""

    status: EvidenceStatus
    supporting_sources: tuple[str, ...]
    missing_information: tuple[str, ...]
    risks: tuple[str, ...]
    recommended_action: EvidenceAction
    confidence: float

    def __post_init__(self) -> None:
        allowed_statuses = {"sufficient", "partial", "insufficient"}
        allowed_actions = {
            "proceed", "gather_more_evidence", "request_help", "stop"
        }
        if self.status not in allowed_statuses:
            raise ValueError("status de evidencia inválido.")
        if self.recommended_action not in allowed_actions:
            raise ValueError("recommended_action de evidencia inválida.")
        for field_name in ("supporting_sources", "missing_information", "risks"):
            values = getattr(self, field_name)
            if not isinstance(values, (list, tuple)) or not all(
                isinstance(item, str) and item.strip() for item in values
            ):
                raise ValueError(f"{field_name} debe contener textos no vacíos.")
            object.__setattr__(
                self, field_name, tuple(item.strip() for item in values)
            )
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not 0 <= self.confidence <= 1
        ):
            raise ValueError("confidence debe estar entre 0 y 1.")
        object.__setattr__(self, "confidence", float(self.confidence))
        expected_action = {
            "sufficient": "proceed",
            "partial": "gather_more_evidence",
        }.get(self.status)
        if expected_action is not None and self.recommended_action != expected_action:
            raise ValueError(
                f"El status '{self.status}' requiere la acción '{expected_action}'."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "supporting_sources": list(self.supporting_sources),
            "missing_information": list(self.missing_information),
            "risks": list(self.risks),
            "recommended_action": self.recommended_action,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceAssessment":
        try:
            return cls(
                status=data["status"],
                supporting_sources=tuple(data["supporting_sources"]),
                missing_information=tuple(data["missing_information"]),
                risks=tuple(data["risks"]),
                recommended_action=data["recommended_action"],
                confidence=data["confidence"],
            )
        except (KeyError, TypeError) as error:
            raise ValueError("EvidenceAssessment serializado inválido.") from error
