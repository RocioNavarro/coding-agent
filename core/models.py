"""Modelos de datos internos, independientes de cualquier proveedor LLM."""

from dataclasses import dataclass, field
from typing import Any, Literal


MessageRole = Literal["system", "developer", "user", "assistant", "tool"]


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
