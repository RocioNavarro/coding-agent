"""Modelos de datos internos, independientes de cualquier proveedor LLM."""

from dataclasses import dataclass
from typing import Any, Literal


MessageRole = Literal["system", "developer", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    """Mensaje textual intercambiado con el modelo."""

    role: MessageRole
    content: str


@dataclass(frozen=True)
class ToolCall:
    """Solicitud normalizada para ejecutar una tool local."""

    id: str
    name: str
    arguments: dict[str, Any]


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
