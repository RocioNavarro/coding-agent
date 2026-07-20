"""Observabilidad desacoplada, segura y opcional para el coding agent."""

from __future__ import annotations

import logging
import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Protocol, runtime_checkable


ObservabilityEventType = Literal[
    "task", "agent", "prompt", "model", "llm_call", "tool", "rag", "web",
    "iteration", "error", "result",
]
_EVENT_TYPES = frozenset(
    {"task", "agent", "prompt", "model", "llm_call", "tool", "rag", "web",
     "iteration", "error", "result"}
)
_SECRET_KEY = re.compile(
    r"(?:api[_-]?key|password|passwd|secret|token|authorization|credential|private[_-]?key)",
    re.IGNORECASE,
)
_SECRET_VALUE = re.compile(
    r"(?i)\b(api[_-]?key|password|passwd|secret|token|authorization|credential)"
    r"(\s*[=:]\s*)([^\s,;]+)"
)
_CREDENTIAL_VALUE = re.compile(
    r"(?i)(?:\bBearer\s+)[A-Za-z0-9._~+/=-]+|"
    r"\b(?:sk-[A-Za-z0-9_-]{4,}|gh[pousr]_[A-Za-z0-9_]{4,})\b"
)
_REDACTED = "***"
_CURRENT_CONTEXT: ContextVar[tuple[str | None, str | None, str | None]] = ContextVar(
    "coding_agent_observation_context", default=(None, None, None)
)


@dataclass(frozen=True)
class ObservabilityEvent:
    """Evento normalizado que no depende del proveedor de telemetría."""

    event_type: ObservabilityEventType
    name: str
    event_id: str | None = None
    parent_event_id: str | None = None
    task_id: str | None = None
    agent: str | None = None
    model: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float | None = None
    estimated_cost: float | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if self.event_type not in _EVENT_TYPES:
            raise ValueError("event_type de observabilidad inválido.")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name no puede estar vacío.")
        object.__setattr__(self, "name", self.name.strip())
        for field_name in ("event_id", "parent_event_id", "task_id", "agent", "model", "timestamp"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field_name} debe ser texto no vacío.")
            if isinstance(value, str):
                object.__setattr__(self, field_name, value.strip())
        if not isinstance(self.payload, Mapping):
            raise ValueError("payload debe ser un mapping.")
        object.__setattr__(self, "payload", dict(self.payload))
        for field_name in ("input_tokens", "output_tokens", "total_tokens"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{field_name} debe ser un entero no negativo.")
        for field_name in ("latency_ms", "estimated_cost"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value < 0
            ):
                raise ValueError(f"{field_name} debe ser un número no negativo.")
            if value is not None:
                object.__setattr__(self, field_name, float(value))


@runtime_checkable
class ObservabilityClient(Protocol):
    """Contrato mínimo para proveedores de observabilidad intercambiables."""

    def record(self, event: ObservabilityEvent) -> None:
        """Registra un evento sin afectar el flujo principal."""

    def flush(self) -> None:
        """Intenta enviar eventos pendientes sin afectar el flujo principal."""


class NoOpObservabilityClient:
    """Implementación segura cuando la observabilidad no está disponible."""

    def record(self, event: ObservabilityEvent) -> None:
        return None

    def flush(self) -> None:
        return None


def sanitize_observability_data(value: Any) -> Any:
    """Copia recursivamente datos y oculta claves y valores sensibles."""
    if isinstance(value, Mapping):
        return {
            str(key): (
                _REDACTED
                if _SECRET_KEY.search(str(key))
                else sanitize_observability_data(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_observability_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_observability_data(item) for item in value)
    if isinstance(value, str):
        redacted = _SECRET_VALUE.sub(r"\1\2***", value)
        return _CREDENTIAL_VALUE.sub(_REDACTED, redacted)
    return value


class LangfuseObservabilityClient:
    """Adaptador tolerante a fallos para Langfuse Python SDK v4."""

    def __init__(self, sdk_client: Any, *, logger: logging.Logger | None = None) -> None:
        self._client = sdk_client
        self._logger = logger or logging.getLogger(__name__)
        self._contexts: dict[str, tuple[str, str]] = {}
        self._task_roots: dict[str, tuple[str, str]] = {}

    def record(self, event: ObservabilityEvent) -> None:
        try:
            safe_payload = sanitize_observability_data(event.payload)
            metadata = sanitize_observability_data(
                {
                    "event_type": event.event_type,
                    "event_id": event.event_id,
                    "parent_event_id": event.parent_event_id,
                    "task_id": event.task_id,
                    "agent": event.agent,
                    "latency_ms": event.latency_ms,
                    "estimated_cost": event.estimated_cost,
                }
            )
            arguments: dict[str, Any] = {
                "name": event.name,
                "as_type": self._observation_type(event.event_type),
                "input": safe_payload,
                "metadata": metadata,
            }
            if event.event_type == "llm_call":
                arguments["model"] = event.model
                usage = {
                    "input_tokens": event.input_tokens,
                    "output_tokens": event.output_tokens,
                    "total_tokens": event.total_tokens,
                }
                arguments["usage_details"] = {
                    key: value for key, value in usage.items() if value is not None
                }
                if event.estimated_cost is not None:
                    arguments["cost_details"] = {
                        "estimated_cost": event.estimated_cost
                    }
            parent_context = self._contexts.get(event.parent_event_id or "")
            if parent_context is None and event.task_id:
                parent_context = self._task_roots.get(event.task_id)
            if parent_context is not None:
                arguments["trace_context"] = {
                    "trace_id": parent_context[0],
                    "parent_span_id": parent_context[1],
                }
            observation = self._client.start_observation(**arguments)
            if event.event_id:
                trace_id = getattr(observation, "trace_id", None)
                observation_id = getattr(observation, "observation_id", None)
                if trace_id and observation_id:
                    self._contexts[event.event_id] = (trace_id, observation_id)
                    if event.event_type == "task" and event.task_id:
                        self._task_roots[event.task_id] = (trace_id, observation_id)
            observation.end()
        except Exception as error:
            self._logger.warning("Langfuse no pudo registrar el evento: %s", error)

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception as error:
            self._logger.warning("Langfuse no pudo enviar eventos pendientes: %s", error)

    @staticmethod
    def _observation_type(event_type: ObservabilityEventType) -> str:
        return {
            "agent": "agent",
            "llm_call": "generation",
            "tool": "tool",
            "rag": "retriever",
        }.get(event_type, "span")


LangfuseFactory = Callable[..., Any]


def emit_observation(client: ObservabilityClient, event: ObservabilityEvent) -> None:
    """Sanitiza y emite sin permitir que telemetría interrumpa la tarea."""
    current_task, current_parent, current_agent = _CURRENT_CONTEXT.get()
    safe = ObservabilityEvent(
        event_type=event.event_type,
        name=event.name,
        event_id=event.event_id,
        parent_event_id=event.parent_event_id or current_parent,
        task_id=event.task_id or current_task,
        agent=event.agent or current_agent,
        model=event.model,
        payload=sanitize_observability_data(event.payload),
        input_tokens=event.input_tokens,
        output_tokens=event.output_tokens,
        total_tokens=event.total_tokens,
        latency_ms=event.latency_ms,
        estimated_cost=event.estimated_cost,
        timestamp=event.timestamp,
    )
    try:
        client.record(safe)
    except Exception:
        logging.getLogger(__name__).warning(
            "El cliente de observabilidad rechazó un evento.", exc_info=True
        )


@contextmanager
def observation_context(
    *, task_id: str | None, parent_event_id: str | None, agent: str | None = None
):
    """Propaga relación lógica a eventos internos sin acoplar componentes."""
    token = _CURRENT_CONTEXT.set((task_id, parent_event_id, agent))
    try:
        yield
    finally:
        _CURRENT_CONTEXT.reset(token)


def build_observability_client(
    *,
    environ: Mapping[str, str] | None = None,
    langfuse_factory: LangfuseFactory | None = None,
    logger: logging.Logger | None = None,
) -> ObservabilityClient:
    """Selecciona Langfuse o No-Op sin exigir credenciales ni SDK al arrancar."""
    environment = environ if environ is not None else os.environ
    selected_logger = logger or logging.getLogger(__name__)
    raw_enabled = environment.get("CODING_AGENT_OBSERVABILITY_ENABLED", "false")
    if raw_enabled.strip().casefold() not in {"true", "1", "yes", "on"}:
        return NoOpObservabilityClient()
    public_key = environment.get("LANGFUSE_PUBLIC_KEY")
    secret_key = environment.get("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        selected_logger.warning(
            "Observabilidad deshabilitada: faltan credenciales de Langfuse."
        )
        return NoOpObservabilityClient()
    try:
        if langfuse_factory is None:
            from langfuse import Langfuse

            langfuse_factory = Langfuse
        arguments = {"public_key": public_key, "secret_key": secret_key}
        host = environment.get("LANGFUSE_HOST")
        if host:
            arguments["host"] = host
        sdk_client = langfuse_factory(**arguments)
        return LangfuseObservabilityClient(sdk_client, logger=selected_logger)
    except Exception as error:
        selected_logger.warning(
            "Observabilidad Langfuse no disponible; se usa NoOp: %s", error
        )
        return NoOpObservabilityClient()
