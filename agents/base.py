"""Runtime mínimo y desacoplado para futuros subagentes."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Sequence

from core.llm_client import LLMClient
from core.models import LLMResponse, Message, ToolCall
from core.task_state import ErrorRecord, SourceReference, SubagentResult, TaskState
from tools.registry import ToolRegistry


class AgentExecutionError(RuntimeError):
    """Error controlado durante la ejecución o normalización de un subagente."""


@dataclass(frozen=True)
class AgentContext:
    """Contexto seleccionado explícitamente para una ejecución de subagente."""

    facts: tuple[str, ...] = ()
    sources: tuple[SourceReference, ...] = ()
    files: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("facts", "files", "constraints"):
            values = getattr(self, field_name)
            if not isinstance(values, (list, tuple)) or not all(
                isinstance(item, str) and item.strip() for item in values
            ):
                raise ValueError(f"{field_name} debe contener textos no vacíos.")
            object.__setattr__(self, field_name, tuple(item.strip() for item in values))
        if not isinstance(self.sources, (list, tuple)) or not all(
            isinstance(item, SourceReference) for item in self.sources
        ):
            raise ValueError("sources debe contener instancias de SourceReference.")
        object.__setattr__(self, "sources", tuple(self.sources))

    def to_dict(self) -> dict[str, Any]:
        return {
            "facts": list(self.facts),
            "sources": [source.to_dict() for source in self.sources],
            "files": list(self.files),
            "constraints": list(self.constraints),
        }


@dataclass(frozen=True)
class AgentInput:
    """Entrada acotada que BaseAgent transforma en mensajes para el LLM."""

    instruction: str
    task_id: str
    context: AgentContext

    def __post_init__(self) -> None:
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError("instruction no puede estar vacía.")
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id no puede estar vacío.")
        if not isinstance(self.context, AgentContext):
            raise TypeError("context debe ser una instancia de AgentContext.")
        object.__setattr__(self, "instruction", self.instruction.strip())
        object.__setattr__(self, "task_id", self.task_id.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "context": self.context.to_dict(),
        }


class BaseAgent(ABC):
    """Base común que limita contexto y tools y normaliza la salida del LLM."""

    _OUTPUT_INSTRUCTION = (
        "Respondé únicamente con un objeto JSON con estas claves: summary (string), "
        "findings (lista de strings), recommendations (lista de strings), sources "
        "(lista de objetos con origin, reference y summary opcional; origin debe ser "
        "exactamente uno de estos 5 valores literales, sin traducir ni parafrasear: "
        "\"repository\", \"project_memory\", \"rag\", \"web\", \"inference\"), "
        "files_relevant (lista de strings), blockers (lista de strings) y confidence "
        "(número de 0 a 1). Si necesitás una tool, solicitala mediante tool calling. "
        "No asumas contexto que no aparezca en la entrada."
    )

    _MAX_OUTPUT_ATTEMPTS = 2
    """Reintentos ante salida mal formada del LLM (no ante políticas rechazadas)."""

    _MAX_TOOL_ROUNDS = 3
    """Rondas de tool calling ejecutadas antes de forzar una respuesta sin tools."""

    def __init__(
        self,
        *,
        name: str,
        role: str,
        system_prompt: str,
        allowed_tools: Sequence[str],
        llm_client: LLMClient,
    ) -> None:
        self.name = self._required_text(name, "name")
        self.role = self._required_text(role, "role")
        self.system_prompt = self._required_text(system_prompt, "system_prompt")
        self.allowed_tools = frozenset(
            self._required_text(tool, "allowed_tool") for tool in allowed_tools
        )
        self.llm_client = llm_client

    @abstractmethod
    def specialization_prompt(self) -> str:
        """Devuelve instrucciones adicionales propias del tipo concreto de agente."""

    @contextmanager
    def _error_guard(
        self, task_state: TaskState, *, action: str = "completar la tarea"
    ) -> Iterator[None]:
        """Convierte fallas no controladas en AgentExecutionError y las registra.

        Todo subagente debe envolver su lógica de ``run`` con este context manager,
        para que un fallo (propio o de este agente) siempre quede auditado en
        ``task_state.errors`` antes de propagarse, sin excepciones sin registrar
        entre distintos subagentes.
        """
        try:
            yield
        except Exception as error:
            if isinstance(error, AgentExecutionError):
                controlled_error = error
            else:
                controlled_error = AgentExecutionError(
                    f"El agente '{self.name}' no pudo {action}: {error}"
                )
            task_state.record_error(
                ErrorRecord(
                    message=str(controlled_error),
                    phase=task_state.current_phase,
                    component=self.name,
                    recoverable=True,
                )
            )
            if controlled_error is error:
                raise
            raise controlled_error from error

    def run(
        self,
        instruction: str,
        task_state: TaskState,
        context: AgentContext | None = None,
        available_tools: ToolRegistry | None = None,
    ) -> SubagentResult:
        """Ejecuta una llamada acotada al LLM y agrega su resultado a TaskState."""
        if not isinstance(task_state, TaskState):
            raise TypeError("task_state debe ser una instancia de TaskState.")
        agent_input = AgentInput(
            instruction=instruction,
            task_id=task_state.task_id,
            context=context or AgentContext(),
        )
        tools = available_tools or ToolRegistry()

        with self._error_guard(task_state):
            result = self._complete_and_parse(
                agent_input, tools,
                lambda response: self.validate_tool_calls(response.tool_calls, tools),
            )

        task_state.add_subagent_result(result)
        for source in result.sources:
            task_state.add_source(source)
        return result

    def _complete_and_parse(
        self,
        agent_input: AgentInput,
        tools: ToolRegistry,
        validate: Callable[[LLMResponse], None],
    ) -> SubagentResult:
        """Reintenta sólo cuando la respuesta final del LLM no se puede normalizar.

        ``validate`` corre sobre la respuesta final (ya sin tool calls pendientes)
        y puede rechazarla —ej. una tool no permitida— sin reintentar: ese tipo de
        rechazo es determinista y volvería a fallar igual. Sólo se reintenta cuando
        ``to_subagent_result`` falla por una salida mal formada del LLM.
        """
        schemas = self._allowed_schemas(tools)
        last_error: AgentExecutionError | None = None
        for _ in range(self._MAX_OUTPUT_ATTEMPTS):
            response = self._complete_until_text(agent_input, schemas, tools)
            validate(response)
            try:
                return self.to_subagent_result(agent_input, response)
            except AgentExecutionError as error:
                last_error = error
        assert last_error is not None
        raise last_error

    def _complete_until_text(
        self,
        agent_input: AgentInput,
        schemas: Sequence[dict[str, Any]],
        tools: ToolRegistry,
    ) -> LLMResponse:
        """Ejecuta las tool calls que pida el LLM hasta obtener una respuesta de texto.

        Antes, el LLM podía pedir una tool (autorizada por el prompt) que nunca se
        ejecutaba: la respuesta quedaba sin texto y ``to_subagent_result`` fallaba
        siempre. Ahora se ejecuta cada tool call —ya validada y confinada por
        ``allowed_tools``— y se le devuelve el resultado al modelo, acotado a
        ``_MAX_TOOL_ROUNDS`` rondas antes de forzar una respuesta final sin tools.
        """
        messages = self.build_context(agent_input)
        for _ in range(self._MAX_TOOL_ROUNDS):
            response = self.llm_client.complete(messages, schemas)
            if not response.tool_calls:
                return response
            self.validate_tool_calls(response.tool_calls, tools)
            messages = [*messages, response.assistant_message]
            for call in response.tool_calls:
                execution = tools.execute(call.name, call.arguments)
                messages.append(
                    Message(
                        role="tool",
                        content=json.dumps(execution, ensure_ascii=False, default=str),
                        tool_call_id=call.id,
                    )
                )
        return self.llm_client.complete(messages, ())

    def build_context(self, agent_input: AgentInput) -> list[Message]:
        """Construye mensajes sólo con la entrada explícita de esta ejecución."""
        specialization = self.specialization_prompt().strip()
        prompt_parts = [
            f"Nombre del agente: {self.name}\nRol: {self.role}",
            self.system_prompt,
        ]
        if specialization:
            prompt_parts.append(specialization)
        prompt_parts.append(self._OUTPUT_INSTRUCTION)
        return [
            Message(role="system", content="\n\n".join(prompt_parts)),
            Message(
                role="user",
                content=json.dumps(agent_input.to_dict(), ensure_ascii=False),
            ),
        ]

    def validate_tool_calls(
        self, tool_calls: Sequence[ToolCall], available_tools: ToolRegistry
    ) -> None:
        """Rechaza calls fuera del allowlist o no disponibles en esta ejecución."""
        for call in tool_calls:
            if call.name not in self.allowed_tools:
                raise AgentExecutionError(
                    f"La tool '{call.name}' no está permitida para el agente '{self.name}'."
                )
            if available_tools.get(call.name) is None:
                raise AgentExecutionError(
                    f"La tool '{call.name}' no está disponible para el agente '{self.name}'."
                )
            try:
                available_tools.validate_arguments(call.name, call.arguments)
            except Exception as error:
                raise AgentExecutionError(
                    f"Argumentos inválidos para la tool '{call.name}': {error}"
                ) from error

    def to_subagent_result(
        self, agent_input: AgentInput, response: LLMResponse
    ) -> SubagentResult:
        """Convierte JSON y tool calls normalizados al modelo compartido."""
        try:
            payload = json.loads(response.text)
        except (json.JSONDecodeError, TypeError) as error:
            raise AgentExecutionError(
                f"El agente '{self.name}' devolvió una respuesta JSON inválida."
            ) from error
        if not isinstance(payload, dict):
            raise AgentExecutionError("La respuesta estructurada debe ser un objeto JSON.")

        summary = self._payload_text(payload, "summary")
        sources = tuple(
            SourceReference.from_dict(source)
            for source in self._payload_list(payload, "sources")
        )
        return SubagentResult(
            subagent_id=self.name,
            task=agent_input.instruction,
            status="completed" if not payload.get("blockers") else "blocked",
            result=summary,
            summary=summary,
            findings=self._payload_text_list(payload, "findings"),
            recommendations=self._payload_text_list(payload, "recommendations"),
            requested_tool_calls=tuple(response.tool_calls),
            sources=sources,
            files_relevant=self._payload_text_list(payload, "files_relevant"),
            blockers=self._payload_text_list(payload, "blockers"),
            confidence=self._confidence(payload.get("confidence")),
        )

    def _allowed_schemas(self, available_tools: ToolRegistry) -> list[dict[str, Any]]:
        schemas: list[dict[str, Any]] = []
        for name in sorted(self.allowed_tools):
            tool = available_tools.get(name)
            if tool is not None:
                schemas.append(tool.to_llm_schema())
        return schemas

    @staticmethod
    def _payload_list(payload: Mapping[str, Any], field_name: str) -> list[Any]:
        value = payload.get(field_name, [])
        if not isinstance(value, list):
            raise AgentExecutionError(f"El campo '{field_name}' debe ser una lista.")
        return value

    @classmethod
    def _payload_text_list(
        cls, payload: Mapping[str, Any], field_name: str
    ) -> tuple[str, ...]:
        values = cls._payload_list(payload, field_name)
        if not all(isinstance(item, str) and item.strip() for item in values):
            raise AgentExecutionError(
                f"El campo '{field_name}' debe contener strings no vacíos."
            )
        return tuple(item.strip() for item in values)

    @staticmethod
    def _payload_text(payload: Mapping[str, Any], field_name: str) -> str:
        value = payload.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise AgentExecutionError(f"El campo '{field_name}' debe ser un string.")
        return value.strip()

    @staticmethod
    def _confidence(value: object) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= value <= 1
        ):
            raise AgentExecutionError("El campo 'confidence' debe estar entre 0 y 1.")
        return float(value)

    @staticmethod
    def _required_text(value: object, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} no puede estar vacío.")
        return value.strip()


class StubAgent(BaseAgent):
    """Implementación mínima para tests y flujos sin un rol especializado."""

    def specialization_prompt(self) -> str:
        return ""
