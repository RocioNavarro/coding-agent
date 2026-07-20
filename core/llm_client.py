"""Abstracción LLM y adaptador para la Responses API de OpenAI."""

from __future__ import annotations

import json
import os
from time import perf_counter
from typing import Any, Protocol, Sequence

from openai import OpenAI, OpenAIError

from core.models import LLMResponse, LLMUsage, Message, ToolCall
from core.observability import (
    NoOpObservabilityClient, ObservabilityClient, ObservabilityEvent, emit_observation,
    estimate_cost,
)


class LLMClientError(RuntimeError):
    """Error base controlado de la capa LLM."""


class LLMConfigurationError(LLMClientError):
    """La configuración necesaria para crear el cliente es inválida."""


class LLMProviderError(LLMClientError):
    """El proveedor rechazó o no pudo completar la solicitud."""


class LLMInvalidResponseError(LLMClientError):
    """El proveedor devolvió una respuesta que no cumple el contrato esperado."""


class LLMClient(Protocol):
    """Contrato independiente del proveedor usado por el futuro harness."""

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] = (),
    ) -> LLMResponse:
        """Genera una respuesta para el historial y las tools disponibles."""
        ...


class OpenAILLMClient:
    """Implementación de :class:`LLMClient` mediante OpenAI Responses API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        client: Any | None = None,
    ) -> None:
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        resolved_model = model or os.getenv("OPENAI_MODEL")
        if not resolved_api_key:
            raise LLMConfigurationError("Falta la variable de entorno OPENAI_API_KEY.")
        if not resolved_model:
            raise LLMConfigurationError("Falta la variable de entorno OPENAI_MODEL.")

        self._model = resolved_model
        self._client = client if client is not None else OpenAI(api_key=resolved_api_key)

    @property
    def model_name(self) -> str:
        return self._model

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] = (),
    ) -> LLMResponse:
        """Solicita una respuesta y la convierte al contrato interno."""
        request: dict[str, Any] = {
            "model": self._model,
            "input": self._messages_to_input(messages),
        }
        if tools:
            request["tools"] = [self._to_responses_tool(tool) for tool in tools]

        started_at = perf_counter()
        try:
            response = self._client.responses.create(**request)
        except OpenAIError as exc:
            raise LLMProviderError(f"Error del proveedor OpenAI: {exc}") from exc
        latency_ms = (perf_counter() - started_at) * 1000

        return self._parse_response(response, latency_ms)

    @staticmethod
    def _messages_to_input(messages: Sequence[Message]) -> list[dict[str, Any]]:
        """Convierte el historial propio a items de entrada de Responses API."""
        items: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "tool":
                if not message.tool_call_id:
                    raise LLMConfigurationError(
                        "Un mensaje de tool debe incluir tool_call_id."
                    )
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.tool_call_id,
                        "output": message.content,
                    }
                )
                continue

            if message.content:
                items.append({"role": message.role, "content": message.content})
            for tool_call in message.tool_calls:
                if message.role != "assistant":
                    raise LLMConfigurationError(
                        "Sólo un mensaje assistant puede contener tool calls."
                    )
                items.append(
                    {
                        "type": "function_call",
                        "call_id": tool_call.id,
                        "name": tool_call.name,
                        "arguments": json.dumps(
                            tool_call.arguments, ensure_ascii=False, separators=(",", ":")
                        ),
                    }
                )
        return items

    @staticmethod
    def _to_responses_tool(tool: dict[str, Any]) -> dict[str, Any]:
        """Acepta schemas internos/Chat Completions y produce el formato Responses."""
        if tool.get("type") != "function":
            raise LLMConfigurationError("Sólo se admiten tools de tipo 'function'.")

        function = tool.get("function", tool)
        try:
            converted = {
                "type": "function",
                "name": function["name"],
                "description": function["description"],
                "parameters": function["parameters"],
            }
        except (KeyError, TypeError) as exc:
            raise LLMConfigurationError("La definición de una tool es inválida.") from exc
        if "strict" in function:
            converted["strict"] = function["strict"]
        return converted

    @staticmethod
    def _parse_response(response: Any, latency_ms: float) -> LLMResponse:
        """Valida y desacopla una respuesta creada por el SDK."""
        try:
            if response.status != "completed":
                raise LLMInvalidResponseError(
                    f"La respuesta de OpenAI no se completó (estado: {response.status!r})."
                )
            if not isinstance(response.model, str) or not response.model:
                raise LLMInvalidResponseError("La respuesta no incluye un modelo válido.")
            if response.usage is None:
                raise LLMInvalidResponseError("La respuesta no incluye métricas de uso.")

            text = response.output_text or ""
            tool_calls: list[ToolCall] = []
            for item in response.output:
                if item.type != "function_call":
                    continue
                arguments = json.loads(item.arguments)
                if not isinstance(arguments, dict):
                    raise LLMInvalidResponseError(
                        f"Los argumentos de la tool {item.name!r} no son un objeto JSON."
                    )
                call_id = item.call_id
                if not isinstance(call_id, str) or not call_id:
                    raise LLMInvalidResponseError("Una tool call no incluye call_id válido.")
                tool_calls.append(
                    ToolCall(id=call_id, name=item.name, arguments=arguments)
                )

            usage = LLMUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=response.usage.total_tokens,
            )
            assistant_message = Message(
                role="assistant", content=text, tool_calls=tool_calls
            )
            return LLMResponse(
                assistant_message=assistant_message,
                text=text,
                tool_calls=tool_calls,
                model=response.model,
                usage=usage,
                latency_ms=latency_ms,
            )
        except LLMInvalidResponseError:
            raise
        except json.JSONDecodeError as exc:
            raise LLMInvalidResponseError(
                "OpenAI devolvió argumentos JSON inválidos en una tool call."
            ) from exc
        except (AttributeError, TypeError, ValueError) as exc:
            raise LLMInvalidResponseError(
                "OpenAI devolvió una respuesta con formato inválido."
            ) from exc


class ObservedLLMClient:
    """Instrumenta una única vez cualquier implementación de LLMClient."""

    def __init__(
        self,
        client: LLMClient,
        observability: ObservabilityClient | None = None,
    ) -> None:
        self._client = client
        self._observability = observability or NoOpObservabilityClient()
        self._task_id: str | None = None
        self._parent_event_id: str | None = None
        self._agent: str | None = None
        self._usage = LLMUsage(0, 0, 0)

    @property
    def model_name(self) -> str | None:
        value = getattr(self._client, "model_name", None)
        return value if isinstance(value, str) else None

    @property
    def total_usage(self) -> LLMUsage:
        return self._usage

    def set_observation_context(
        self, *, task_id: str | None, parent_event_id: str | None, agent: str | None = None
    ) -> None:
        self._task_id = task_id
        self._parent_event_id = parent_event_id
        self._agent = agent
        self._usage = LLMUsage(0, 0, 0)

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] = (),
    ) -> LLMResponse:
        started = perf_counter()
        prompt = [
            {"role": message.role, "content": message.content}
            for message in messages
        ]
        try:
            response = self._client.complete(messages, tools)
        except Exception as error:
            emit_observation(
                self._observability,
                ObservabilityEvent(
                    "llm_call", "llm-error", task_id=self._task_id,
                    parent_event_id=self._parent_event_id, agent=self._agent,
                    payload={"prompt": prompt, "error": str(error)},
                    latency_ms=(perf_counter() - started) * 1000,
                ),
            )
            raise
        emit_observation(
            self._observability,
            ObservabilityEvent(
                "llm_call", "llm-completion", task_id=self._task_id,
                parent_event_id=self._parent_event_id, agent=self._agent,
                model=response.model,
                payload={
                    "prompt": prompt, "response": response.text,
                    "tool_calls": [call.name for call in response.tool_calls],
                },
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=response.usage.total_tokens,
                latency_ms=response.latency_ms,
                estimated_cost=estimate_cost(
                    response.model, response.usage.input_tokens, response.usage.output_tokens
                ),
            ),
        )
        self._usage = LLMUsage(
            self._usage.input_tokens + response.usage.input_tokens,
            self._usage.output_tokens + response.usage.output_tokens,
            self._usage.total_tokens + response.usage.total_tokens,
        )
        return response
