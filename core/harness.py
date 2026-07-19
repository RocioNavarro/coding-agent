"""Loop interno que conecta el LLM con las tools locales."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from core.llm_client import LLMClient
from core.models import InternalLoopResult, Message
from core.settings import AgentSettings
from core.supervision import ConfirmationCallback, SupervisedToolExecutor
from tools.registry import ToolRegistry


OutputCallback = Callable[[str], None]
_SECRET_FIELD = re.compile(r"(?:api[_-]?key|password|secret|token)", re.IGNORECASE)
_SECRET_VALUE = re.compile(
    r"(?i)(api[_-]?key|password|secret|token)(\s*[=:]\s*)([^\s,;]+)"
)


class MaxIterationsError(RuntimeError):
    """El modelo siguió solicitando tools después del límite configurado."""

    def __init__(self, max_iterations: int) -> None:
        super().__init__(
            f"Se alcanzó el máximo de {max_iterations} iteraciones sin una respuesta final."
        )
        self.max_iterations = max_iterations


def run_internal_loop(
    llm_client: LLMClient,
    tool_registry: ToolRegistry,
    settings: AgentSettings,
    history: list[Message],
    confirm: ConfirmationCallback | None = None,
    output: OutputCallback = print,
) -> InternalLoopResult:
    """Ejecuta tool calling hasta obtener texto final o agotar el límite."""
    executor = SupervisedToolExecutor(tool_registry, settings, confirm)

    for iteration in range(1, settings.max_iterations + 1):
        output(f"--- Iteración {iteration} ---")
        response = llm_client.complete(history, tool_registry.list_schemas())
        history.append(response.assistant_message)

        if not response.tool_calls:
            return InternalLoopResult(response=response, iterations=iteration)

        for tool_call in response.tool_calls:
            output(f"Tool: {tool_call.name}")
            output(f"Argumentos: {_summarize(tool_call.arguments)}")
            execution = executor.execute(tool_call.name, tool_call.arguments)
            output(f"Resultado: {_summarize(execution)}")
            history.append(
                Message(
                    role="tool",
                    content=json.dumps(execution, ensure_ascii=False, default=str),
                    tool_call_id=tool_call.id,
                )
            )

    raise MaxIterationsError(settings.max_iterations)


def _summarize(value: Any, max_length: int = 300) -> str:
    """Produce una vista breve y redactada para el registro visible."""
    safe_value = _redact(value)
    try:
        rendered = json.dumps(safe_value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        rendered = str(safe_value)
    rendered = _SECRET_VALUE.sub(r"\1\2***", rendered)
    if len(rendered) > max_length:
        return f"{rendered[:max_length]}… [truncado]"
    return rendered


def _redact(value: Any) -> Any:
    """Oculta valores asociados a nombres habituales de secretos."""
    if isinstance(value, dict):
        return {
            str(key): "***" if _SECRET_FIELD.search(str(key)) else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact(item) for item in value)
    return value
