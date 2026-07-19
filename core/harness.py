"""Loop interno que conecta el LLM con las tools locales."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from core.llm_client import LLMClient
from core.models import InternalLoopResult, Message, PlanningResult, PlanReview
from core.settings import AgentSettings
from core.supervision import ConfirmationCallback, SupervisedToolExecutor
from tools.registry import ToolRegistry


OutputCallback = Callable[[str], None]
PlanReviewer = Callable[[str], PlanReview]
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


class PlanningError(RuntimeError):
    """No fue posible obtener un plan válido y aprobado."""


def run_planning_loop(
    llm_client: LLMClient,
    history: list[Message],
    review: PlanReviewer,
    *,
    max_revisions: int,
    output: OutputCallback = print,
) -> PlanningResult:
    """Genera y revisa planes sin exponer schemas ni modificar el historial principal."""
    planning_history = list(history)
    planning_history.append(
        Message(
            role="developer",
            content=(
                "Proponé un plan numerado, concreto y breve para resolver el último "
                "pedido. No ejecutes acciones ni afirmes que ya fueron realizadas."
            ),
        )
    )

    for _ in range(max_revisions):
        response = llm_client.complete(planning_history, ())
        plan = response.text.strip()
        if response.tool_calls:
            raise PlanningError("El LLM intentó usar tools durante la planificación.")
        if not plan:
            raise PlanningError("El LLM devolvió un plan vacío.")

        output(f"\nPlan propuesto:\n\n{plan}")
        decision = review(plan)
        if decision.action == "approve":
            return PlanningResult(approved=True, plan=plan)
        if decision.action == "reject":
            return PlanningResult(approved=False)
        if decision.action != "modify" or not decision.modification:
            raise PlanningError("La modificación solicitada no es válida.")

        planning_history.append(response.assistant_message)
        planning_history.append(
            Message(
                role="developer",
                content=(
                    "El usuario solicita modificar el plan de esta manera: "
                    f"{decision.modification}. Generá un nuevo plan completo."
                ),
            )
        )

    raise PlanningError(
        f"Se alcanzó el máximo de {max_revisions} revisiones del plan."
    )


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
        output(_format_iteration_heading(iteration))
        response = llm_client.complete(history, tool_registry.list_schemas())
        history.append(response.assistant_message)

        if not response.tool_calls:
            return InternalLoopResult(response=response, iterations=iteration)

        for tool_call in response.tool_calls:
            output(_describe_tool(tool_call.name, tool_call.arguments))
            output(f"Tool: {tool_call.name}")
            output(f"Argumentos:\n{_format_arguments(tool_call.arguments)}")
            execution = executor.execute(tool_call.name, tool_call.arguments)
            output(f"Resultado:\n{_format_result(execution)}")
            history.append(
                Message(
                    role="tool",
                    content=json.dumps(execution, ensure_ascii=False, default=str),
                    tool_call_id=tool_call.id,
                )
            )

    raise MaxIterationsError(settings.max_iterations)


def _format_iteration_heading(iteration: int) -> str:
    """Separa iteraciones posteriores con exactamente una línea visual."""
    prefix = "" if iteration == 1 else "\n"
    return f"{prefix}--- Iteración {iteration} ---"


def _describe_tool(name: str, arguments: dict[str, Any]) -> str:
    """Explica brevemente la intención de una tool sin alterar su ejecución."""
    if name == "list_files":
        path = _display_text(arguments.get("path", "."), 120)
        if path == ".":
            return "El agente está explorando los archivos del proyecto."
        return f"El agente está explorando el directorio {path}."
    if name == "read_file":
        path = _display_text(arguments.get("path", "un archivo"), 120)
        return f"El agente está leyendo {path}."
    if name == "write_file":
        path = _display_text(arguments.get("path", "un archivo"), 120)
        return f"El agente quiere modificar {path}."
    if name == "run_command":
        command = _display_text(arguments.get("command", ""), 160)
        if "pytest" in command.lower() or "test" in command.lower():
            return "El agente quiere ejecutar los tests."
        return f"El agente quiere ejecutar el comando: {command}."
    if name == "web_search":
        query = _display_text(arguments.get("query", "la consulta indicada"), 140)
        return f"El agente está buscando en la web: {query}."
    return f"El agente quiere usar la tool {name}."


def _format_arguments(arguments: dict[str, Any]) -> str:
    """Presenta argumentos legibles y omite cuerpos completos de escritura."""
    if not arguments:
        return "  (sin argumentos)"

    safe_arguments = _redact(arguments)
    lines: list[str] = []
    for key, value in safe_arguments.items():
        if key == "content" and isinstance(arguments.get(key), str):
            length = len(arguments[key])
            rendered = f"<{length} caracteres; contenido omitido>"
        else:
            rendered = _display_text(value, 240)
        lines.append(f"  {key}: {rendered}")
    return "\n".join(lines)


def _format_result(execution: dict[str, Any]) -> str:
    """Conserva estado y detalles técnicos con una presentación compacta."""
    if not execution.get("success"):
        error = _display_text(execution.get("error") or "Error desconocido.", 300)
        return f"  estado: error\n  error: {error}"

    result = execution.get("result")
    lines = ["  estado: ok"]
    if isinstance(result, dict) and {"exit_code", "stdout", "stderr"} <= set(result):
        lines.append(f"  exit_code: {_display_text(result['exit_code'], 80)}")
        lines.append(f"  stdout: {_display_text(result['stdout'] or '(vacío)', 300)}")
        lines.append(f"  stderr: {_display_text(result['stderr'] or '(vacío)', 300)}")
    else:
        lines.append(f"  valor: {_display_text(result, 300)}")
    return "\n".join(lines)


def _display_text(value: Any, max_length: int) -> str:
    """Convierte un valor a texto seguro, legible y acotado."""
    safe_value = _redact(value)
    if isinstance(safe_value, str):
        rendered = " ".join(safe_value.split())
    elif safe_value is None:
        rendered = "(vacío)"
    elif isinstance(safe_value, (int, float, bool)):
        rendered = str(safe_value)
    else:
        try:
            rendered = json.dumps(safe_value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            rendered = str(safe_value)
    rendered = _SECRET_VALUE.sub(r"\1\2***", rendered)
    if len(rendered) > max_length:
        return f"{rendered[:max_length]}… [truncado]"
    return rendered


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
