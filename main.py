"""Chat interactivo del coding agent."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from uuid import uuid4
from typing import Any

from dotenv import load_dotenv

from core.harness import run_internal_loop, run_planning_loop
from core.llm_client import LLMClient, LLMClientError, ObservedLLMClient, OpenAILLMClient
from core.models import Message, PlanReview
from core.settings import AgentSettings
from core.observability import (
    NoOpObservabilityClient, ObservabilityClient, ObservabilityEvent,
    build_observability_client, emit_observation,
)
from core.supervision import ConfirmationCallback
from tools.definitions import ToolDefinition
from tools.registry import ToolRegistry, build_default_registry


InputCallback = Callable[[str], str]
OutputCallback = Callable[[str], None]
SYSTEM_PROMPT = (
    "Sos un coding agent. Usá las tools disponibles cuando sean necesarias, "
    "respetá el workspace y nunca expongas secretos ni ejecutes acciones destructivas. "
    "En la respuesta final explicá en lenguaje natural, cuando corresponda, qué "
    "encontraste, qué cambiaste, cómo lo verificaste, qué archivos modificaste y "
    "cuál fue el resultado de los tests."
)


def load_environment(dotenv_path: str | Path | None = None) -> None:
    """Carga variables desde ``.env`` sin imprimir ni devolver secretos."""
    load_dotenv(dotenv_path=dotenv_path)


@dataclass(frozen=True)
class CommandResult:
    """Resultado del procesamiento de una posible orden del chat."""

    handled: bool
    exit_requested: bool = False


def process_command(
    message: str,
    settings: AgentSettings,
    output: OutputCallback = print,
) -> CommandResult:
    """Procesa comandos locales sin enviarlos al modelo."""
    normalized = " ".join(message.strip().lower().split())
    if not normalized.startswith("/"):
        return CommandResult(handled=False)

    if normalized == "/exit":
        return CommandResult(handled=True, exit_requested=True)
    if normalized == "/status":
        output(_format_status(settings))
        return CommandResult(handled=True)
    if normalized in {"/plan on", "/plan off"}:
        settings.plan_mode_enabled = normalized.endswith("on")
        output(f"Plan mode: {_on_off(settings.plan_mode_enabled)}")
        return CommandResult(handled=True)
    if normalized in {"/supervision on", "/supervision off"}:
        settings.supervision_enabled = normalized.endswith("on")
        output(f"Supervision mode: {_on_off(settings.supervision_enabled)}")
        return CommandResult(handled=True)

    output(f"Comando desconocido: {message.strip()}")
    return CommandResult(handled=True)


def run_chat(
    llm_client: LLMClient,
    tool_registry: ToolRegistry,
    settings: AgentSettings,
    *,
    input_func: InputCallback = input,
    output: OutputCallback = print,
    confirm: ConfirmationCallback | None = None,
    observability: ObservabilityClient | None = None,
) -> list[Message]:
    """Ejecuta el loop externo y devuelve el historial al finalizar."""
    history = [Message(role="system", content=SYSTEM_PROMPT)]
    approval = confirm or _interactive_confirmation(input_func, output)
    plan_review = _interactive_plan_review(input_func, output)
    observed = observability or NoOpObservabilityClient()

    while True:
        try:
            raw_message = input_func("Usuario> ")
        except KeyboardInterrupt:
            output("\nInterrumpido por el usuario.")
            break
        except EOFError:
            output("\nFin de entrada.")
            break

        message = raw_message.strip()
        if not message:
            continue

        command = process_command(message, settings, output)
        if command.exit_requested:
            output("Hasta luego.")
            break
        if command.handled:
            continue

        history.append(Message(role="user", content=message))
        task_id = str(uuid4())
        root_id = f"task:{task_id}"
        started = perf_counter()
        emit_observation(
            observed,
            ObservabilityEvent(
                "task", "coding-agent-task", event_id=root_id, task_id=task_id,
                model=(llm_client.model_name if isinstance(llm_client, ObservedLLMClient) else None),
                payload={"request": message, "phase": "intake",
                         "project": {"workspace": "workspace"}},
            ),
        )
        if isinstance(llm_client, ObservedLLMClient):
            llm_client.set_observation_context(
                task_id=task_id, parent_event_id=root_id, agent="main"
            )
        try:
            if settings.plan_mode_enabled:
                planning = run_planning_loop(
                    llm_client,
                    history,
                    plan_review,
                    max_revisions=settings.max_iterations,
                    output=output,
                )
                if not planning.approved:
                    history.append(
                        Message(
                            role="assistant",
                            content="Tarea cancelada por el usuario durante plan mode.",
                        )
                    )
                    output("Tarea cancelada.")
                    continue
                assert planning.plan is not None
                history.append(
                    Message(role="assistant", content=planning.plan)
                )
                history.append(
                    Message(
                        role="developer",
                        content=(
                            "El usuario aprobó el plan anterior. Ejecutalo ahora usando "
                            "las tools disponibles cuando corresponda."
                        ),
                    )
                )
                output("")

            result = run_internal_loop(
                llm_client,
                tool_registry,
                settings,
                history,
                approval,
                output,
                observability=observed,
                task_id=task_id,
                parent_event_id=root_id,
            )
        except KeyboardInterrupt:
            output("\nInterrumpido por el usuario.")
            break
        except EOFError:
            output("\nFin de entrada.")
            break
        except Exception as error:
            emit_observation(
                observed,
                ObservabilityEvent(
                    "error", "task-error", task_id=task_id,
                    parent_event_id=root_id,
                    payload={"status": "error", "error": str(error)},
                    latency_ms=(perf_counter() - started) * 1000,
                ),
            )
            output(f"Error: {error}")
            continue

        output("--- Respuesta final ---")
        output(result.response.text)
        output(f"Iteraciones del turno: {result.iterations}")
        emit_observation(
            observed,
            ObservabilityEvent(
                "result", "task-finished", task_id=task_id,
                parent_event_id=root_id, model=result.response.model,
                payload={"status": "completed", "result": result.response.text,
                         "files_modified": [], "iterations": result.iterations,
                         "error_count": 0, "sources": []},
                total_tokens=(llm_client.total_usage.total_tokens
                              if isinstance(llm_client, ObservedLLMClient)
                              else result.response.usage.total_tokens),
                input_tokens=(llm_client.total_usage.input_tokens
                              if isinstance(llm_client, ObservedLLMClient)
                              else result.response.usage.input_tokens),
                output_tokens=(llm_client.total_usage.output_tokens
                               if isinstance(llm_client, ObservedLLMClient)
                               else result.response.usage.output_tokens),
                latency_ms=(perf_counter() - started) * 1000,
                estimated_cost=None,
            ),
        )

    return history


def _interactive_confirmation(
    input_func: InputCallback, output: OutputCallback
) -> ConfirmationCallback:
    """Crea el callback de aprobación usado por tools supervisadas."""

    def confirm(tool: ToolDefinition, arguments: dict[str, Any]) -> bool:
        answer = input_func(f"¿Aprobar {tool.name}? [s/n]: ").strip().lower()
        approved = answer in {"s", "si", "sí", "y", "yes"}
        if not approved:
            output(f"Tool rechazada: {tool.name}")
        return approved

    return confirm


def _interactive_plan_review(
    input_func: InputCallback, output: OutputCallback
) -> Callable[[str], PlanReview]:
    """Crea el diálogo de aprobación, rechazo o modificación de planes."""

    def review(plan: str) -> PlanReview:
        while True:
            raw_answer = input_func(
                "Plan: [a]probar, [r]echazar o [m]odificar: "
            )
            answer = raw_answer.strip().lower()
            if answer in {"a", "aprobar"}:
                return PlanReview("approve")
            if answer in {"r", "rechazar"}:
                return PlanReview("reject")
            if answer in {"m", "modificar"}:
                modification = input_func("Modificación solicitada: ").strip()
                if modification:
                    return PlanReview("modify", modification)
                output("La modificación no puede estar vacía.")
                continue
            output("Opción inválida. Usá a, r o m.")

    return review


def _format_status(settings: AgentSettings) -> str:
    """Devuelve las configuraciones visibles de la sesión."""
    return "\n".join(
        (
            f"Plan mode: {_on_off(settings.plan_mode_enabled)}",
            f"Supervision mode: {_on_off(settings.supervision_enabled)}",
            f"Max iterations: {settings.max_iterations}",
            f"Command timeout: {settings.command_timeout_seconds}s",
        )
    )


def _on_off(enabled: bool) -> str:
    return "on" if enabled else "off"


def main() -> None:
    """Construye las dependencias concretas e inicia el chat."""
    try:
        load_environment()
        settings = AgentSettings.from_environment()
        observability = build_observability_client()
        client = ObservedLLMClient(OpenAILLMClient(), observability)
        registry = build_default_registry()
        run_chat(client, registry, settings, observability=observability)
        observability.flush()
    except LLMClientError as error:
        print(f"Error de configuración del LLM: {error}")
    except Exception as error:
        print(f"Error al iniciar el agente: {error}")


if __name__ == "__main__":
    main()
