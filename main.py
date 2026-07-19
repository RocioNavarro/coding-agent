"""Chat interactivo del coding agent."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from core.harness import run_internal_loop
from core.llm_client import LLMClient, LLMClientError, OpenAILLMClient
from core.models import Message
from core.settings import AgentSettings
from core.supervision import ConfirmationCallback
from tools.definitions import ToolDefinition
from tools.registry import ToolRegistry, build_default_registry


InputCallback = Callable[[str], str]
OutputCallback = Callable[[str], None]
SYSTEM_PROMPT = (
    "Sos un coding agent. Usá las tools disponibles cuando sean necesarias, "
    "respetá el workspace y nunca expongas secretos ni ejecutes acciones destructivas."
)


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
) -> list[Message]:
    """Ejecuta el loop externo y devuelve el historial al finalizar."""
    history = [Message(role="system", content=SYSTEM_PROMPT)]
    approval = confirm or _interactive_confirmation(input_func, output)

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
        try:
            result = run_internal_loop(
                llm_client,
                tool_registry,
                settings,
                history,
                approval,
                output,
            )
        except KeyboardInterrupt:
            output("\nInterrumpido por el usuario.")
            break
        except EOFError:
            output("\nFin de entrada.")
            break
        except Exception as error:
            output(f"Error: {error}")
            continue

        output(f"Asistente: {result.response.text}")
        output(f"Iteraciones: {result.iterations}")

    return history


def _interactive_confirmation(
    input_func: InputCallback, output: OutputCallback
) -> ConfirmationCallback:
    """Crea el callback de aprobación usado por tools supervisadas."""

    def confirm(tool: ToolDefinition, arguments: dict[str, Any]) -> bool:
        answer = input_func(f"¿Aprobar {tool.name}? [s/N]: ").strip().lower()
        approved = answer in {"s", "si", "sí", "y", "yes"}
        if not approved:
            output(f"Tool rechazada: {tool.name}")
        return approved

    return confirm


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
        settings = AgentSettings()
        client = OpenAILLMClient()
        registry = build_default_registry()
        run_chat(client, registry, settings)
    except LLMClientError as error:
        print(f"Error de configuración del LLM: {error}")
    except Exception as error:
        print(f"Error al iniciar el agente: {error}")


if __name__ == "__main__":
    main()
