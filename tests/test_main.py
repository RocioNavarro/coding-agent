"""Tests del chat interactivo sin stdin, red ni API real."""

from collections.abc import Sequence
from unittest.mock import Mock, patch

import pytest

import main as main_module
from core.models import LLMResponse, LLMUsage, Message
from core.settings import AgentSettings
from main import CommandResult, process_command, run_chat
from tools.registry import ToolRegistry
from tools.definitions import ToolDefinition


class FakeLLMClient:
    """LLM determinista que registra cada snapshot recibido."""

    def __init__(self, texts: list[str]) -> None:
        self._texts = iter(texts)
        self.histories: list[list[Message]] = []
        self.schemas: list[list[dict[str, object]]] = []

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, object]] = (),
    ) -> LLMResponse:
        self.histories.append(list(messages))
        self.schemas.append(list(tools))
        text = next(self._texts)
        assistant = Message(role="assistant", content=text)
        return LLMResponse(
            assistant_message=assistant,
            text=text,
            tool_calls=[],
            model="fake-model",
            usage=LLMUsage(1, 1, 2),
            latency_ms=1.0,
        )


def scripted_input(values: list[object]):
    """Crea una función input que devuelve valores o lanza excepciones."""
    iterator = iter(values)

    def read(prompt: str) -> str:
        value = next(iterator)
        if isinstance(value, BaseException):
            raise value
        assert isinstance(value, str)
        return value

    return read


def test_chat_keeps_history_across_multiple_turns() -> None:
    client = FakeLLMClient(["Primera respuesta", "Segunda respuesta"])
    output: list[str] = []

    history = run_chat(
        client,
        ToolRegistry(),
        AgentSettings(supervision_enabled=False, plan_mode_enabled=False),
        input_func=scripted_input(["Primer mensaje", "Segundo mensaje", "/exit"]),
        output=output.append,
    )

    assert [message.role for message in history] == [
        "system", "user", "assistant", "user", "assistant"
    ]
    assert client.histories[0][0].role == "system"
    assert [message.content for message in client.histories[1][-3:]] == [
        "Primer mensaje", "Primera respuesta", "Segundo mensaje"
    ]
    assert "Asistente: Primera respuesta" in output
    assert "Asistente: Segunda respuesta" in output
    assert output.count("Iteraciones: 1") == 2


def test_empty_messages_are_ignored() -> None:
    client = FakeLLMClient(["Respuesta"])

    history = run_chat(
        client, ToolRegistry(), AgentSettings(plan_mode_enabled=False),
        input_func=scripted_input(["", "   ", "Hola", "/exit"]),
        output=lambda text: None,
    )

    assert len(client.histories) == 1
    assert [message.role for message in history] == ["system", "user", "assistant"]


@pytest.mark.parametrize(
    ("command", "attribute", "expected"),
    [
        ("/plan on", "plan_mode_enabled", True),
        ("/plan off", "plan_mode_enabled", False),
        ("/supervision on", "supervision_enabled", True),
        ("/supervision off", "supervision_enabled", False),
    ],
)
def test_toggle_commands(
    command: str, attribute: str, expected: bool
) -> None:
    settings = AgentSettings()
    output: list[str] = []

    result = process_command(command, settings, output.append)

    assert result == CommandResult(handled=True)
    assert getattr(settings, attribute) is expected
    assert output


def test_status_displays_all_settings() -> None:
    output: list[str] = []

    result = process_command("/status", AgentSettings(), output.append)

    assert result.handled is True
    assert "Plan mode: on" in output[0]
    assert "Supervision mode: on" in output[0]
    assert "Max iterations: 20" in output[0]
    assert "Command timeout: 60s" in output[0]


def test_exit_and_unknown_command_are_handled_locally() -> None:
    assert process_command("/exit", AgentSettings()) == CommandResult(True, True)
    output: list[str] = []
    assert process_command("/unknown", AgentSettings(), output.append).handled is True
    assert "desconocido" in output[0]


@pytest.mark.parametrize(
    ("exception", "expected"),
    [(KeyboardInterrupt(), "Interrumpido"), (EOFError(), "Fin de entrada")],
)
def test_chat_handles_terminal_endings(exception: BaseException, expected: str) -> None:
    output: list[str] = []

    history = run_chat(
        FakeLLMClient([]), ToolRegistry(), AgentSettings(plan_mode_enabled=False),
        input_func=scripted_input([exception]), output=output.append,
    )

    assert [message.role for message in history] == ["system"]
    assert expected in output[-1]


def test_loop_error_is_shown_without_traceback_and_chat_continues() -> None:
    client = Mock()
    client.complete.side_effect = RuntimeError("fallo controlado")
    output: list[str] = []

    history = run_chat(
        client, ToolRegistry(), AgentSettings(plan_mode_enabled=False),
        input_func=scripted_input(["Hola", "/exit"]), output=output.append,
    )

    assert history[-1] == Message(role="user", content="Hola")
    assert "Error: fallo controlado" in output
    assert not any("Traceback" in line for line in output)


def test_main_builds_dependencies_and_starts_chat() -> None:
    fake_client = Mock()
    fake_registry = Mock()
    with (
        patch.object(main_module, "OpenAILLMClient", return_value=fake_client),
        patch.object(main_module, "build_default_registry", return_value=fake_registry),
        patch.object(main_module, "run_chat") as chat,
    ):
        main_module.main()

    chat.assert_called_once()
    args = chat.call_args.args
    assert args[0] is fake_client
    assert args[1] is fake_registry
    assert isinstance(args[2], AgentSettings)


def registry_with_schema() -> ToolRegistry:
    """Crea una tool observable para distinguir planificación de ejecución."""
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="inspect",
            description="Inspecciona.",
            parameters={
                "type": "object", "properties": {}, "required": [],
                "additionalProperties": False,
            },
            executor=lambda: "ok",
            modifies_system=False,
        )
    )
    return registry


def test_plan_approval_happens_before_tools_are_exposed() -> None:
    client = FakeLLMClient(["1. Inspeccionar\n2. Resolver", "Trabajo terminado"])
    output: list[str] = []

    history = run_chat(
        client, registry_with_schema(), AgentSettings(),
        input_func=scripted_input(["Resolvé la tarea", "a", "/exit"]),
        output=output.append,
    )

    assert client.schemas[0] == []
    assert client.schemas[1] != []
    assert [message.role for message in history] == [
        "system", "user", "assistant", "developer", "assistant"
    ]
    assert history[2].content == "1. Inspeccionar\n2. Resolver"
    assert "Plan propuesto:" in output[0]
    assert "Asistente: Trabajo terminado" in output


def test_plan_rejection_cancels_only_current_task() -> None:
    client = FakeLLMClient([
        "1. Plan rechazable", "1. Segundo plan", "Segunda tarea terminada"
    ])

    history = run_chat(
        client, registry_with_schema(), AgentSettings(),
        input_func=scripted_input([
            "Primera tarea", "r", "Segunda tarea", "a", "/exit"
        ]),
        output=lambda text: None,
    )

    assert client.schemas == [[], [], client.schemas[2]]
    assert client.schemas[2] != []
    assert "cancelada" in history[2].content
    assert [message.content for message in history if message.role == "user"] == [
        "Primera tarea", "Segunda tarea"
    ]


def test_plan_modification_regenerates_without_becoming_user_request() -> None:
    client = FakeLLMClient([
        "1. Plan inicial", "1. Plan con tests", "Implementación terminada"
    ])

    history = run_chat(
        client, registry_with_schema(), AgentSettings(),
        input_func=scripted_input([
            "Implementá", "m", "Agregá tests primero", "a", "/exit"
        ]),
        output=lambda text: None,
    )

    assert client.schemas[0] == []
    assert client.schemas[1] == []
    assert client.schemas[2] != []
    assert any(
        message.role == "developer" and "Agregá tests primero" in message.content
        for message in client.histories[1]
    )
    assert [message.content for message in history if message.role == "user"] == [
        "Implementá"
    ]
    assert "1. Plan con tests" in [message.content for message in history]
