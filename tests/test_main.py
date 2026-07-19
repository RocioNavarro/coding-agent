"""Tests del chat interactivo sin stdin, red ni API real."""

from collections.abc import Sequence
from unittest.mock import Mock, patch

import pytest

import main as main_module
from core.models import LLMResponse, LLMUsage, Message
from core.settings import AgentSettings
from main import CommandResult, process_command, run_chat
from tools.registry import ToolRegistry


class FakeLLMClient:
    """LLM determinista que registra cada snapshot recibido."""

    def __init__(self, texts: list[str]) -> None:
        self._texts = iter(texts)
        self.histories: list[list[Message]] = []

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, object]] = (),
    ) -> LLMResponse:
        self.histories.append(list(messages))
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
        AgentSettings(supervision_enabled=False),
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
        client, ToolRegistry(), AgentSettings(),
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
        FakeLLMClient([]), ToolRegistry(), AgentSettings(),
        input_func=scripted_input([exception]), output=output.append,
    )

    assert [message.role for message in history] == ["system"]
    assert expected in output[-1]


def test_loop_error_is_shown_without_traceback_and_chat_continues() -> None:
    client = Mock()
    client.complete.side_effect = RuntimeError("fallo controlado")
    output: list[str] = []

    history = run_chat(
        client, ToolRegistry(), AgentSettings(),
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
