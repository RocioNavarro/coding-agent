"""Tests del loop interno con un LLM falso y tools sin efectos externos."""

from collections.abc import Sequence
from typing import Any

import pytest

from core.harness import MaxIterationsError, run_internal_loop
from core.models import LLMResponse, LLMUsage, Message, ToolCall
from core.settings import AgentSettings
from tools.definitions import ToolDefinition
from tools.registry import ToolRegistry


class FakeLLMClient:
    """Devuelve respuestas predeterminadas y conserva snapshots del historial."""

    def __init__(self, responses: Sequence[LLMResponse]) -> None:
        self._responses = iter(responses)
        self.histories: list[list[Message]] = []
        self.schemas: list[list[dict[str, Any]]] = []

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] = (),
    ) -> LLMResponse:
        self.histories.append(list(messages))
        self.schemas.append(list(tools))
        return next(self._responses)


def llm_response(
    text: str = "", tool_calls: list[ToolCall] | None = None
) -> LLMResponse:
    """Crea una respuesta interna concisa para los escenarios del harness."""
    calls = tool_calls or []
    return LLMResponse(
        assistant_message=Message(role="assistant", content=text, tool_calls=calls),
        text=text,
        tool_calls=calls,
        model="fake-model",
        usage=LLMUsage(1, 1, 2),
        latency_ms=1.0,
    )


def make_registry(
    events: list[tuple[str, dict[str, Any]]], *, command_error: bool = False
) -> ToolRegistry:
    """Crea write_file y run_command observables, sin tocar disco ni procesos."""
    registry = ToolRegistry()

    def write_file(path: str, content: str) -> str:
        events.append(("write_file", {"path": path, "content": content}))
        return "archivo escrito"

    def run_command(command: str) -> dict[str, Any]:
        events.append(("run_command", {"command": command}))
        if command_error:
            raise RuntimeError("proceso fallido")
        return {"exit_code": 0, "stdout": "Hello\n", "stderr": ""}

    registry.register(
        ToolDefinition(
            name="write_file",
            description="Escribe un archivo.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            executor=write_file,
            modifies_system=True,
        )
    )
    registry.register(
        ToolDefinition(
            name="run_command",
            description="Ejecuta un comando.",
            parameters={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
            executor=run_command,
            modifies_system=True,
        )
    )
    return registry


def test_two_successive_tool_calls_then_final_response() -> None:
    """El LLM pide escribir, luego ejecutar y finalmente responde sin tools."""
    write_call = ToolCall(
        "call-write", "write_file", {"path": "hello.py", "content": "print('Hello')\n"}
    )
    run_call = ToolCall("call-run", "run_command", {"command": "python hello.py"})
    fake_llm = FakeLLMClient(
        [
            llm_response(tool_calls=[write_call]),
            llm_response(tool_calls=[run_call]),
            llm_response(text="Creé y ejecuté hello.py correctamente."),
        ]
    )
    events: list[tuple[str, dict[str, Any]]] = []
    history = [Message(role="user", content="Creá y ejecutá hello.py")]
    approvals: list[str] = []
    log: list[str] = []

    result = run_internal_loop(
        fake_llm,
        make_registry(events),
        AgentSettings(max_iterations=5),
        history,
        lambda tool, arguments: approvals.append(tool.name) or True,
        log.append,
    )

    assert result.iterations == 3
    assert result.response.text == "Creé y ejecuté hello.py correctamente."
    assert [name for name, _ in events] == ["write_file", "run_command"]
    assert approvals == ["write_file", "run_command"]
    assert [message.role for message in history] == [
        "user", "assistant", "tool", "assistant", "tool", "assistant"
    ]
    assert fake_llm.histories[1][-1].tool_call_id == "call-write"
    assert fake_llm.histories[2][-1].tool_call_id == "call-run"
    assert log[0] == "--- Iteración 1 ---"
    assert "Tool: write_file" in log
    assert "Tool: run_command" in log
    assert log[-1] == "--- Iteración 3 ---"


def test_rejected_write_file_is_added_to_history() -> None:
    call = ToolCall("call-write", "write_file", {"path": "hello.py", "content": "x"})
    fake_llm = FakeLLMClient([llm_response(tool_calls=[call]), llm_response("Cancelado")])
    events: list[tuple[str, dict[str, Any]]] = []
    history = [Message(role="user", content="Creá hello.py")]

    result = run_internal_loop(
        fake_llm, make_registry(events), AgentSettings(), history,
        lambda tool, arguments: False, lambda text: None,
    )

    assert result.response.text == "Cancelado"
    assert events == []
    assert "rechazada" in history[-2].content


@pytest.mark.parametrize(
    ("call", "expected_error"),
    [
        (ToolCall("missing", "unknown", {}), "no está registrada"),
        (ToolCall("invalid", "write_file", {"path": "hello.py"}), "Faltan argumentos"),
    ],
)
def test_tool_errors_are_returned_to_llm(
    call: ToolCall, expected_error: str
) -> None:
    fake_llm = FakeLLMClient([llm_response(tool_calls=[call]), llm_response("Listo")])
    history = [Message(role="user", content="Intentá")]

    result = run_internal_loop(
        fake_llm, make_registry([]), AgentSettings(), history,
        lambda tool, arguments: True, lambda text: None,
    )

    assert result.iterations == 2
    assert expected_error in history[-2].content


def test_max_iterations_raises_and_preserves_history() -> None:
    call = ToolCall("call-1", "unknown", {})
    fake_llm = FakeLLMClient([llm_response(tool_calls=[call])] * 2)
    history = [Message(role="user", content="No termines")]

    with pytest.raises(MaxIterationsError, match="máximo de 2"):
        run_internal_loop(
            fake_llm, make_registry([]), AgentSettings(max_iterations=2), history,
            lambda tool, arguments: True, lambda text: None,
        )

    assert len(history) == 5
    assert history[-1].role == "tool"


def test_executor_error_is_returned_to_llm() -> None:
    call = ToolCall("call-run", "run_command", {"command": "python hello.py"})
    fake_llm = FakeLLMClient([llm_response(tool_calls=[call]), llm_response("Falló")])
    history = [Message(role="user", content="Ejecutá")]

    result = run_internal_loop(
        fake_llm, make_registry([], command_error=True), AgentSettings(), history,
        lambda tool, arguments: True, lambda text: None,
    )

    assert result.response.text == "Falló"
    assert "proceso fallido" in history[-2].content


def test_visible_log_redacts_secrets_and_truncates_large_content() -> None:
    call = ToolCall(
        "call-write", "write_file",
        {"path": "hello.py", "content": "x" * 1000, "api_key": "forbidden"},
    )
    fake_llm = FakeLLMClient([llm_response(tool_calls=[call]), llm_response("Listo")])
    log: list[str] = []

    run_internal_loop(
        fake_llm, make_registry([]), AgentSettings(),
        [Message(role="user", content="Intentá")],
        lambda tool, arguments: True, log.append,
    )

    rendered = "\n".join(log)
    assert "forbidden" not in rendered
    assert "[truncado]" in rendered
