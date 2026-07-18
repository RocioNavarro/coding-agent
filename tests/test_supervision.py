"""Tests del modo de supervisión de tools."""

from typing import Any

import pytest

from core.settings import AgentSettings
from core.supervision import SupervisedToolExecutor
from tools.command_tools import COMMAND_TIMEOUT_SECONDS
from tools.definitions import ToolDefinition
from tools.registry import ToolRegistry


EMPTY_PARAMETERS = {
    "type": "object",
    "properties": {},
    "required": [],
    "additionalProperties": False,
}


def build_registry(calls: list[str]) -> ToolRegistry:
    """Crea las cuatro tools con ejecutores observables y sin efectos reales."""
    registry = ToolRegistry()
    for name, modifies_system in (
        ("read_file", False),
        ("write_file", True),
        ("list_files", False),
        ("run_command", True),
    ):
        registry.register(
            ToolDefinition(
                name=name,
                description=f"Tool simulada {name}.",
                parameters=EMPTY_PARAMETERS,
                executor=lambda tool_name=name: calls.append(tool_name) or tool_name,
                modifies_system=modifies_system,
            )
        )
    return registry


def test_settings_defaults() -> None:
    settings = AgentSettings()

    assert settings.supervision_enabled is True
    assert settings.plan_mode_enabled is True
    assert settings.max_iterations == 20
    assert settings.command_timeout_seconds == 60
    assert COMMAND_TIMEOUT_SECONDS == 60


@pytest.mark.parametrize("name", ["write_file", "run_command"])
def test_modifying_tools_request_confirmation(name: str) -> None:
    calls: list[str] = []
    confirmations: list[str] = []
    executor = SupervisedToolExecutor(
        build_registry(calls),
        AgentSettings(),
        lambda tool, arguments: confirmations.append(tool.name) or True,
    )

    result = executor.execute(name, {})

    assert result["success"] is True
    assert confirmations == [name]
    assert calls == [name]


@pytest.mark.parametrize("name", ["read_file", "list_files"])
def test_read_only_tools_do_not_request_confirmation(name: str) -> None:
    calls: list[str] = []

    def unexpected_confirmation(tool: ToolDefinition, arguments: dict[str, Any]) -> bool:
        raise AssertionError("No se debía solicitar confirmación")

    executor = SupervisedToolExecutor(
        build_registry(calls), AgentSettings(), unexpected_confirmation
    )

    result = executor.execute(name, {})

    assert result["success"] is True
    assert calls == [name]


def test_disabled_supervision_executes_without_confirmation() -> None:
    calls: list[str] = []

    def unexpected_confirmation(tool: ToolDefinition, arguments: dict[str, Any]) -> bool:
        raise AssertionError("No se debía solicitar confirmación")

    executor = SupervisedToolExecutor(
        build_registry(calls),
        AgentSettings(supervision_enabled=False),
        unexpected_confirmation,
    )

    result = executor.execute("write_file", {})

    assert result["success"] is True
    assert calls == ["write_file"]


def test_rejection_returns_controlled_result_without_execution() -> None:
    calls: list[str] = []
    executor = SupervisedToolExecutor(
        build_registry(calls), AgentSettings(), lambda tool, arguments: False
    )

    result = executor.execute("run_command", {})

    assert result == {
        "success": False,
        "result": None,
        "error": "Ejecución rechazada por el usuario: run_command.",
    }
    assert calls == []


def test_missing_confirmation_callback_denies_execution() -> None:
    calls: list[str] = []
    executor = SupervisedToolExecutor(build_registry(calls), AgentSettings())

    result = executor.execute("write_file", {})

    assert result["success"] is False
    assert result["error"] == "La tool 'write_file' requiere confirmación."
    assert calls == []


def test_confirmation_exception_returns_controlled_result() -> None:
    calls: list[str] = []

    def fail(tool: ToolDefinition, arguments: dict[str, Any]) -> bool:
        raise RuntimeError("entrada cerrada")

    executor = SupervisedToolExecutor(build_registry(calls), AgentSettings(), fail)

    result = executor.execute("run_command", {})

    assert result["success"] is False
    assert result["error"] == (
        "Error al solicitar confirmación para 'run_command': entrada cerrada"
    )
    assert calls == []


def test_invalid_arguments_do_not_request_confirmation() -> None:
    calls: list[str] = []
    confirmations: list[str] = []
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="write_file",
            description="Escritura simulada.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
            executor=lambda path: calls.append(path),
            modifies_system=True,
        )
    )
    executor = SupervisedToolExecutor(
        registry,
        AgentSettings(),
        lambda tool, arguments: confirmations.append(tool.name) or True,
    )

    result = executor.execute("write_file", {})

    assert result["success"] is False
    assert "path" in result["error"]  # type: ignore[operator]
    assert confirmations == []
    assert calls == []


def test_unknown_tool_returns_controlled_result() -> None:
    executor = SupervisedToolExecutor(ToolRegistry(), AgentSettings())

    result = executor.execute("missing", {})

    assert result["success"] is False
    assert result["error"] == "La tool 'missing' no está registrada."
