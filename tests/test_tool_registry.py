"""Tests del registro genérico de tools."""

from typing import Any

import pytest

from tools.definitions import DuplicateToolError, ToolDefinition, ToolValidationError
from tools.registry import TOOL_REGISTRY, ToolRegistry, build_default_registry


def make_tool(name: str = "demo", executor: Any = lambda: "ok") -> ToolDefinition:
    """Crea una definición mínima para tests unitarios."""
    return ToolDefinition(
        name=name,
        description="Tool de prueba.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        executor=executor,
        modifies_system=False,
    )


def test_registers_and_gets_tool() -> None:
    registry = ToolRegistry()
    tool = make_tool()

    registry.register(tool)

    assert registry.get("demo") is tool
    assert registry.get("missing") is None


def test_rejects_duplicate_name() -> None:
    registry = ToolRegistry()
    registry.register(make_tool())

    with pytest.raises(DuplicateToolError, match="ya está registrada"):
        registry.register(make_tool())


def test_default_registry_contains_only_expected_tools() -> None:
    registry = build_default_registry()

    assert {schema["function"]["name"] for schema in registry.list_schemas()} == {
        "read_file",
        "write_file",
        "list_files",
        "run_command",
    }
    assert registry.get("web_search") is None


def test_default_tools_mark_system_modification() -> None:
    assert TOOL_REGISTRY.get("read_file").modifies_system is False  # type: ignore[union-attr]
    assert TOOL_REGISTRY.get("list_files").modifies_system is False  # type: ignore[union-attr]
    assert TOOL_REGISTRY.get("write_file").modifies_system is True  # type: ignore[union-attr]
    assert TOOL_REGISTRY.get("run_command").modifies_system is True  # type: ignore[union-attr]


def test_schema_does_not_expose_executor_or_metadata() -> None:
    schema = TOOL_REGISTRY.list_schemas()[0]

    assert set(schema) == {"type", "function"}
    assert set(schema["function"]) == {"name", "description", "parameters"}
    assert "executor" not in schema["function"]
    assert "modifies_system" not in schema["function"]


def test_validation_rejects_missing_required_argument() -> None:
    with pytest.raises(ToolValidationError, match="path"):
        TOOL_REGISTRY.validate_arguments("read_file", {})


def test_validation_rejects_wrong_type() -> None:
    with pytest.raises(ToolValidationError, match="string"):
        TOOL_REGISTRY.validate_arguments("read_file", {"path": 123})


def test_validation_rejects_unknown_argument() -> None:
    with pytest.raises(ToolValidationError, match="desconocidos"):
        TOOL_REGISTRY.validate_arguments("list_files", {"extra": True})


def test_validation_rejects_unknown_tool() -> None:
    with pytest.raises(ToolValidationError, match="no está registrada"):
        TOOL_REGISTRY.validate_arguments("missing", {})


def test_execute_returns_controlled_validation_error() -> None:
    result = TOOL_REGISTRY.execute("read_file", {})

    assert result["success"] is False
    assert result["result"] is None
    assert "path" in result["error"]  # type: ignore[operator]


def test_execute_calls_executor_with_validated_arguments() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo",
            description="Devuelve un texto.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
            executor=lambda text: text.upper(),
            modifies_system=False,
        )
    )

    assert registry.execute("echo", {"text": "hola"}) == {
        "success": True,
        "result": "HOLA",
        "error": None,
    }


def test_execute_applies_list_files_default() -> None:
    registry = ToolRegistry()
    received: dict[str, str] = {}
    tool = TOOL_REGISTRY.get("list_files")
    assert tool is not None
    registry.register(
        ToolDefinition(
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
            executor=lambda path: received.setdefault("path", path),
            modifies_system=tool.modifies_system,
        )
    )

    result = registry.execute("list_files", {})

    assert result["success"] is True
    assert received == {"path": "."}


def test_execute_controls_executor_exception() -> None:
    def fail() -> None:
        raise RuntimeError("fallo esperado")

    registry = ToolRegistry()
    registry.register(make_tool(executor=fail))

    result = registry.execute("demo", {})

    assert result["success"] is False
    assert result["result"] is None
    assert result["error"] == "Error al ejecutar la tool 'demo': fallo esperado"
