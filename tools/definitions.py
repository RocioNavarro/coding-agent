"""Tipos compartidos por el registro genérico de tools."""

from dataclasses import dataclass
from typing import Any, Callable, TypedDict


JSONSchema = dict[str, Any]
ToolExecutor = Callable[..., Any]


class ToolRegistryError(ValueError):
    """Error base para operaciones inválidas sobre el registro."""


class DuplicateToolError(ToolRegistryError):
    """Indica que se intentó registrar un nombre existente."""


class ToolValidationError(ToolRegistryError):
    """Indica que los argumentos no cumplen el schema de una tool."""


class ToolExecutionResult(TypedDict):
    """Resultado controlado producido por el registro al ejecutar una tool."""

    success: bool
    result: Any
    error: str | None


@dataclass(frozen=True)
class ToolDefinition:
    """Describe una tool, su contrato público y su función ejecutora."""

    name: str
    description: str
    parameters: JSONSchema
    executor: ToolExecutor
    modifies_system: bool

    def to_llm_schema(self) -> dict[str, Any]:
        """Devuelve el formato de tool calling que consume un LLM."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
