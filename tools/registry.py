"""Registro, validación y ejecución controlada de tools."""

from typing import Any

from tools.command_tools import run_command
from tools.definitions import (
    DuplicateToolError,
    JSONSchema,
    ToolDefinition,
    ToolExecutionResult,
    ToolValidationError,
)
from tools.file_tools import list_files, read_file, write_file
from tools.web_tools import web_search


JSON_TYPES: dict[str, type[Any] | tuple[type[Any], ...]] = {
    "array": list,
    "boolean": bool,
    "integer": int,
    "null": type(None),
    "number": (int, float),
    "object": dict,
    "string": str,
}


class ToolRegistry:
    """Mantiene definiciones de tools y centraliza su ejecución segura."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        """Registra una tool y rechaza nombres duplicados."""
        if tool.name in self._tools:
            raise DuplicateToolError(f"La tool '{tool.name}' ya está registrada.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition | None:
        """Obtiene una tool por nombre o devuelve ``None`` si no existe."""
        return self._tools.get(name)

    def list_schemas(self) -> list[dict[str, Any]]:
        """Lista los schemas públicos listos para enviarse al LLM."""
        return [tool.to_llm_schema() for tool in self._tools.values()]

    def validate_arguments(self, name: str, arguments: object) -> dict[str, Any]:
        """Valida argumentos y aplica defaults definidos por el JSON Schema."""
        tool = self.get(name)
        if tool is None:
            raise ToolValidationError(f"La tool '{name}' no está registrada.")
        if not isinstance(arguments, dict):
            raise ToolValidationError("Los argumentos deben ser un objeto JSON.")

        schema = tool.parameters
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        unknown = set(arguments) - set(properties)
        if schema.get("additionalProperties") is False and unknown:
            names = ", ".join(sorted(unknown))
            raise ToolValidationError(f"Argumentos desconocidos: {names}.")

        missing = [key for key in required if key not in arguments]
        if missing:
            names = ", ".join(missing)
            raise ToolValidationError(f"Faltan argumentos requeridos: {names}.")

        validated = dict(arguments)
        for key, property_schema in properties.items():
            if key not in validated and "default" in property_schema:
                validated[key] = property_schema["default"]
            if key in validated:
                self._validate_value(key, validated[key], property_schema)

        return validated

    def execute(self, name: str, arguments: object) -> ToolExecutionResult:
        """Valida y ejecuta una tool, devolviendo siempre un resultado controlado."""
        try:
            tool = self.get(name)
            if tool is None:
                raise ToolValidationError(f"La tool '{name}' no está registrada.")
            validated = self.validate_arguments(name, arguments)
            result = tool.executor(**validated)
            return {"success": True, "result": result, "error": None}
        except ToolValidationError as error:
            return {"success": False, "result": None, "error": str(error)}
        except Exception as error:  # El límite del registro controla cada executor.
            return {
                "success": False,
                "result": None,
                "error": f"Error al ejecutar la tool '{name}': {error}",
            }

    @staticmethod
    def _validate_value(name: str, value: Any, schema: JSONSchema) -> None:
        """Valida un valor contra los tipos JSON Schema soportados."""
        expected_name = schema.get("type")
        if expected_name is None:
            return

        expected_type = JSON_TYPES.get(expected_name)
        if expected_type is None:
            raise ToolValidationError(
                f"Tipo JSON Schema no soportado para '{name}': {expected_name}."
            )

        is_valid = isinstance(value, expected_type)
        if expected_name in {"integer", "number"} and isinstance(value, bool):
            is_valid = False
        if not is_valid:
            raise ToolValidationError(
                f"El argumento '{name}' debe ser de tipo {expected_name}."
            )


def _parameters(
    properties: dict[str, JSONSchema], required: list[str] | None = None
) -> JSONSchema:
    """Construye el schema de parámetros común a las tools locales."""
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def build_default_registry() -> ToolRegistry:
    """Crea el registro estándar de tools locales de la Parte 1."""
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="read_file",
            description="Lee como UTF-8 un archivo relativo al workspace.",
            parameters=_parameters(
                {"path": {"type": "string", "description": "Ruta del archivo."}},
                required=["path"],
            ),
            executor=read_file,
            modifies_system=False,
        )
    )
    registry.register(
        ToolDefinition(
            name="write_file",
            description="Reemplaza el contenido de un archivo del workspace.",
            parameters=_parameters(
                {
                    "path": {"type": "string", "description": "Ruta del archivo."},
                    "content": {
                        "type": "string",
                        "description": "Contenido UTF-8 que se escribirá.",
                    },
                },
                required=["path", "content"],
            ),
            executor=write_file,
            modifies_system=True,
        )
    )
    registry.register(
        ToolDefinition(
            name="list_files",
            description="Lista archivos y directorios dentro del workspace.",
            parameters=_parameters(
                {
                    "path": {
                        "type": "string",
                        "description": "Directorio que se listará.",
                        "default": ".",
                    }
                }
            ),
            executor=list_files,
            modifies_system=False,
        )
    )
    registry.register(
        ToolDefinition(
            name="run_command",
            description="Ejecuta un comando controlado desde el workspace.",
            parameters=_parameters(
                {
                    "command": {
                        "type": "string",
                        "description": "Comando y argumentos que se ejecutarán.",
                    }
                },
                required=["command"],
            ),
            executor=run_command,
            modifies_system=True,
        )
    )
    registry.register(
        ToolDefinition(
            name="web_search",
            description="Busca información actual en la web mediante Tavily.",
            parameters=_parameters(
                {
                    "query": {
                        "type": "string",
                        "description": "Consulta concreta que se buscará.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Cantidad de resultados, entre 1 y 10.",
                        "default": 5,
                    },
                },
                required=["query"],
            ),
            executor=web_search,
            modifies_system=False,
        )
    )
    return registry


TOOL_REGISTRY = build_default_registry()
