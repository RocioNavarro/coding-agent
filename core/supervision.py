"""Aprobación de tools que pueden modificar el sistema."""

from typing import Any, Callable

from core.settings import AgentSettings
from tools.definitions import ToolDefinition, ToolExecutionResult, ToolValidationError
from tools.registry import ToolRegistry


ConfirmationCallback = Callable[[ToolDefinition, dict[str, Any]], bool]


class SupervisedToolExecutor:
    """Ejecuta tools y solicita aprobación cuando la configuración lo exige."""

    def __init__(
        self,
        registry: ToolRegistry,
        settings: AgentSettings,
        confirm: ConfirmationCallback | None = None,
    ) -> None:
        self._registry = registry
        self._settings = settings
        self._confirm = confirm

    def requires_confirmation(self, tool: ToolDefinition) -> bool:
        """Indica si una tool debe aprobarse antes de su ejecución."""
        return self._settings.supervision_enabled and tool.modifies_system

    def execute(self, name: str, arguments: object) -> ToolExecutionResult:
        """Valida, solicita aprobación cuando corresponde y ejecuta una tool."""
        tool = self._registry.get(name)
        if tool is None:
            return self._registry.execute(name, arguments)

        try:
            validated = self._registry.validate_arguments(name, arguments)
        except ToolValidationError as error:
            return {"success": False, "result": None, "error": str(error)}

        if self.requires_confirmation(tool):
            approval_error = self._request_approval(tool, validated)
            if approval_error is not None:
                return {"success": False, "result": None, "error": approval_error}

        return self._registry.execute(name, validated)

    def _request_approval(
        self, tool: ToolDefinition, arguments: dict[str, Any]
    ) -> str | None:
        """Devuelve un mensaje controlado si una ejecución no fue aprobada."""
        if self._confirm is None:
            return f"La tool '{tool.name}' requiere confirmación."

        try:
            approved = self._confirm(tool, arguments)
        except Exception as error:
            return f"Error al solicitar confirmación para '{tool.name}': {error}"

        if not approved:
            return f"Ejecución rechazada por el usuario: {tool.name}."

        return None
