"""Motor central y genérico de autorización para tool calls."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from core.config import AgentConfig
from core.settings import AgentSettings
from security.command_policy import CommandPolicyError, SENSITIVE_FILE_NAMES, validate_command


PolicyOutcome = Literal["allow", "deny", "require_approval"]
_PATH_KEYS = frozenset({"path", "directory", "cwd", "file", "filename"})


@dataclass(frozen=True)
class AgentToolPermissions:
    """Restricciones adicionales de un subagente; nunca amplían la configuración."""

    allowed_tools: frozenset[str] | None = None
    approval_tools: frozenset[str] = frozenset()


@dataclass(frozen=True)
class PolicyContext:
    agent: str
    workspace: Path
    permissions: AgentToolPermissions = AgentToolPermissions()
    config: AgentConfig | None = None
    settings: AgentSettings | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.agent, str) or not self.agent.strip():
            raise ValueError("agent no puede estar vacío.")
        workspace = Path(self.workspace).resolve()
        if not workspace.is_dir():
            raise ValueError("workspace debe ser un directorio existente.")
        object.__setattr__(self, "agent", self.agent.strip())
        object.__setattr__(self, "workspace", workspace)


@dataclass(frozen=True)
class PolicyDecision:
    outcome: PolicyOutcome
    reason: str
    agent: str
    tool: str


class PolicyEngine:
    """Combina políticas base, configuración y permisos del agente."""

    def __init__(self, *, known_tools: frozenset[str] | None = None) -> None:
        self.known_tools = known_tools

    def evaluate(
        self,
        tool: str,
        parameters: object,
        context: PolicyContext,
        *,
        modifies_system: bool | None = None,
    ) -> PolicyDecision:
        if not isinstance(tool, str) or not tool.strip():
            return self._decision("deny", "La tool no tiene un nombre válido.", context, str(tool))
        name = tool.strip()
        if self.known_tools is not None and name not in self.known_tools:
            return self._decision("deny", f"La tool '{name}' no está registrada.", context, name)
        if not isinstance(parameters, Mapping) or not all(
            isinstance(key, str) for key in parameters
        ):
            return self._decision("deny", "Los parámetros deben ser un objeto.", context, name)
        values = dict(parameters)

        allowed = context.permissions.allowed_tools
        if allowed is not None and name not in allowed:
            return self._decision(
                "deny",
                f"La tool '{name}' no está autorizada para el agente '{context.agent}'.",
                context,
                name,
            )
        configuration_error = self._configuration_policy(name, values, context)
        if configuration_error:
            return self._decision("deny", configuration_error, context, name)
        path_error = self._path_policy(values, context.workspace)
        if path_error:
            return self._decision("deny", path_error, context, name)
        if name == "run_command" and "command" in values:
            command_error = self._command_policy(values, context)
            if command_error:
                return self._decision("deny", command_error, context, name)

        modifies = modifies_system if modifies_system is not None else name in {
            "write_file", "run_command"
        }
        supervision = context.settings.supervision_enabled if context.settings else True
        if name in context.permissions.approval_tools or (modifies and supervision):
            return self._decision(
                "require_approval",
                f"La tool '{name}' requiere aprobación antes de ejecutarse.",
                context,
                name,
            )
        return self._decision("allow", "La tool call cumple las políticas.", context, name)

    def _configuration_policy(
        self, tool: str, parameters: dict[str, Any], context: PolicyContext
    ) -> str | None:
        config = context.config
        if config is None:
            return None
        if context.workspace != config.workspace.path.resolve():
            return "El workspace de ejecución no coincide con agent.config.yaml."
        permission = {
            "read_file": config.permissions.read,
            "list_files": config.permissions.read,
            "write_file": config.permissions.write,
            "run_command": config.permissions.run_commands,
            "web_search": config.permissions.web_search and config.web_search.enabled,
        }.get(tool, True)
        if not permission:
            return f"La tool '{tool}' está deshabilitada por agent.config.yaml."
        if tool == "web_search":
            maximum = parameters.get("max_results", config.web_search.max_results)
            if isinstance(maximum, bool) or not isinstance(maximum, int):
                return "web_search.max_results debe ser un entero."
            if maximum > config.web_search.max_results:
                return (
                    "web_search.max_results supera el máximo configurado "
                    f"({config.web_search.max_results})."
                )
        return None

    def _path_policy(self, parameters: dict[str, Any], workspace: Path) -> str | None:
        for key, raw in parameters.items():
            normalized_key = key.casefold()
            if not (
                normalized_key in _PATH_KEYS
                or normalized_key.endswith("_path")
                or normalized_key.endswith("_file")
                or normalized_key.endswith("_directory")
            ):
                continue
            if not isinstance(raw, str) or not raw.strip():
                return f"El parámetro de ruta '{key}' debe ser texto no vacío."
            candidate = Path(raw)
            if candidate.is_absolute():
                return f"La ruta '{raw}' está fuera del workspace."
            resolved = (workspace / candidate).resolve()
            try:
                resolved.relative_to(workspace)
            except ValueError:
                return f"La ruta '{raw}' está fuera del workspace."
            sensitive = {item.casefold() for item in SENSITIVE_FILE_NAMES}
            if any(
                part.casefold() in sensitive
                or part.casefold().startswith(".env")
                or "secret" in part.casefold()
                or "credential" in part.casefold()
                for part in candidate.parts
            ):
                return f"El acceso al archivo sensible '{raw}' no está permitido."
        return None

    def _command_policy(
        self, parameters: dict[str, Any], context: PolicyContext
    ) -> str | None:
        command = parameters.get("command")
        if not isinstance(command, str) or not command.strip():
            return "El parámetro command debe ser texto no vacío."
        try:
            arguments = shlex.split(command)
            validate_command(arguments, context.workspace)
        except (ValueError, CommandPolicyError) as error:
            return str(error)
        if context.config is not None and context.config.commands:
            configured = {
                context.config.commands[key].strip()
                for key in context.config.commands
            }
            if command.strip() not in configured:
                return "El comando no pertenece a los comandos configurados."
        return None

    @staticmethod
    def _decision(
        outcome: PolicyOutcome,
        reason: str,
        context: PolicyContext,
        tool: str,
    ) -> PolicyDecision:
        return PolicyDecision(outcome, reason, context.agent, tool)
