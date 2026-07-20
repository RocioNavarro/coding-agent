"""Aprobación de tools que pueden modificar el sistema."""

from typing import Any, Callable
from time import perf_counter

from core.observability import NoOpObservabilityClient, ObservabilityClient, ObservabilityEvent, emit_observation
from core.settings import AgentSettings
from security.paths import WORKSPACE_ROOT
from security.policy_engine import PolicyContext, PolicyEngine
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
        *,
        policy_engine: PolicyEngine | None = None,
        policy_context: PolicyContext | None = None,
        observability: ObservabilityClient | None = None,
        task_id: str | None = None,
        parent_event_id: str | None = None,
    ) -> None:
        self._registry = registry
        self._settings = settings
        self._confirm = confirm
        self._policy_engine = policy_engine or PolicyEngine()
        self._observability = observability or NoOpObservabilityClient()
        self._task_id = task_id
        self._parent_event_id = parent_event_id
        configured = settings.agent_config
        self._policy_context = policy_context or PolicyContext(
            agent="main",
            workspace=configured.workspace.path if configured else WORKSPACE_ROOT,
            config=configured,
            settings=settings,
        )

    def requires_confirmation(self, tool: ToolDefinition) -> bool:
        """Indica si una tool debe aprobarse antes de su ejecución."""
        return self._settings.supervision_enabled and tool.modifies_system

    def execute(self, name: str, arguments: object) -> ToolExecutionResult:
        """Valida, solicita aprobación cuando corresponde y ejecuta una tool."""
        started = perf_counter()
        tool = self._registry.get(name)
        if tool is None:
            return self._registry.execute(name, arguments)

        try:
            validated = self._registry.validate_arguments(name, arguments)
        except ToolValidationError as error:
            return {"success": False, "result": None, "error": str(error)}

        decision = self._policy_engine.evaluate(
            name,
            validated,
            self._policy_context,
            modifies_system=tool.modifies_system,
        )
        emit_observation(
            self._observability,
            ObservabilityEvent(
                "tool", "policy-decision", task_id=self._task_id,
                parent_event_id=self._parent_event_id,
                agent=self._policy_context.agent,
                payload={"tool_name": name, "parameters": validated,
                         "decision": decision.outcome, "reason": decision.reason,
                         "required_approval": decision.outcome == "require_approval"},
            ),
        )
        if decision.outcome == "deny":
            result = {"success": False, "result": None, "error": decision.reason}
            self._record_execution(name, validated, decision.outcome, result, started)
            return result

        if decision.outcome == "require_approval":
            approval_error = self._request_approval(tool, validated)
            if approval_error is not None:
                result = {"success": False, "result": None, "error": approval_error}
                self._record_execution(name, validated, decision.outcome, result, started)
                return result

        result = self._registry.execute(name, validated)
        self._record_execution(name, validated, decision.outcome, result, started)
        return result

    def _record_execution(
        self, name: str, arguments: dict[str, Any], decision: str,
        result: ToolExecutionResult, started: float,
    ) -> None:
        payload = result.get("result")
        exit_code = payload.get("exit_code") if isinstance(payload, dict) else None
        emit_observation(
            self._observability,
            ObservabilityEvent(
                "tool", name, task_id=self._task_id,
                parent_event_id=self._parent_event_id,
                agent=self._policy_context.agent,
                payload={"tool_name": name, "parameters": arguments,
                         "decision": decision, "result": payload,
                         "exit_code": exit_code, "error": result.get("error")},
                latency_ms=(perf_counter() - started) * 1000,
            ),
        )

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
