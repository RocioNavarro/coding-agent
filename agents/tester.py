"""Tester genérico: selecciona evidencia de validación y ejecuta checks acotados."""

from __future__ import annotations

import re
import shlex
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal, Sequence

from agents.base import AgentContext, BaseAgent
from core.llm_client import LLMClient
from core.task_state import SubagentResult, TaskState, ToolExecutionRecord
from security.command_policy import CommandPolicyError, validate_command
from tools.registry import ToolRegistry


ValidationStatus = Literal["passed", "failed", "skipped", "blocked", "unavailable"]
CommandOrigin = Literal[
    "script", "configuration", "documentation", "project_memory",
    "agent_configuration", "explorer"
]


@dataclass(frozen=True)
class TesterLimits:
    timeout_seconds: float = 60.0
    max_commands: int = 5
    max_output_chars: int = 8_000
    max_retries: int = 0

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds debe ser positivo.")
        if self.max_commands <= 0:
            raise ValueError("max_commands debe ser positivo.")
        if self.max_output_chars <= 0:
            raise ValueError("max_output_chars debe ser positivo.")
        if self.max_retries < 0:
            raise ValueError("max_retries no puede ser negativo.")


@dataclass(frozen=True)
class ValidationCommand:
    command: str
    origin: CommandOrigin
    evidence: str
    check_type: str = "validation"
    scope_files: tuple[str, ...] = ()
    priority: int = 100

    def __post_init__(self) -> None:
        for name in ("command", "evidence", "check_type"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} no puede estar vacío.")
            object.__setattr__(self, name, value.strip())
        if not all(isinstance(path, str) and path.strip() for path in self.scope_files):
            raise ValueError("scope_files debe contener rutas no vacías.")
        object.__setattr__(self, "scope_files", tuple(self.scope_files))


@dataclass(frozen=True)
class CommandOutcome:
    exit_code: int | None
    duration_ms: float
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    unavailable: bool = False


@dataclass(frozen=True)
class ValidationRecord:
    command: str
    origin: CommandOrigin
    evidence: str
    check_type: str
    exit_code: int | None
    duration_ms: float
    output: str
    status: ValidationStatus
    attempts: int


@dataclass(frozen=True)
class TesterResult:
    status: ValidationStatus
    records: tuple[ValidationRecord, ...]
    modified_files: tuple[str, ...]
    summary: str
    subagent_result: SubagentResult


class ValidationCommandProvider(ABC):
    """Fuente desacoplada de comandos respaldados por evidencia."""

    @abstractmethod
    def get_commands(
        self, task_state: TaskState, modified_files: Sequence[str]
    ) -> Sequence[ValidationCommand]:
        """Devuelve comandos existentes; nunca debe inventarlos."""


_EXPLORER_COMMAND_RE = re.compile(
    r"Comando detectado:\s*(?P<command>.*?);\s*evidencia:\s*(?P<evidence>.*?)\.?$",
    re.IGNORECASE,
)


class TaskStateCommandProvider(ValidationCommandProvider):
    """Recupera comandos que Explorer registró con su evidencia concreta."""

    def get_commands(
        self, task_state: TaskState, modified_files: Sequence[str]
    ) -> Sequence[ValidationCommand]:
        commands: list[ValidationCommand] = []
        for observation in task_state.observations:
            match = _EXPLORER_COMMAND_RE.search(observation)
            if match is None:
                continue
            evidence = match.group("evidence").strip()
            commands.append(
                ValidationCommand(
                    command=match.group("command").strip(),
                    origin=self._origin_from_evidence(evidence),
                    evidence=evidence,
                    priority=50,
                )
            )
        return commands

    @staticmethod
    def _origin_from_evidence(evidence: str) -> CommandOrigin:
        lowered = evidence.casefold()
        if any(token in lowered for token in ("readme", ".md", ".rst", ".adoc")):
            return "documentation"
        if any(token in lowered for token in ("scripts/", "bin/", ".sh", ".ps1", ".bat")):
            return "script"
        return "configuration"


class StaticCommandProvider(ValidationCommandProvider):
    """Proveedor útil para configuración del agente, memoria y tests."""

    def __init__(self, commands: Sequence[ValidationCommand]) -> None:
        self._commands = tuple(commands)

    def get_commands(
        self, task_state: TaskState, modified_files: Sequence[str]
    ) -> Sequence[ValidationCommand]:
        return self._commands


class ValidationExecutor(ABC):
    @abstractmethod
    def execute(self, command: str, *, timeout_seconds: float) -> CommandOutcome:
        """Ejecuta una única validación sin shell."""


class SubprocessValidationExecutor(ValidationExecutor):
    """Executor real confinado al repositorio y sin interpretación de shell."""

    def __init__(self, repository_root: str | Path) -> None:
        root = Path(repository_root).resolve()
        if not root.is_dir():
            raise ValueError("repository_root debe ser un directorio existente.")
        self._root = root

    def execute(self, command: str, *, timeout_seconds: float) -> CommandOutcome:
        arguments = shlex.split(command)
        started = perf_counter()
        try:
            completed = subprocess.run(
                arguments,
                cwd=self._root,
                shell=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                check=False,
            )
            return CommandOutcome(
                completed.returncode,
                (perf_counter() - started) * 1000,
                completed.stdout,
                completed.stderr,
            )
        except subprocess.TimeoutExpired as error:
            return CommandOutcome(
                None,
                (perf_counter() - started) * 1000,
                self._text(error.stdout),
                self._text(error.stderr),
                timed_out=True,
            )
        except (FileNotFoundError, OSError) as error:
            return CommandOutcome(
                None,
                (perf_counter() - started) * 1000,
                stderr=str(error),
                unavailable=True,
            )

    @staticmethod
    def _text(value: str | bytes | None) -> str:
        if value is None:
            return ""
        return value if isinstance(value, str) else value.decode(errors="replace")


class ValidationSafetyPolicy:
    """Bloquea instalaciones, VCS mutador y comandos fuera del workspace."""

    _FORBIDDEN_ACTIONS = frozenset(
        {"add", "commit", "install", "publish", "push", "remove", "uninstall",
         "update", "upgrade"}
    )
    _SHELL_TOKENS = frozenset({"&&", "||", ";", "|", ">", ">>", "<"})

    def __init__(self, repository_root: str | Path) -> None:
        self._root = Path(repository_root).resolve()

    def validate(self, command: str) -> tuple[bool, str]:
        try:
            arguments = shlex.split(command)
        except ValueError as error:
            return False, f"Comando inválido: {error}"
        if not arguments:
            return False, "El comando está vacío."
        lowered = {argument.casefold() for argument in arguments[1:]}
        if lowered & self._FORBIDDEN_ACTIONS:
            return False, "El comando solicita una operación mutadora o instalación."
        if any(argument in self._SHELL_TOKENS for argument in arguments):
            return False, "No se permiten operadores de shell."
        try:
            validate_command(arguments, self._root)
        except CommandPolicyError as error:
            return False, str(error)
        return True, "Comando permitido."


class TesterAgent(BaseAgent):
    """Selecciona checks proporcionales, los ejecuta una vez y registra evidencia."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        repository_root: str | Path,
        providers: Sequence[ValidationCommandProvider],
        executor: ValidationExecutor | None = None,
        limits: TesterLimits | None = None,
        safety_policy: ValidationSafetyPolicy | None = None,
        name: str = "tester",
    ) -> None:
        super().__init__(
            name=name,
            role="Generic Change Validator",
            system_prompt="Validá sólo con comandos respaldados por evidencia.",
            allowed_tools=(),
            llm_client=llm_client,
        )
        self.providers = tuple(providers)
        self.executor = executor or SubprocessValidationExecutor(repository_root)
        self.limits = limits or TesterLimits()
        self.safety_policy = safety_policy or ValidationSafetyPolicy(repository_root)

    def specialization_prompt(self) -> str:
        return "No generes ni ejecutes comandos desde el LLM."

    def run(
        self,
        instruction: str,
        task_state: TaskState,
        context: AgentContext | None = None,
        available_tools: ToolRegistry | None = None,
    ) -> TesterResult:
        modified = task_state.files_modified
        if not modified:
            return self._finish(
                task_state, "skipped", (), modified,
                "No hay archivos modificados que validar.", instruction
            )
        candidates = tuple(
            command
            for provider in self.providers
            for command in provider.get_commands(task_state, modified)
        )
        selected = self.select_commands(candidates, modified)
        if not selected:
            return self._finish(
                task_state, "unavailable", (), modified,
                "No se encontraron comandos de validación respaldados por evidencia.",
                instruction,
            )

        records = tuple(self._execute(command, task_state, index) for index, command in enumerate(selected, 1))
        status = self._overall_status(records)
        summary = self._summary(status, records)
        return self._finish(task_state, status, records, modified, summary, instruction)

    def select_commands(
        self,
        commands: Sequence[ValidationCommand],
        modified_files: Sequence[str],
    ) -> tuple[ValidationCommand, ...]:
        """Deduplica y antepone checks cuyo alcance intersecta el cambio."""
        unique: dict[str, ValidationCommand] = {}
        for command in commands:
            current = unique.get(command.command)
            if current is None or command.priority < current.priority:
                unique[command.command] = command
        modified = set(modified_files)
        ordered = sorted(
            unique.values(),
            key=lambda item: (
                0 if item.scope_files and modified.intersection(item.scope_files) else 1,
                item.priority,
                item.command,
            ),
        )
        return tuple(ordered[: self.limits.max_commands])

    def _execute(
        self, command: ValidationCommand, state: TaskState, index: int
    ) -> ValidationRecord:
        allowed, reason = self.safety_policy.validate(command.command)
        if not allowed:
            record = ValidationRecord(
                command.command, command.origin, command.evidence, command.check_type,
                None, 0.0, reason, "blocked", 0
            )
            self._record_state(state, record, index)
            return record

        outcome: CommandOutcome | None = None
        attempts = 0
        for attempts in range(1, self.limits.max_retries + 2):
            outcome = self.executor.execute(
                command.command, timeout_seconds=self.limits.timeout_seconds
            )
            if outcome.exit_code == 0 or outcome.timed_out or outcome.unavailable:
                break
        assert outcome is not None
        if outcome.unavailable:
            status: ValidationStatus = "unavailable"
        elif outcome.timed_out or outcome.exit_code != 0:
            status = "failed"
        else:
            status = "passed"
        output = self._truncate("\n".join(part for part in (outcome.stdout, outcome.stderr) if part))
        record = ValidationRecord(
            command.command, command.origin, command.evidence, command.check_type,
            outcome.exit_code, outcome.duration_ms, output, status, attempts
        )
        state.record_command(command.command)
        self._record_state(state, record, index)
        return record

    def _record_state(self, state: TaskState, record: ValidationRecord, index: int) -> None:
        state.record_tool_call(
            ToolExecutionRecord(
                f"tester-{index}",
                "validation_command",
                {"command": record.command, "origin": record.origin},
                record.status == "passed",
                {
                    "exit_code": record.exit_code,
                    "duration_ms": record.duration_ms,
                    "output": record.output,
                    "status": record.status,
                    "attempts": record.attempts,
                    "evidence": record.evidence,
                },
                None if record.status == "passed" else record.output,
            )
        )

    def _truncate(self, output: str) -> str:
        if len(output) <= self.limits.max_output_chars:
            return output
        return output[: self.limits.max_output_chars] + "… [truncado]"

    @staticmethod
    def _overall_status(records: Sequence[ValidationRecord]) -> ValidationStatus:
        statuses = {record.status for record in records}
        if "blocked" in statuses:
            return "blocked"
        if "failed" in statuses:
            return "failed"
        if statuses == {"unavailable"}:
            return "unavailable"
        if "unavailable" in statuses:
            return "blocked"
        if statuses == {"passed"}:
            return "passed"
        return "unavailable"

    @staticmethod
    def _summary(status: ValidationStatus, records: Sequence[ValidationRecord]) -> str:
        return (
            f"Validación {status}: {sum(record.status == 'passed' for record in records)} "
            f"de {len(records)} checks pasaron."
        )

    def _finish(
        self,
        state: TaskState,
        status: ValidationStatus,
        records: Sequence[ValidationRecord],
        modified: Sequence[str],
        summary: str,
        instruction: str,
    ) -> TesterResult:
        result = SubagentResult(
            self.name,
            instruction,
            status,
            result=summary,
            summary=summary,
            findings=tuple(
                f"{record.check_type}: {record.command} -> {record.status}"
                for record in records
            ),
            files_relevant=tuple(modified),
            blockers=(summary,) if status in {"blocked", "unavailable"} else (),
            confidence=1.0 if status in {"passed", "failed"} else 0.0,
        )
        state.add_subagent_result(result)
        state.add_observation(summary)
        return TesterResult(status, tuple(records), tuple(modified), summary, result)
