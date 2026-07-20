"""Tests del Tester genérico con repositorios y executors simulados."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from agents.tester import (
    CommandOutcome,
    StaticCommandProvider,
    TaskStateCommandProvider,
    TesterAgent as AgentTester,
    TesterLimits as ValidationLimits,
    ValidationCommand,
    ValidationExecutor,
)
from core.models import LLMResponse, LLMUsage, Message
from core.task_state import TaskState
from core.observability import ObservabilityEvent
from core.progress import ProgressLimits, ProgressMonitor
from core.settings import AgentSettings
from security.policy_engine import AgentToolPermissions, PolicyContext


class UnusedLLM:
    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] = (),
    ) -> LLMResponse:
        raise AssertionError("Tester no debe usar el LLM para inventar comandos.")


class FakeExecutor(ValidationExecutor):
    def __init__(self, outcomes: dict[str, CommandOutcome]) -> None:
        self.outcomes = outcomes
        self.calls: list[tuple[str, float]] = []

    def execute(self, command: str, *, timeout_seconds: float) -> CommandOutcome:
        self.calls.append((command, timeout_seconds))
        return self.outcomes[command]


@pytest.fixture()
def python_repository(tmp_path: Path) -> Path:
    root = tmp_path / "python"
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src/service.py").write_text("VALUE = 2\n", encoding="utf-8")
    (root / "tests/test_service.py").write_text("def test_value(): pass\n", encoding="utf-8")
    return root


@pytest.fixture()
def javascript_repository(tmp_path: Path) -> Path:
    root = tmp_path / "javascript"
    root.mkdir()
    (root / "index.js").write_text("module.exports = 2;\n", encoding="utf-8")
    (root / "package.json").write_text('{"scripts":{"test":"jest"}}\n', encoding="utf-8")
    return root


def changed_state(*files: str) -> TaskState:
    state = TaskState.create("Validar cambios", task_id="tester-task")
    state.add_repository_finding("language detectado con evidencia del repositorio")
    for path in files:
        state.record_file_modified(path)
    return state


def build_tester(
    root: Path,
    commands: Sequence[ValidationCommand],
    executor: FakeExecutor,
    *,
    limits: ValidationLimits | None = None,
    progress_monitor: ProgressMonitor | None = None,
    observability=None,
    policy_context: PolicyContext | None = None,
) -> AgentTester:
    return AgentTester(
        llm_client=UnusedLLM(),
        repository_root=root,
        providers=(StaticCommandProvider(commands),),
        executor=executor,
        limits=limits,
        progress_monitor=progress_monitor,
        observability=observability,
        policy_context=policy_context,
    )


class RecordingObservability:
    def __init__(self, *, fail: bool = False) -> None:
        self.events: list[ObservabilityEvent] = []
        self.fail = fail

    def record(self, event: ObservabilityEvent) -> None:
        if self.fail:
            raise RuntimeError("provider unavailable")
        self.events.append(event)

    def flush(self) -> None:
        return None


class SequentialExecutor(ValidationExecutor):
    def __init__(self, outcomes: Sequence[CommandOutcome]) -> None:
        self.outcomes = iter(outcomes)
        self.calls: list[tuple[str, float]] = []

    def execute(self, command: str, *, timeout_seconds: float) -> CommandOutcome:
        self.calls.append((command, timeout_seconds))
        return next(self.outcomes)


def test_prioritizes_specific_check_before_global_validation(
    python_repository: Path,
) -> None:
    specific = ValidationCommand(
        "python -m pytest tests/test_service.py",
        "configuration",
        "pyproject.toml",
        "test",
        scope_files=("src/service.py",),
        priority=50,
    )
    global_check = ValidationCommand(
        "python -m pytest", "configuration", "pyproject.toml", "test", priority=1
    )
    executor = FakeExecutor(
        {
            specific.command: CommandOutcome(0, 12.0, "1 passed"),
            global_check.command: CommandOutcome(0, 30.0, "10 passed"),
        }
    )
    tester = build_tester(python_repository, (global_check, specific), executor)

    result = tester.run("Validar service", changed_state("src/service.py"))

    assert result.status == "passed"
    assert [record.command for record in result.records] == [
        specific.command, global_check.command
    ]
    assert [call[0] for call in executor.calls] == [specific.command, global_check.command]


def test_executes_javascript_command_successfully_and_records_task_state(
    javascript_repository: Path,
) -> None:
    command = ValidationCommand(
        "npm test", "script", "package.json scripts.test", "test", priority=10
    )
    executor = FakeExecutor(
        {"npm test": CommandOutcome(0, 25.5, "PASS index.test.js\n")}
    )
    state = changed_state("index.js")
    tester = build_tester(javascript_repository, (command,), executor)

    result = tester.run("Validar JavaScript", state)

    assert result.status == "passed"
    assert result.records[0].origin == "script"
    assert result.records[0].exit_code == 0
    assert result.records[0].duration_ms == 25.5
    assert result.records[0].output == "PASS index.test.js\n"
    assert state.commands_executed == ("npm test",)
    assert state.tool_calls[-1].result["status"] == "passed"
    assert state.subagent_results[-1].status == "passed"


def test_reports_failed_command(python_repository: Path) -> None:
    command = ValidationCommand(
        "python -m pytest", "configuration", "pyproject.toml", "test"
    )
    executor = FakeExecutor(
        {command.command: CommandOutcome(1, 18.0, "", "AssertionError")}
    )
    tester = build_tester(python_repository, (command,), executor)

    result = tester.run("Validar", changed_state("src/service.py"))

    assert result.status == "failed"
    assert result.records[0].status == "failed"
    assert result.records[0].exit_code == 1
    assert "AssertionError" in result.records[0].output


def test_registers_equivalent_failures_in_progress_monitor(
    python_repository: Path,
) -> None:
    command = ValidationCommand(
        "python -m pytest", "configuration", "pyproject.toml", "test"
    )
    executor = SequentialExecutor(
        (
            CommandOutcome(1, 1.0, stderr="same failure"),
            CommandOutcome(1, 1.0, stderr="same failure"),
        )
    )
    progress = ProgressMonitor(ProgressLimits(command_error_repeats=2))
    tester = AgentTester(
        llm_client=UnusedLLM(), repository_root=python_repository,
        providers=(StaticCommandProvider((command,)),), executor=executor,
        limits=ValidationLimits(max_retries=1), progress_monitor=progress,
    )

    result = tester.run("Validar", changed_state("src/service.py"))

    assert result.records[0].attempts == 2
    detected = [item for item in result.progress_assessments if item.detected]
    assert detected[-1].recommendation == "retry_with_new_strategy"


def test_changed_command_result_is_progress(python_repository: Path) -> None:
    command = ValidationCommand(
        "python -m pytest", "configuration", "pyproject.toml", "test"
    )
    executor = SequentialExecutor(
        (
            CommandOutcome(1, 1.0, stderr="first failure"),
            CommandOutcome(0, 1.0, stdout="passed"),
        )
    )
    progress = ProgressMonitor(ProgressLimits(command_error_repeats=2))
    tester = AgentTester(
        llm_client=UnusedLLM(), repository_root=python_repository,
        providers=(StaticCommandProvider((command,)),), executor=executor,
        limits=ValidationLimits(max_retries=1), progress_monitor=progress,
    )

    result = tester.run("Validar", changed_state("src/service.py"))

    assert result.status == "passed"
    assert not any(item.detected for item in result.progress_assessments)


def test_repeated_attempts_without_new_evidence_are_detected(
    python_repository: Path,
) -> None:
    command = ValidationCommand(
        "python -m pytest", "configuration", "pyproject.toml", "test"
    )
    failure = CommandOutcome(1, 1.0, stderr="unchanged failure")
    executor = SequentialExecutor((failure, failure, failure))
    progress = ProgressMonitor(
        ProgressLimits(command_error_repeats=99, no_evidence_iterations=2)
    )
    tester = AgentTester(
        llm_client=UnusedLLM(), repository_root=python_repository,
        providers=(StaticCommandProvider((command,)),), executor=executor,
        limits=ValidationLimits(max_retries=2), progress_monitor=progress,
    )

    result = tester.run("Validar", changed_state("src/service.py"))

    detected = [item for item in result.progress_assessments if item.detected]
    assert detected[-1].kind == "no_new_evidence"
    assert detected[-1].recommendation == "stop"


@pytest.mark.parametrize(
    ("command", "supervision", "expected"),
    [
        ("python -m pytest", False, "allow"),
        ("rm -rf .", False, "deny"),
        ("python -m pytest", True, "require_approval"),
    ],
)
def test_policy_decision_emits_one_observation(
    python_repository: Path, command: str, supervision: bool, expected: str
) -> None:
    validation = ValidationCommand(
        command, "configuration", "repository configuration", "test"
    )
    executor = FakeExecutor(
        {command: CommandOutcome(0, 1.0, stdout="passed")}
        if expected == "allow" else {}
    )
    observed = RecordingObservability()
    policy_context = PolicyContext(
        agent="tester", workspace=python_repository,
        settings=AgentSettings(supervision_enabled=supervision),
        permissions=AgentToolPermissions(
            approval_tools=(frozenset({"run_command"}) if supervision else frozenset())
        ),
    )
    tester = build_tester(
        python_repository, (validation,), executor,
        observability=observed, policy_context=policy_context,
    )

    result = tester.run("Validar", changed_state("src/service.py"))

    events = [event for event in observed.events if event.name == "tester-policy-decision"]
    assert len(events) == 1
    assert events[0].payload["decision"] == expected
    assert events[0].payload["approval_required"] is (expected == "require_approval")
    assert result.status == ("passed" if expected == "allow" else "blocked")


def test_observability_failure_does_not_interrupt_tester(
    python_repository: Path,
) -> None:
    command = ValidationCommand(
        "python -m pytest", "configuration", "pyproject.toml", "test"
    )
    executor = FakeExecutor(
        {command.command: CommandOutcome(0, 1.0, stdout="passed")}
    )
    tester = build_tester(
        python_repository, (command,), executor,
        observability=RecordingObservability(fail=True),
    )

    assert tester.run("Validar", changed_state("src/service.py")).status == "passed"


def test_reports_timeout_and_respects_configured_limit(python_repository: Path) -> None:
    command = ValidationCommand(
        "python -m pytest", "configuration", "pyproject.toml", "test"
    )
    executor = FakeExecutor(
        {command.command: CommandOutcome(None, 10.0, stderr="timeout", timed_out=True)}
    )
    limits = ValidationLimits(
        timeout_seconds=0.01,
        max_commands=1,
        max_output_chars=20,
        max_retries=3,
    )
    tester = build_tester(python_repository, (command,), executor, limits=limits)

    result = tester.run("Validar con límite", changed_state("src/service.py"))

    assert result.status == "failed"
    assert result.records[0].status == "failed"
    assert result.records[0].attempts == 1
    assert executor.calls == [(command.command, 0.01)]


def test_reports_unavailable_without_reliable_commands(python_repository: Path) -> None:
    executor = FakeExecutor({})
    tester = build_tester(python_repository, (), executor)
    state = changed_state("src/service.py")

    result = tester.run("Validar", state)

    assert result.status == "unavailable"
    assert result.records == ()
    assert executor.calls == []
    assert state.commands_executed == ()
    assert "No se encontraron comandos" in result.summary


def test_blocks_prohibited_install_command(javascript_repository: Path) -> None:
    command = ValidationCommand(
        "npm install", "documentation", "README.md", "setup"
    )
    executor = FakeExecutor({})
    tester = build_tester(javascript_repository, (command,), executor)
    state = changed_state("index.js")

    result = tester.run("Validar sin instalar", state)

    assert result.status == "blocked"
    assert result.records[0].status == "blocked"
    assert "instalación" in result.records[0].output
    assert executor.calls == []
    assert state.commands_executed == ()
    assert state.tool_calls[-1].result["status"] == "blocked"


def test_recovers_explorer_command_and_preserves_its_origin(
    python_repository: Path,
) -> None:
    state = changed_state("src/service.py")
    state.add_observation(
        "Comando detectado: python -m pytest; evidencia: pyproject.toml."
    )
    executor = FakeExecutor(
        {"python -m pytest": CommandOutcome(0, 8.0, "passed")}
    )
    tester = AgentTester(
        llm_client=UnusedLLM(),
        repository_root=python_repository,
        providers=(TaskStateCommandProvider(),),
        executor=executor,
    )

    result = tester.run("Validar desde Explorer", state)

    assert result.status == "passed"
    assert result.records[0].origin == "configuration"
    assert result.records[0].evidence == "pyproject.toml"
