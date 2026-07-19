"""Política central para toda ejecución de tools."""

from pathlib import Path

import pytest

from core.config import load_agent_config
from core.settings import AgentSettings
from core.supervision import SupervisedToolExecutor
from security.policy_engine import (
    AgentToolPermissions,
    PolicyContext,
    PolicyEngine,
)
from tools.definitions import ToolDefinition
from tools.registry import ToolRegistry, build_default_registry


def config_for(tmp_path: Path, body: str):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    path = tmp_path / "agent.config.yaml"
    path.write_text(f"workspace:\n  path: workspace\n{body}", encoding="utf-8")
    return load_agent_config(path)


def context(
    tmp_path: Path,
    *,
    allowed_tools: frozenset[str] | None = None,
    approval_tools: frozenset[str] = frozenset(),
    config=None,
) -> PolicyContext:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return PolicyContext(
        agent="tester",
        workspace=workspace,
        permissions=AgentToolPermissions(allowed_tools, approval_tools),
        config=config,
        settings=AgentSettings(),
    )


@pytest.mark.parametrize(
    ("tool", "parameters", "outcome"),
    [
        ("read_file", {"path": "src/app.py"}, "allow"),
        ("list_files", {"path": "."}, "allow"),
        ("write_file", {"path": "src/app.py", "content": "x"}, "require_approval"),
        ("run_command", {"command": "python tests.py"}, "require_approval"),
    ],
)
def test_base_policy_returns_three_explicit_outcomes(
    tmp_path: Path, tool: str, parameters: dict[str, object], outcome: str
) -> None:
    decision = PolicyEngine().evaluate(tool, parameters, context(tmp_path))

    assert decision.outcome == outcome
    assert decision.agent == "tester"
    assert decision.tool == tool


@pytest.mark.parametrize(
    "path",
    ["../secret.txt", "/etc/passwd", "nested/../../secret.txt", ".env", "config/secrets.json"],
)
def test_denies_traversal_outside_and_sensitive_paths(tmp_path: Path, path: str) -> None:
    decision = PolicyEngine().evaluate("read_file", {"path": path}, context(tmp_path))

    assert decision.outcome == "deny"
    assert "ruta" in decision.reason.casefold() or "sensible" in decision.reason.casefold()


@pytest.mark.parametrize(
    "command",
    ["rm -rf .", "git push", "git reset --hard", "cat ../secret", "python -c 'print(1)'"],
)
def test_denies_destructive_or_escaping_commands(tmp_path: Path, command: str) -> None:
    decision = PolicyEngine().evaluate(
        "run_command", {"command": command}, context(tmp_path)
    )

    assert decision.outcome == "deny"


def test_denies_tools_not_authorized_for_subagent(tmp_path: Path) -> None:
    policy_context = context(
        tmp_path, allowed_tools=frozenset({"read_file", "list_files"})
    )

    decision = PolicyEngine().evaluate(
        "write_file", {"path": "file.txt", "content": "x"}, policy_context
    )

    assert decision.outcome == "deny"
    assert "tester" in decision.reason


def test_agent_permissions_can_require_approval_for_read_tool(tmp_path: Path) -> None:
    policy_context = context(
        tmp_path,
        allowed_tools=frozenset({"read_file"}),
        approval_tools=frozenset({"read_file"}),
    )

    decision = PolicyEngine().evaluate(
        "read_file", {"path": "file.txt"}, policy_context
    )

    assert decision.outcome == "require_approval"


def test_agent_config_permissions_restrict_tools_and_web_limits(tmp_path: Path) -> None:
    config = config_for(
        tmp_path,
        """permissions:
  read: true
  write: false
  run_commands: false
  web_search: true
web_search:
  enabled: true
  max_results: 2
""",
    )
    policy_context = context(tmp_path, config=config)

    assert PolicyEngine().evaluate(
        "write_file", {"path": "file.txt", "content": "x"}, policy_context
    ).outcome == "deny"
    assert PolicyEngine().evaluate(
        "run_command", {"command": "python tests.py"}, policy_context
    ).outcome == "deny"
    web = PolicyEngine().evaluate(
        "web_search", {"query": "topic", "max_results": 3}, policy_context
    )
    assert web.outcome == "deny"
    assert "max_results" in web.reason


def test_configured_commands_act_as_allowlist(tmp_path: Path) -> None:
    config = config_for(
        tmp_path,
        """permissions:
  run_commands: true
commands:
  test: python -m pytest
""",
    )
    policy_context = context(tmp_path, config=config)

    assert PolicyEngine().evaluate(
        "run_command", {"command": "python -m pytest"}, policy_context
    ).outcome == "require_approval"
    denied = PolicyEngine().evaluate(
        "run_command", {"command": "python other.py"}, policy_context
    )
    assert denied.outcome == "deny"
    assert "configurados" in denied.reason


def test_symlink_escape_is_denied(tmp_path: Path) -> None:
    policy_context = context(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (policy_context.workspace / "link.txt").symlink_to(outside)

    decision = PolicyEngine().evaluate(
        "read_file", {"path": "link.txt"}, policy_context
    )

    assert decision.outcome == "deny"


def test_executor_never_calls_tool_after_policy_denial(tmp_path: Path) -> None:
    calls: list[str] = []
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            "read_file", "fake",
            {
                "type": "object", "properties": {"path": {"type": "string"}},
                "required": ["path"], "additionalProperties": False,
            },
            lambda path: calls.append(path),
            False,
        )
    )
    executor = SupervisedToolExecutor(
        registry,
        AgentSettings(),
        policy_context=context(tmp_path),
    )

    result = executor.execute("read_file", {"path": "../secret.txt"})

    assert result["success"] is False
    assert calls == []


def test_executor_automatically_uses_agent_config_permissions(tmp_path: Path) -> None:
    config = config_for(
        tmp_path,
        """permissions:
  read: true
  write: false
""",
    )
    calls: list[str] = []
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            "write_file", "fake",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            lambda path, content: calls.append(path),
            True,
        )
    )
    executor = SupervisedToolExecutor(
        registry,
        AgentSettings(agent_config=config),
        lambda tool, arguments: True,
    )

    result = executor.execute(
        "write_file", {"path": "file.txt", "content": "content"}
    )

    assert result["success"] is False
    assert "agent.config.yaml" in str(result["error"])
    assert calls == []


def test_unknown_tool_and_invalid_parameters_are_denied(tmp_path: Path) -> None:
    policy_context = context(tmp_path)
    engine = PolicyEngine(known_tools=frozenset({"read_file"}))

    assert engine.evaluate("missing", {}, policy_context).outcome == "deny"
    assert engine.evaluate("read_file", {"path": 123}, policy_context).outcome == "deny"


def test_default_registry_tools_all_pass_through_policy(tmp_path: Path) -> None:
    registry = build_default_registry()
    engine = PolicyEngine(known_tools=frozenset(tool["function"]["name"] for tool in registry.list_schemas()))

    assert engine.evaluate("list_files", {"path": "."}, context(tmp_path)).outcome == "allow"
