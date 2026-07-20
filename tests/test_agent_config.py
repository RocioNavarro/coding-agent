"""Carga y validación del manifiesto genérico del agente."""

from pathlib import Path

import pytest

from core.config import AgentConfigError, load_agent_config


@pytest.fixture(autouse=True)
def clear_workspace_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Aísla los tests de una configuración local exportada por el usuario."""
    for variable in (
        "CODING_AGENT_WORKSPACE",
        "CODING_AGENT_RAG_INDEX_PATH",
        "CODING_AGENT_MEMORY_PATH",
    ):
        monkeypatch.delenv(variable, raising=False)


def write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "agent.config.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_minimal_config_with_defaults_and_without_technology(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    config = load_agent_config(write_config(tmp_path, "workspace:\n  path: project\n"))

    assert config.workspace.path == workspace.resolve()
    assert config.project is None
    assert config.permissions.read is True
    assert config.permissions.write is False
    assert dict(config.commands) == {}
    assert config.limits.max_iterations == 20
    assert config.rag.enabled is False
    assert config.rag.sources == ()
    assert config.memory.enabled is True
    assert config.observability.log_level == "INFO"
    assert config.web_search.enabled is False
    assert config.web_search.allowed_domains == ()


def test_workspace_environment_override_has_priority_over_yaml(tmp_path: Path) -> None:
    yaml_workspace = tmp_path / "yaml-project"
    yaml_workspace.mkdir()
    environment_workspace = tmp_path / "environment-project"
    environment_workspace.mkdir()

    config = load_agent_config(
        write_config(tmp_path, "workspace:\n  path: yaml-project\n"),
        environ={"CODING_AGENT_WORKSPACE": str(environment_workspace)},
    )

    assert config.workspace.path == environment_workspace.resolve()


def test_workspace_environment_override_expands_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))

    config = load_agent_config(
        write_config(tmp_path, "workspace:\n  path: missing\n"),
        environ={"CODING_AGENT_WORKSPACE": "~/project"},
    )

    assert config.workspace.path == workspace.resolve()


def test_workspace_yaml_expands_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))

    config = load_agent_config(
        write_config(tmp_path, "workspace:\n  path: ~/project\n"),
        environ={},
    )

    assert config.workspace.path == workspace.resolve()


def test_workspace_environment_override_rejects_missing_directory(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(
        AgentConfigError, match="CODING_AGENT_WORKSPACE no existe o no es un directorio"
    ):
        load_agent_config(
            write_config(tmp_path, "workspace:\n  path: project\n"),
            environ={"CODING_AGENT_WORKSPACE": str(missing)},
        )


def test_workspace_environment_override_must_be_absolute(tmp_path: Path) -> None:
    with pytest.raises(AgentConfigError, match="debe ser una ruta absoluta"):
        load_agent_config(
            write_config(tmp_path, "workspace:\n  path: project\n"),
            environ={"CODING_AGENT_WORKSPACE": "relative/project"},
        )


def test_rag_and_memory_environment_overrides_have_priority_over_yaml(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    config = load_agent_config(
        write_config(
            tmp_path,
            """
workspace:
  path: project
rag:
  index_path: yaml/index.json
memory:
  path: yaml/memory
""",
        ),
        environ={
            "CODING_AGENT_RAG_INDEX_PATH": ".coding-agent/rag/index.json",
            "CODING_AGENT_MEMORY_PATH": ".coding-agent/memory",
        },
    )

    assert config.rag.index_path == (workspace / ".coding-agent/rag/index.json").resolve()
    assert config.memory.path == (workspace / ".coding-agent/memory").resolve()


def test_rag_and_memory_defaults_resolve_inside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()

    config = load_agent_config(
        write_config(tmp_path, "workspace:\n  path: project\n"), environ={}
    )

    assert config.rag.index_path == (workspace / ".coding-agent/rag/index.json").resolve()
    assert config.memory.path == (workspace / ".coding-agent/memory").resolve()
    assert config.rag.index_path.exists() is False
    assert config.memory.path.exists() is False
    assert config.rag.index_path_explicit is False


def test_rag_and_memory_environment_overrides_accept_absolute_paths_inside_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    index_path = workspace / "local/rag.json"
    memory_path = workspace / "local/memory"

    config = load_agent_config(
        write_config(tmp_path, "workspace:\n  path: project\n"),
        environ={
            "CODING_AGENT_RAG_INDEX_PATH": str(index_path),
            "CODING_AGENT_MEMORY_PATH": str(memory_path),
        },
    )

    assert config.rag.index_path == index_path.resolve()
    assert config.memory.path == memory_path.resolve()
    assert config.rag.index_path_explicit is True


def test_rag_and_memory_environment_overrides_expand_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    monkeypatch.setenv("HOME", str(workspace))

    config = load_agent_config(
        write_config(tmp_path, "workspace:\n  path: project\n"),
        environ={
            "CODING_AGENT_RAG_INDEX_PATH": "~/.coding-agent/rag/index.json",
            "CODING_AGENT_MEMORY_PATH": "~/.coding-agent/memory",
        },
    )

    assert config.rag.index_path == (workspace / ".coding-agent/rag/index.json").resolve()
    assert config.memory.path == (workspace / ".coding-agent/memory").resolve()


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("CODING_AGENT_RAG_INDEX_PATH", "../outside/index.json"),
        ("CODING_AGENT_MEMORY_PATH", "../outside/memory"),
    ],
)
def test_rag_and_memory_environment_overrides_reject_relative_escape(
    tmp_path: Path, variable: str, value: str
) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()

    with pytest.raises(AgentConfigError, match=variable):
        load_agent_config(
            write_config(tmp_path, "workspace:\n  path: project\n"),
            environ={variable: value},
        )


@pytest.mark.parametrize(
    ("variable", "suffix"),
    [
        ("CODING_AGENT_RAG_INDEX_PATH", "index.json"),
        ("CODING_AGENT_MEMORY_PATH", "memory"),
    ],
)
def test_rag_and_memory_environment_overrides_reject_absolute_escape(
    tmp_path: Path, variable: str, suffix: str
) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    outside = tmp_path / "outside" / suffix

    with pytest.raises(AgentConfigError, match=variable):
        load_agent_config(
            write_config(tmp_path, "workspace:\n  path: project\n"),
            environ={variable: str(outside)},
        )


def test_loads_all_sections_and_resolves_paths_relative_to_config(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    docs = workspace / "docs"
    docs.mkdir(parents=True)
    config = load_agent_config(
        write_config(
            tmp_path,
            """
workspace:
  path: project
  ignore: [vendor/**, .cache/**]
project:
  name: Sample
  description: Neutral metadata
  tags: [service]
permissions:
  read: true
  write: true
  run_commands: true
  web_search: true
commands:
  test: python -m pytest
  lint: tool check .
limits:
  max_iterations: 12
  context_chars: 6000
  command_timeout_seconds: 45
  max_rag_results: 4
  max_web_results: 3
rag:
  enabled: true
  index_path: .agent/index.json
  sources:
    - name: docs
      loader: local
      source_type: documentation
      path: docs
      patterns: ['**/*.md']
memory:
  enabled: true
  path: .agent/memory
  identifier: sample-id
observability:
  enabled: true
  log_level: DEBUG
  trace_tools: true
web_search:
  enabled: true
  allowed_domains: [docs.example]
  priority_domains: [reference.example]
  blocked_domains: [unsafe.example]
  max_results: 3
  technology_domains:
    runtime-x: [runtime.example]
""",
        )
    )

    assert config.project is not None
    assert config.project.name == "Sample"
    assert config.project.language is None
    assert config.project.framework is None
    assert config.commands["test"] == "python -m pytest"
    assert config.rag.index_path == (workspace / ".agent/index.json").resolve()
    assert config.rag.sources[0].location == docs.resolve().as_posix()
    assert config.memory.path == (workspace / ".agent/memory").resolve()
    assert config.web_search.technology_domains == {
        "runtime-x": ("runtime.example",)
    }


@pytest.mark.parametrize(
    ("yaml_text", "message"),
    [
        ("workspace: []\n", "workspace"),
        ("workspace:\n  path: missing\n", "no existe"),
        ("workspace:\n  path: project\nunknown: true\n", "desconocidos"),
        ("workspace:\n  path: project\nlimits:\n  max_iterations: 0\n", "max_iterations"),
        ("workspace:\n  path: project\npermissions:\n  write: 'yes'\n", "write"),
        ("workspace:\n  path: project\ncommands:\n  test: ''\n", "commands.test"),
        ("workspace:\n  path: project\nobservability:\n  log_level: VERBOSE\n", "log_level"),
        ("workspace:\n  path: project\nweb_search:\n  max_results: 20\n", "max_results"),
    ],
)
def test_rejects_invalid_values(tmp_path: Path, yaml_text: str, message: str) -> None:
    (tmp_path / "project").mkdir()
    with pytest.raises(AgentConfigError, match=message):
        load_agent_config(write_config(tmp_path, yaml_text))


def test_rejects_paths_that_escape_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    config_path = write_config(
        tmp_path,
        """
workspace:
  path: project
memory:
  path: ../outside
""",
    )

    with pytest.raises(AgentConfigError, match="fuera del workspace"):
        load_agent_config(config_path)


def test_rejects_invalid_yaml_and_missing_file(tmp_path: Path) -> None:
    with pytest.raises(AgentConfigError, match="No se pudo leer"):
        load_agent_config(tmp_path / "missing.yaml")
    with pytest.raises(AgentConfigError, match="YAML inválido"):
        load_agent_config(write_config(tmp_path, "workspace: [\n"))
