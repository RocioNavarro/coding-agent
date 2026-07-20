"""Carga y combinación de perfiles declarativos de proyecto."""

from pathlib import Path

import pytest

from core.config import load_agent_config
from core.profiles import ProfileLoader, ProjectProfileError


def write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def valid_profile(tmp_path: Path) -> Path:
    return write(
        tmp_path / "profile.yaml",
        """
name: Profile name
description: Profile description
expected_technologies: [runtime-a, datastore-b]
rag_sources:
  - name: docs
    loader: local
    source_type: documentation
    path: docs
priority_web_domains: [docs.example.com]
important_files: [README.md, config/project.yaml]
suggested_commands:
  test: profile-test
  lint: profile-lint
additional_policies:
  protected_paths: [generated/**]
  review:
    required: true
search_tags: [reference, stable]
""",
    )


def config_with_profile(tmp_path: Path, extra: str = "") -> Path:
    workspace = tmp_path / "workspace"
    (workspace / "docs").mkdir(parents=True)
    valid_profile(tmp_path)
    return write(
        tmp_path / "agent.config.yaml",
        f"profile: profile.yaml\nworkspace:\n  path: workspace\n{extra}",
    )


def test_loads_valid_profile_and_preserves_profile_only_fields(tmp_path: Path) -> None:
    profile = ProfileLoader().load(valid_profile(tmp_path))

    assert profile.name == "Profile name"
    assert profile.expected_technologies == ("runtime-a", "datastore-b")
    assert profile.important_files == ("README.md", "config/project.yaml")
    assert profile.additional_policies["review"]["required"] is True
    assert profile.search_tags == ("reference", "stable")


def test_missing_profile_is_controlled_error(tmp_path: Path) -> None:
    with pytest.raises(ProjectProfileError, match="No se pudo leer"):
        ProfileLoader().load(tmp_path / "missing.yaml")


def test_invalid_yaml_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ProjectProfileError, match="YAML de perfil inválido"):
        ProfileLoader().load(write(tmp_path / "profile.yaml", "name: [\n"))


def test_unknown_profile_fields_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ProjectProfileError, match="Campos desconocidos"):
        ProfileLoader().load(write(tmp_path / "profile.yaml", "unknown: true\n"))


def test_default_profile_is_empty() -> None:
    profile = ProfileLoader().load("profiles/default.yaml")

    assert profile.name is None
    assert profile.expected_technologies == ()
    assert profile.rag_sources == ()
    assert dict(profile.suggested_commands) == {}
    assert dict(profile.additional_policies) == {}


def test_recursive_merge_combines_commands_and_user_values_win(tmp_path: Path) -> None:
    config = load_agent_config(
        config_with_profile(
            tmp_path,
            """
project:
  name: Explicit name
commands:
  test: explicit-test
  build: explicit-build
web_search:
  priority_domains: [user.example.com]
""",
        )
    )

    assert config.project is not None
    assert config.project.name == "Explicit name"
    assert config.project.description == "Profile description"
    assert dict(config.commands) == {
        "test": "explicit-test", "lint": "profile-lint", "build": "explicit-build"
    }
    assert config.web_search.priority_domains == ("user.example.com",)
    assert dict(config.profile.suggested_commands) == dict(config.commands)


def test_explicit_empty_lists_replace_inherited_lists(tmp_path: Path) -> None:
    config = load_agent_config(
        config_with_profile(
            tmp_path,
            """
rag:
  enabled: false
  sources: []
web_search:
  priority_domains: []
""",
        )
    )

    assert config.rag.sources == ()
    assert config.web_search.priority_domains == ()
    assert config.profile.rag_sources == ()
    assert config.profile.priority_web_domains == ()


def test_profile_only_fields_survive_config_merge(tmp_path: Path) -> None:
    config = load_agent_config(config_with_profile(tmp_path))

    assert config.profile.expected_technologies == ("runtime-a", "datastore-b")
    assert config.profile.important_files == ("README.md", "config/project.yaml")
    assert config.profile.additional_policies["protected_paths"] == ("generated/**",)
    assert config.profile.search_tags == ("reference", "stable")


def test_profile_local_rag_paths_resolve_against_final_workspace(tmp_path: Path) -> None:
    config = load_agent_config(config_with_profile(tmp_path))

    expected = (tmp_path / "workspace" / "docs").resolve().as_posix()
    assert config.rag.sources[0].location == expected
    assert config.profile.rag_sources[0].location == expected


def test_agent_config_without_profile_remains_compatible(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = load_agent_config(
        write(
            tmp_path / "agent.config.yaml",
            "workspace:\n  path: workspace\ncommands:\n  test: direct-test\n",
        )
    )

    assert config.profile.expected_technologies == ()
    assert config.profile.important_files == ()
    assert dict(config.commands) == {"test": "direct-test"}
