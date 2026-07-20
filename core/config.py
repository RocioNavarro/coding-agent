"""Modelos y carga estricta de ``agent.config.yaml``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from collections.abc import Iterator
from typing import Any, Mapping
from urllib.parse import urlsplit

import yaml

from rag.models import SourceConfig
from core.profiles import (
    ProfileLoader, ProjectProfile, merge_profile_config, profile_to_config,
)


class AgentConfigError(ValueError):
    """La configuración no puede cargarse o viola su esquema."""


def _mapping(value: object, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise AgentConfigError(f"{field} debe ser un objeto.")
    return dict(value)


def _known(values: Mapping[str, Any], allowed: set[str], field: str) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise AgentConfigError(f"Campos desconocidos en {field}: {', '.join(unknown)}.")


def _text(value: object, field: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        raise AgentConfigError(f"{field} debe ser texto no vacío.")
    return value.strip()


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise AgentConfigError(f"{field} debe ser booleano.")
    return value


def _integer(value: object, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise AgentConfigError(f"{field} debe estar entre {minimum} y {maximum}.")
    return value


def _strings(value: object, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, list):
        raise AgentConfigError(f"{field} debe ser una lista de textos.")
    return tuple(_text(item, field) or "" for item in value)


def _domain(value: object, field: str) -> str:
    text = _text(value, field) or ""
    candidate = text.casefold()
    if "://" in candidate:
        candidate = urlsplit(candidate).hostname or ""
    candidate = candidate.removeprefix("www.").strip(".")
    if not candidate or "/" in candidate or " " in candidate:
        raise AgentConfigError(f"{field} contiene un dominio inválido: {text!r}.")
    return candidate


def _domains(value: object, field: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_domain(item, field) for item in _strings(value, field)))


def _inside(workspace: Path, value: object, field: str) -> Path:
    text = _text(value, field) or ""
    path = Path(text)
    resolved = path.resolve() if path.is_absolute() else (workspace / path).resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as error:
        raise AgentConfigError(f"{field} resuelve fuera del workspace.") from error
    return resolved


@dataclass(frozen=True)
class WorkspaceConfig:
    path: Path
    ignore: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProjectMetadata:
    name: str | None = None
    description: str | None = None
    language: str | None = None
    framework: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class PermissionsConfig:
    read: bool = True
    write: bool = False
    run_commands: bool = False
    web_search: bool = False


@dataclass(frozen=True)
class CommandsConfig(Mapping[str, str]):
    """Comandos nombrados por el proyecto, sin asumir su toolchain."""

    values: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))

    def __getitem__(self, key: str) -> str:
        return self.values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)


@dataclass(frozen=True)
class LimitsConfig:
    max_iterations: int = 20
    context_chars: int = 8_000
    command_timeout_seconds: int = 60
    max_rag_results: int = 5
    max_web_results: int = 5


@dataclass(frozen=True)
class RagConfig:
    enabled: bool
    index_path: Path
    sources: tuple[SourceConfig, ...] = ()


@dataclass(frozen=True)
class MemoryConfig:
    enabled: bool
    path: Path
    identifier: str | None = None


@dataclass(frozen=True)
class ObservabilityConfig:
    enabled: bool = True
    log_level: str = "INFO"
    trace_tools: bool = True


@dataclass(frozen=True)
class WebSearchSettings:
    enabled: bool = False
    allowed_domains: tuple[str, ...] = ()
    priority_domains: tuple[str, ...] = ()
    blocked_domains: tuple[str, ...] = ()
    max_results: int = 5
    technology_domains: Mapping[str, tuple[str, ...]] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "technology_domains",
            MappingProxyType(dict(self.technology_domains or {})),
        )


@dataclass(frozen=True)
class AgentConfig:
    workspace: WorkspaceConfig
    project: ProjectMetadata | None
    permissions: PermissionsConfig
    commands: CommandsConfig
    limits: LimitsConfig
    rag: RagConfig
    memory: MemoryConfig
    observability: ObservabilityConfig
    web_search: WebSearchSettings
    profile: ProjectProfile


def load_agent_config(path: str | Path = "agent.config.yaml") -> AgentConfig:
    """Carga YAML, aplica defaults y resuelve rutas locales con confinamiento."""
    config_path = Path(path).resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as error:
        raise AgentConfigError(f"No se pudo leer '{config_path}': {error}") from error
    except yaml.YAMLError as error:
        raise AgentConfigError(f"YAML inválido en '{config_path}': {error}") from error
    root = _mapping(raw, "configuración")
    profile_value = root.pop("profile", None)
    loaded_profile = ProjectProfile()
    if profile_value is not None:
        profile_text = _text(profile_value, "profile") or ""
        loaded_profile = ProfileLoader().load(config_path.parent / profile_text)
        root = merge_profile_config(profile_to_config(loaded_profile), root)
    _known(
        root,
        {"workspace", "project", "permissions", "commands", "limits", "rag", "memory", "observability", "web_search"},
        "configuración",
    )

    workspace_values = _mapping(root.get("workspace"), "workspace")
    _known(workspace_values, {"path", "ignore"}, "workspace")
    workspace_text = _text(workspace_values.get("path"), "workspace.path")
    workspace_path = (config_path.parent / (workspace_text or "")).resolve()
    if not workspace_path.is_dir():
        raise AgentConfigError(f"workspace.path no existe o no es un directorio: {workspace_path}")
    workspace = WorkspaceConfig(
        workspace_path,
        _strings(workspace_values.get("ignore"), "workspace.ignore"),
    )

    project = _project(root.get("project"))
    permissions = _permissions(root.get("permissions"))
    commands = _commands(root.get("commands"))
    limits = _limits(root.get("limits"))
    rag = _rag(root.get("rag"), workspace_path)
    memory = _memory(root.get("memory"), workspace_path)
    observability = _observability(root.get("observability"))
    web_search = _web_search(root.get("web_search"), limits.max_web_results)
    if web_search.enabled and not permissions.web_search:
        raise AgentConfigError("web_search.enabled requiere permissions.web_search=true.")
    return AgentConfig(
        workspace, project, permissions, commands, limits, rag, memory,
        observability, web_search,
        _effective_profile(loaded_profile, project, commands, rag, web_search),
    )


def _effective_profile(
    original: ProjectProfile,
    project: ProjectMetadata | None,
    commands: CommandsConfig,
    rag: RagConfig,
    web_search: WebSearchSettings,
) -> ProjectProfile:
    """Conserva datos exclusivos y refleja overrides de campos proyectados."""
    return ProjectProfile(
        name=project.name if project is not None else None,
        description=project.description if project is not None else None,
        expected_technologies=original.expected_technologies,
        rag_sources=rag.sources,
        priority_web_domains=web_search.priority_domains,
        important_files=original.important_files,
        suggested_commands=dict(commands),
        additional_policies=original.additional_policies,
        search_tags=original.search_tags,
    )


def _project(value: object) -> ProjectMetadata | None:
    if value is None:
        return None
    values = _mapping(value, "project")
    _known(values, {"name", "description", "language", "framework", "tags"}, "project")
    return ProjectMetadata(
        *(_text(values.get(field), f"project.{field}", optional=True) for field in ("name", "description", "language", "framework")),
        tags=_strings(values.get("tags"), "project.tags"),
    )


def _permissions(value: object) -> PermissionsConfig:
    values = _mapping(value, "permissions")
    _known(values, {"read", "write", "run_commands", "web_search"}, "permissions")
    defaults = PermissionsConfig()
    return PermissionsConfig(
        *(_boolean(values.get(field, getattr(defaults, field)), f"permissions.{field}") for field in ("read", "write", "run_commands", "web_search"))
    )


def _commands(value: object) -> CommandsConfig:
    values = _mapping(value, "commands")
    return CommandsConfig(
        {
            _text(name, "commands key") or "": _text(command, f"commands.{name}") or ""
            for name, command in values.items()
        }
    )


def _limits(value: object) -> LimitsConfig:
    values = _mapping(value, "limits")
    fields = (
        "max_iterations", "context_chars", "command_timeout_seconds",
        "max_rag_results", "max_web_results",
    )
    _known(values, set(fields), "limits")
    defaults = LimitsConfig()
    ranges = {
        "max_iterations": (1, 100), "context_chars": (256, 1_000_000),
        "command_timeout_seconds": (1, 3600), "max_rag_results": (1, 100),
        "max_web_results": (1, 10),
    }
    return LimitsConfig(
        *(_integer(values.get(field, getattr(defaults, field)), f"limits.{field}", *ranges[field]) for field in fields)
    )


def _rag(value: object, workspace: Path) -> RagConfig:
    values = _mapping(value, "rag")
    _known(values, {"enabled", "index_path", "sources"}, "rag")
    enabled = _boolean(values.get("enabled", False), "rag.enabled")
    index_path = _inside(workspace, values.get("index_path", ".coding-agent/rag/index.json"), "rag.index_path")
    raw_sources = values.get("sources", [])
    if not isinstance(raw_sources, list):
        raise AgentConfigError("rag.sources debe ser una lista.")
    sources: list[SourceConfig] = []
    for index, item in enumerate(raw_sources):
        data = _mapping(item, f"rag.sources[{index}]")
        _known(
            data,
            {
                "name", "loader", "source_type", "path", "url", "parser",
                "title", "detected_language", "tags", "patterns", "encoding",
            },
            f"rag.sources[{index}]",
        )
        try:
            source = SourceConfig.from_dict(data)
        except ValueError as error:
            raise AgentConfigError(f"rag.sources[{index}] inválida: {error}") from error
        if source.loader not in {"local", "url"}:
            raise AgentConfigError(
                f"rag.sources[{index}].loader debe ser local o url."
            )
        if source.loader == "local":
            location = _inside(workspace, source.location, f"rag.sources[{index}].path")
            source = SourceConfig(
                source.name, source.loader, source.source_type, location.as_posix(),
                source.parser, source.title, source.detected_language, source.tags,
                source.patterns, source.encoding,
            )
        sources.append(source)
    if enabled and not sources:
        raise AgentConfigError("rag.enabled requiere al menos una fuente.")
    return RagConfig(enabled, index_path, tuple(sources))


def _memory(value: object, workspace: Path) -> MemoryConfig:
    values = _mapping(value, "memory")
    _known(values, {"enabled", "path", "identifier"}, "memory")
    return MemoryConfig(
        _boolean(values.get("enabled", True), "memory.enabled"),
        _inside(workspace, values.get("path", ".coding-agent/memory"), "memory.path"),
        _text(values.get("identifier"), "memory.identifier", optional=True),
    )


def _observability(value: object) -> ObservabilityConfig:
    values = _mapping(value, "observability")
    _known(values, {"enabled", "log_level", "trace_tools"}, "observability")
    level = (_text(values.get("log_level", "INFO"), "observability.log_level") or "").upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise AgentConfigError("observability.log_level es inválido.")
    return ObservabilityConfig(
        _boolean(values.get("enabled", True), "observability.enabled"),
        level,
        _boolean(values.get("trace_tools", True), "observability.trace_tools"),
    )


def _web_search(value: object, default_max: int) -> WebSearchSettings:
    values = _mapping(value, "web_search")
    fields = {"enabled", "allowed_domains", "priority_domains", "blocked_domains", "max_results", "technology_domains"}
    _known(values, fields, "web_search")
    raw_mapping = _mapping(values.get("technology_domains"), "web_search.technology_domains")
    technology_domains = {
        (_text(name, "technology_domains key") or "").casefold(): _domains(domains, f"technology_domains.{name}")
        for name, domains in raw_mapping.items()
    }
    return WebSearchSettings(
        _boolean(values.get("enabled", False), "web_search.enabled"),
        _domains(values.get("allowed_domains"), "web_search.allowed_domains"),
        _domains(values.get("priority_domains"), "web_search.priority_domains"),
        _domains(values.get("blocked_domains"), "web_search.blocked_domains"),
        _integer(values.get("max_results", default_max), "web_search.max_results", 1, 10),
        technology_domains,
    )
