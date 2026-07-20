"""Perfiles declarativos de proyecto, independientes de la ejecución del agente."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping
from urllib.parse import urlsplit

import yaml

if TYPE_CHECKING:
    from rag.models import SourceConfig


class ProjectProfileError(ValueError):
    """El perfil no existe o no cumple el esquema declarado."""


def _text(value: object, field: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ProjectProfileError(f"{field} debe ser texto no vacío.")
    return value.strip()


def _strings(value: object, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, list):
        raise ProjectProfileError(f"{field} debe ser una lista de textos.")
    return tuple(_text(item, field) or "" for item in value)


def _domain(value: str, field: str) -> str:
    candidate = value.casefold()
    if "://" in candidate:
        candidate = urlsplit(candidate).hostname or ""
    candidate = candidate.removeprefix("www.").strip(".")
    if not candidate or "/" in candidate or " " in candidate:
        raise ProjectProfileError(f"{field} contiene un dominio inválido: {value!r}.")
    return candidate


def _freeze(value: Any, field: str) -> Any:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) and key.strip() for key in value):
            raise ProjectProfileError(f"{field} debe usar claves de texto no vacías.")
        return MappingProxyType(
            {key.strip(): _freeze(item, f"{field}.{key}") for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item, field) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ProjectProfileError(f"{field} contiene un valor no soportado.")


@dataclass(frozen=True)
class ProjectProfile:
    """Preferencias reutilizables que no ejecutan comportamiento por sí mismas."""

    name: str | None = None
    description: str | None = None
    expected_technologies: tuple[str, ...] = ()
    rag_sources: tuple["SourceConfig", ...] = ()
    priority_web_domains: tuple[str, ...] = ()
    important_files: tuple[str, ...] = ()
    suggested_commands: Mapping[str, str] = field(default_factory=dict)
    additional_policies: Mapping[str, Any] = field(default_factory=dict)
    search_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "name", optional=True))
        object.__setattr__(
            self, "description", _text(self.description, "description", optional=True)
        )
        for field in ("expected_technologies", "important_files", "search_tags"):
            raw = getattr(self, field)
            if not isinstance(raw, (list, tuple)):
                raise ProjectProfileError(f"{field} debe ser una lista o tupla.")
            object.__setattr__(
                self, field, tuple(_text(item, field) or "" for item in raw)
            )
        from rag.models import SourceConfig

        if not isinstance(self.rag_sources, (list, tuple)) or not all(
            isinstance(item, SourceConfig) for item in self.rag_sources
        ):
            raise ProjectProfileError("rag_sources debe contener fuentes RAG válidas.")
        object.__setattr__(self, "rag_sources", tuple(self.rag_sources))
        domains = tuple(
            dict.fromkeys(
                _domain(_text(item, "priority_web_domains") or "", "priority_web_domains")
                for item in self.priority_web_domains
            )
        )
        object.__setattr__(self, "priority_web_domains", domains)
        if not isinstance(self.suggested_commands, Mapping):
            raise ProjectProfileError("suggested_commands debe ser un objeto.")
        commands = {
            _text(key, "suggested_commands key") or "":
            _text(value, f"suggested_commands.{key}") or ""
            for key, value in self.suggested_commands.items()
        }
        object.__setattr__(self, "suggested_commands", MappingProxyType(commands))
        if not isinstance(self.additional_policies, Mapping):
            raise ProjectProfileError("additional_policies debe ser un objeto.")
        object.__setattr__(
            self, "additional_policies", _freeze(self.additional_policies, "additional_policies")
        )


class ProfileLoader:
    """Carga YAML estricto sin resolver aún rutas contra un workspace."""

    _FIELDS = frozenset(
        {"name", "description", "expected_technologies", "rag_sources",
         "priority_web_domains", "important_files", "suggested_commands",
         "additional_policies", "search_tags"}
    )

    def load(self, path: str | Path) -> ProjectProfile:
        profile_path = Path(path).resolve()
        try:
            raw = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError) as error:
            raise ProjectProfileError(
                f"No se pudo leer el perfil '{profile_path}': {error}"
            ) from error
        except yaml.YAMLError as error:
            raise ProjectProfileError(f"YAML de perfil inválido: {error}") from error
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping) or not all(isinstance(key, str) for key in raw):
            raise ProjectProfileError("El perfil debe ser un objeto YAML.")
        unknown = sorted(set(raw) - self._FIELDS)
        if unknown:
            raise ProjectProfileError(
                f"Campos desconocidos en perfil: {', '.join(unknown)}."
            )
        sources = self._sources(raw.get("rag_sources"))
        return ProjectProfile(
            name=_text(raw.get("name"), "name", optional=True),
            description=_text(raw.get("description"), "description", optional=True),
            expected_technologies=_strings(
                raw.get("expected_technologies"), "expected_technologies"
            ),
            rag_sources=sources,
            priority_web_domains=_strings(
                raw.get("priority_web_domains"), "priority_web_domains"
            ),
            important_files=_strings(raw.get("important_files"), "important_files"),
            suggested_commands=self._mapping(
                raw.get("suggested_commands"), "suggested_commands"
            ),
            additional_policies=self._mapping(
                raw.get("additional_policies"), "additional_policies"
            ),
            search_tags=_strings(raw.get("search_tags"), "search_tags"),
        )

    @staticmethod
    def _mapping(value: object, field: str) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
            raise ProjectProfileError(f"{field} debe ser un objeto.")
        return dict(value)

    def _sources(self, value: object) -> tuple["SourceConfig", ...]:
        from rag.models import SourceConfig

        if value is None:
            return ()
        if not isinstance(value, list):
            raise ProjectProfileError("rag_sources debe ser una lista.")
        sources: list[SourceConfig] = []
        for index, item in enumerate(value):
            data = self._mapping(item, f"rag_sources[{index}]")
            allowed = {
                "name", "loader", "source_type", "path", "url", "parser",
                "title", "detected_language", "tags", "patterns", "encoding",
            }
            unknown = sorted(set(data) - allowed)
            if unknown:
                raise ProjectProfileError(
                    f"Campos desconocidos en rag_sources[{index}]: {', '.join(unknown)}."
                )
            try:
                source = SourceConfig.from_dict(data)
            except ValueError as error:
                raise ProjectProfileError(f"rag_sources[{index}] inválida: {error}") from error
            if source.loader not in {"local", "url"}:
                raise ProjectProfileError(
                    f"rag_sources[{index}].loader debe ser local o url."
                )
            sources.append(source)
        return tuple(sources)


def profile_to_config(profile: ProjectProfile) -> dict[str, Any]:
    """Proyecta sólo campos con equivalente actual en agent.config.yaml."""
    project: dict[str, Any] = {}
    if profile.name is not None:
        project["name"] = profile.name
    if profile.description is not None:
        project["description"] = profile.description
    return {
        **({"project": project} if project else {}),
        "commands": dict(profile.suggested_commands),
        "rag": {"sources": [
            {
                "name": source.name, "loader": source.loader,
                "source_type": source.source_type,
                ("path" if source.loader == "local" else "url"): source.location,
                "parser": source.parser, "title": source.title,
                "detected_language": source.detected_language,
                "tags": list(source.tags), "patterns": list(source.patterns),
                "encoding": source.encoding,
            }
            for source in profile.rag_sources
        ]},
        "web_search": {"priority_domains": list(profile.priority_web_domains)},
    }


def merge_profile_config(base: Mapping[str, Any], explicit: Mapping[str, Any]) -> dict[str, Any]:
    """Combina mappings; cualquier valor explícito, incluso [], reemplaza la base."""
    merged = dict(base)
    for key, value in explicit.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = merge_profile_config(merged[key], value)
        else:
            merged[key] = value
    return merged
