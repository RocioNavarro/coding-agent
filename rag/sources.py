"""Carga de fuentes definida exclusivamente mediante manifiestos."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.request import Request, urlopen

from rag.models import RawDocument, SourceConfig


class SourceLoadError(RuntimeError):
    pass


class SourceLoader(ABC):
    @abstractmethod
    def load(self, sources: Sequence[SourceConfig]) -> tuple[RawDocument, ...]:
        """Carga documentos declarados, sin descubrir tecnologías implícitamente."""


UrlFetcher = Callable[[str, str], str]


class SourceManifest:
    """Configuración resoluble contra un workspace o, por compatibilidad, el manifiesto."""

    def __init__(
        self,
        *,
        sources: Sequence[SourceConfig],
        index_path: str,
        chunking: Mapping[str, Any] | None = None,
        embedding: Mapping[str, Any] | None = None,
        base_path: str | Path = ".",
        workspace: str | Path | None = None,
    ) -> None:
        self.sources = tuple(sources)
        self.index_path = index_path
        self.chunking = dict(chunking or {})
        self.embedding = dict(embedding or {})
        self.base_path = Path(base_path).resolve()
        self.workspace = Path(workspace).expanduser().resolve() if workspace else None
        if self.workspace is not None and not self.workspace.is_dir():
            raise SourceLoadError(
                f"El workspace del manifiesto no existe o no es un directorio: {self.workspace}"
            )

    @classmethod
    def load(
        cls, path: str | Path, *, workspace: str | Path | None = None
    ) -> SourceManifest:
        manifest_path = Path(path).resolve()
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            sources = tuple(SourceConfig.from_dict(item) for item in data["sources"])
            index_path = data["index_path"]
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise SourceLoadError(f"Manifiesto inválido: {error}") from error
        if not isinstance(index_path, str) or not index_path.strip():
            raise SourceLoadError("index_path no puede estar vacío.")
        return cls(
            sources=sources,
            index_path=index_path,
            chunking=data.get("chunking"),
            embedding=data.get("embedding"),
            base_path=manifest_path.parent,
            workspace=workspace,
        )

    def _resolve_local_path(self, value: str, field: str) -> Path:
        path = Path(value).expanduser()
        root = self.workspace or self.base_path
        resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
        if self.workspace is not None:
            try:
                resolved.relative_to(self.workspace)
            except ValueError as error:
                raise SourceLoadError(
                    f"{field} resuelve fuera del workspace: {resolved}"
                ) from error
        return resolved

    def resolved_sources(self) -> tuple[SourceConfig, ...]:
        result = []
        for source in self.sources:
            location = source.location
            if source.loader == "local":
                location = self._resolve_local_path(
                    location, f"sources.{source.name}.path"
                ).as_posix()
            result.append(
                SourceConfig(
                    source.name, source.loader, source.source_type, location,
                    source.parser, source.title, source.detected_language,
                    source.tags, source.patterns, source.encoding,
                )
            )
        return tuple(result)

    def resolved_index_path(self) -> Path:
        return self._resolve_local_path(self.index_path, "index_path")


class ConfiguredSourceLoader(SourceLoader):
    """Carga archivos/directorios o URLs sólo cuando el manifiesto lo indica."""

    def __init__(self, url_fetcher: UrlFetcher | None = None) -> None:
        self.url_fetcher = url_fetcher or self._fetch_url

    def load(self, sources: Sequence[SourceConfig]) -> tuple[RawDocument, ...]:
        documents: dict[str, RawDocument] = {}
        for source in sources:
            loaded = self._load_local(source) if source.loader == "local" else self._load_url(source)
            for document in loaded:
                documents.setdefault(document.document_id, document)
        return tuple(documents[key] for key in sorted(documents))

    def _load_local(self, source: SourceConfig) -> tuple[RawDocument, ...]:
        location = Path(source.location).resolve()
        if location.is_file():
            paths = (location,)
        elif location.is_dir():
            if not source.patterns:
                raise SourceLoadError(f"La fuente directorio '{source.name}' requiere patterns.")
            paths = tuple(
                sorted(
                    {
                        candidate.resolve()
                        for pattern in source.patterns
                        for candidate in location.glob(pattern)
                        if candidate.is_file() and not candidate.is_symlink()
                    }
                )
            )
        else:
            raise SourceLoadError(f"No existe la fuente local: {location}")
        result = []
        for path in paths:
            try:
                content = path.read_text(encoding=source.encoding)
            except (OSError, UnicodeError) as error:
                raise SourceLoadError(f"No se pudo leer '{path}': {error}") from error
            result.append(self._document(source, path.as_posix(), content))
        return tuple(result)

    def _load_url(self, source: SourceConfig) -> tuple[RawDocument, ...]:
        if source.loader != "url" or not source.location.startswith(("http://", "https://")):
            raise SourceLoadError(f"Loader no soportado: {source.loader}")
        try:
            content = self.url_fetcher(source.location, source.encoding)
        except Exception as error:
            raise SourceLoadError(f"No se pudo descargar '{source.location}': {error}") from error
        return (self._document(source, source.location, content),)

    @staticmethod
    def _document(source: SourceConfig, location: str, content: str) -> RawDocument:
        document_id = hashlib.sha256(location.encode("utf-8")).hexdigest()
        return RawDocument(document_id, source, location, content)

    @staticmethod
    def _fetch_url(url: str, encoding: str) -> str:
        request = Request(url, headers={"User-Agent": "coding-agent-rag-indexer/1"})
        with urlopen(request, timeout=20) as response:
            return response.read().decode(encoding)
