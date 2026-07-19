"""Modelos neutrales usados durante carga, parsing e indexación."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


def required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} no puede estar vacío.")
    return value.strip()


@dataclass(frozen=True)
class SourceConfig:
    name: str
    loader: str
    source_type: str
    location: str
    parser: str = "sections"
    title: str | None = None
    detected_language: str | None = None
    tags: tuple[str, ...] = ()
    patterns: tuple[str, ...] = ()
    encoding: str = "utf-8"

    def __post_init__(self) -> None:
        for field in ("name", "loader", "source_type", "location", "parser", "encoding"):
            object.__setattr__(self, field, required_text(getattr(self, field), field))
        if self.title is not None:
            object.__setattr__(self, "title", required_text(self.title, "title"))
        if self.detected_language is not None:
            object.__setattr__(
                self, "detected_language",
                required_text(self.detected_language, "detected_language"),
            )
        object.__setattr__(self, "tags", tuple(required_text(tag, "tag") for tag in self.tags))
        object.__setattr__(
            self, "patterns", tuple(required_text(pattern, "pattern") for pattern in self.patterns)
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SourceConfig:
        try:
            location = data.get("path", data.get("url"))
            return cls(
                name=data["name"],
                loader=data["loader"],
                source_type=data["source_type"],
                location=location,
                parser=data.get("parser", "sections"),
                title=data.get("title"),
                detected_language=data.get("detected_language"),
                tags=tuple(data.get("tags", ())),
                patterns=tuple(data.get("patterns", ())),
                encoding=data.get("encoding", "utf-8"),
            )
        except (KeyError, TypeError) as error:
            raise ValueError(f"Fuente inválida: {error}") from error


@dataclass(frozen=True)
class RawDocument:
    document_id: str
    source: SourceConfig
    path_or_url: str
    content: str


@dataclass(frozen=True)
class DocumentSection:
    title: str
    content: str


@dataclass(frozen=True)
class ParsedDocument:
    raw: RawDocument
    title: str
    sections: tuple[DocumentSection, ...]


@dataclass(frozen=True)
class ChunkDraft:
    document_id: str
    source_name: str
    source_type: str
    path_or_url: str
    title: str
    section: str
    detected_language: str | None
    tags: tuple[str, ...]
    chunk_index: int
    content: str


@dataclass(frozen=True)
class ChunkMetadata:
    document_id: str
    source_name: str
    source_type: str
    path_or_url: str
    title: str
    section: str
    detected_language: str | None
    tags: tuple[str, ...]
    chunk_index: int
    content_hash: str
    indexed_at: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["tags"] = list(self.tags)
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ChunkMetadata:
        values = dict(data)
        values["tags"] = tuple(values.get("tags", ()))
        return cls(**values)


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    content: str
    embedding: tuple[float, ...]
    metadata: ChunkMetadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "embedding": list(self.embedding),
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DocumentChunk:
        return cls(
            chunk_id=data["chunk_id"],
            content=data["content"],
            embedding=tuple(float(item) for item in data["embedding"]),
            metadata=ChunkMetadata.from_dict(data["metadata"]),
        )
