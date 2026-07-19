"""Vector store persistente con contrato intercambiable."""

from __future__ import annotations

import json
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

from rag.models import DocumentChunk


INDEX_SCHEMA_VERSION = 1


class VectorStoreError(RuntimeError):
    pass


class VectorStore(ABC):
    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def save(self) -> None:
        ...

    @abstractmethod
    def document_hash(self, document_id: str) -> str | None:
        ...

    @abstractmethod
    def replace_document(
        self, document_id: str, content_hash: str, chunks: Sequence[DocumentChunk]
    ) -> None:
        ...

    @abstractmethod
    def remove_documents_except(self, document_ids: set[str]) -> int:
        ...

    @abstractmethod
    def all_chunks(self) -> tuple[DocumentChunk, ...]:
        ...


class JsonVectorStore(VectorStore):
    """Store JSON auditable; las implementaciones externas pueden reemplazarlo."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self._documents: dict[str, dict[str, Any]] = {}
        self._chunks: dict[str, DocumentChunk] = {}

    def load(self) -> None:
        if not self.path.exists():
            self._documents = {}
            self._chunks = {}
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("schema_version") != INDEX_SCHEMA_VERSION:
                raise ValueError("versión de índice no soportada")
            self._documents = dict(data["documents"])
            self._chunks = {
                item["chunk_id"]: DocumentChunk.from_dict(item)
                for item in data["chunks"]
            }
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise VectorStoreError(f"Índice inválido: {error}") from error

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "documents": self._documents,
            "chunks": [self._chunks[key].to_dict() for key in sorted(self._chunks)],
        }
        serialized = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        temporary: str | None = None
        try:
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            temporary = None
        except OSError as error:
            raise VectorStoreError(f"No se pudo guardar el índice: {error}") from error
        finally:
            if temporary:
                try:
                    Path(temporary).unlink()
                except OSError:
                    pass

    def document_hash(self, document_id: str) -> str | None:
        document = self._documents.get(document_id)
        return document.get("content_hash") if document else None

    def replace_document(
        self, document_id: str, content_hash: str, chunks: Sequence[DocumentChunk]
    ) -> None:
        previous = self._documents.get(document_id, {})
        for chunk_id in previous.get("chunk_ids", []):
            self._chunks.pop(chunk_id, None)
        unique: dict[str, DocumentChunk] = {}
        seen_hashes: set[str] = set()
        for chunk in chunks:
            if chunk.metadata.content_hash in seen_hashes:
                continue
            seen_hashes.add(chunk.metadata.content_hash)
            unique[chunk.chunk_id] = chunk
        self._chunks.update(unique)
        self._documents[document_id] = {
            "content_hash": content_hash,
            "chunk_ids": sorted(unique),
        }

    def remove_documents_except(self, document_ids: set[str]) -> int:
        removed = 0
        for document_id in tuple(self._documents):
            if document_id in document_ids:
                continue
            for chunk_id in self._documents[document_id].get("chunk_ids", []):
                self._chunks.pop(chunk_id, None)
            del self._documents[document_id]
            removed += 1
        return removed

    def all_chunks(self) -> tuple[DocumentChunk, ...]:
        return tuple(self._chunks[key] for key in sorted(self._chunks))
