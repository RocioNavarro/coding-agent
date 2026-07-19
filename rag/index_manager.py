"""Coordinación incremental de carga, transformación, embeddings y persistencia."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Sequence

from rag.embeddings import EmbeddingProvider
from rag.models import ChunkMetadata, DocumentChunk, RawDocument, SourceConfig
from rag.processing import Chunker, DocumentParser, TextNormalizer
from rag.sources import SourceLoader
from rag.vector_store import VectorStore


@dataclass(frozen=True)
class IndexingResult:
    loaded_documents: int
    indexed_documents: int
    updated_documents: int
    skipped_documents: int
    duplicate_documents: int
    chunks_written: int
    removed_documents: int


class IndexManager:
    """Ejecuta el pipeline y conserva sólo cambios reales del corpus."""

    def __init__(
        self,
        *,
        source_loader: SourceLoader,
        parsers: Mapping[str, DocumentParser],
        normalizer: TextNormalizer,
        chunker: Chunker,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        if not parsers:
            raise ValueError("Debe configurarse al menos un parser.")
        self.source_loader = source_loader
        self.parsers = dict(parsers)
        self.normalizer = normalizer
        self.chunker = chunker
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def index(
        self, sources: Sequence[SourceConfig], *, prune: bool = False
    ) -> IndexingResult:
        loaded = self.source_loader.load(sources)
        documents: dict[str, RawDocument] = {}
        duplicates = 0
        for document in loaded:
            if document.document_id in documents:
                duplicates += 1
            else:
                documents[document.document_id] = document

        self.vector_store.load()
        indexed = updated = skipped = chunks_written = 0
        indexed_at = datetime.now(timezone.utc).isoformat()
        for document_id in sorted(documents):
            document = documents[document_id]
            parser = self.parsers.get(document.source.parser)
            if parser is None:
                raise ValueError(f"Parser no configurado: {document.source.parser}")
            document_hash = self._document_hash(document, parser)
            previous = self.vector_store.document_hash(document_id)
            if previous == document_hash:
                skipped += 1
                continue
            drafts = self.chunker.chunk(parser.parse(document), self.normalizer)
            unique_drafts = []
            seen_content: set[str] = set()
            for draft in drafts:
                content_hash = hashlib.sha256(draft.content.encode("utf-8")).hexdigest()
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_drafts.append((draft, content_hash))
            embeddings = self.embedding_provider.embed(
                [draft.content for draft, _ in unique_drafts]
            )
            chunks = tuple(
                DocumentChunk(
                    hashlib.sha256(f"{document_id}:{content_hash}".encode("utf-8")).hexdigest(),
                    draft.content,
                    embedding,
                    ChunkMetadata(
                        draft.document_id,
                        draft.source_name,
                        draft.source_type,
                        draft.path_or_url,
                        draft.title,
                        draft.section,
                        draft.detected_language,
                        draft.tags,
                        draft.chunk_index,
                        content_hash,
                        indexed_at,
                    ),
                )
                for (draft, content_hash), embedding in zip(unique_drafts, embeddings)
            )
            self.vector_store.replace_document(document_id, document_hash, chunks)
            chunks_written += len(chunks)
            if previous is None:
                indexed += 1
            else:
                updated += 1

        removed = (
            self.vector_store.remove_documents_except(set(documents)) if prune else 0
        )
        self.vector_store.save()
        return IndexingResult(
            len(loaded), indexed, updated, skipped, duplicates,
            chunks_written, removed,
        )

    def _document_hash(
        self, document: RawDocument, parser: DocumentParser
    ) -> str:
        payload = {
            "content": document.content,
            "source_name": document.source.name,
            "source_type": document.source.source_type,
            "title": document.source.title,
            "language": document.source.detected_language,
            "tags": document.source.tags,
            "parser": document.source.parser,
            "parser_fingerprint": parser.fingerprint(),
            "normalizer_fingerprint": self.normalizer.fingerprint(),
            "chunker_fingerprint": self.chunker.fingerprint(),
            "embedding_fingerprint": self.embedding_provider.fingerprint(),
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
