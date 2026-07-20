"""Recuperación vectorial genérica con filtros y trazabilidad explícita."""

from __future__ import annotations

import math
from time import perf_counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from core.research_ports import EvidenceFragment, KnowledgeRetriever
from rag.embeddings import EmbeddingProvider
from rag.models import ChunkMetadata, DocumentChunk
from rag.vector_store import VectorStore
from core.observability import NoOpObservabilityClient, ObservabilityClient, ObservabilityEvent, emit_observation


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    content: str
    metadata: ChunkMetadata
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "metadata": self.metadata.to_dict(),
            "score": self.score,
        }


@dataclass(frozen=True)
class RetrievalSufficiency:
    sufficient: bool
    confidence: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RagRetrievalTrace:
    query: str
    filters: dict[str, tuple[str, ...]]
    retrieved_chunks: tuple[RetrievedChunk, ...]
    used_chunks: tuple[RetrievedChunk, ...]
    documents: tuple[str, ...]
    conclusions: tuple[str, ...]
    sufficiency: RetrievalSufficiency

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "filters": {key: list(value) for key, value in self.filters.items()},
            "retrieved_chunks": [item.to_dict() for item in self.retrieved_chunks],
            "used_chunks": [item.to_dict() for item in self.used_chunks],
            "scores": {item.chunk_id: item.score for item in self.retrieved_chunks},
            "documents": list(self.documents),
            "conclusions": list(self.conclusions),
            "sufficiency": {
                "sufficient": self.sufficiency.sufficient,
                "confidence": self.sufficiency.confidence,
                "reasons": list(self.sufficiency.reasons),
            },
        }


class RagRetriever(KnowledgeRetriever):
    """Busca en un índice sin asumir tecnologías ni esquemas de proyecto."""

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        top_k: int = 5,
        relevance_threshold: float = 0.2,
        min_chunks_for_sufficiency: int = 1,
        min_documents_for_sufficiency: int = 1,
        default_filters: Mapping[str, str | Sequence[str]] | None = None,
        observability: ObservabilityClient | None = None,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k debe ser positivo.")
        if not 0 <= relevance_threshold <= 1:
            raise ValueError("relevance_threshold debe estar entre 0 y 1.")
        if min_chunks_for_sufficiency < 1 or min_documents_for_sufficiency < 1:
            raise ValueError("Los mínimos de suficiencia deben ser positivos.")
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold
        self.min_chunks = min_chunks_for_sufficiency
        self.min_documents = min_documents_for_sufficiency
        self.default_filters = self._normalize_filters(default_filters or {})
        self.last_trace: RagRetrievalTrace | None = None
        self.observability = observability or NoOpObservabilityClient()

    def retrieve(self, query: str, *, limit: int = 5) -> Sequence[EvidenceFragment]:
        return self.retrieve_filtered(query, filters=None, limit=limit)

    def retrieve_filtered(
        self,
        query: str,
        *,
        filters: Mapping[str, str | Sequence[str]] | None = None,
        limit: int = 5,
    ) -> Sequence[EvidenceFragment]:
        trace = self.retrieve_context(
            query, filters=filters, top_k=min(limit, self.top_k)
        )
        return tuple(
            EvidenceFragment(
                "rag",
                f"rag://{item.metadata.document_id}/{item.metadata.chunk_index}",
                item.content,
                item.score,
            )
            for item in trace.used_chunks
        )

    def retrieve_context(
        self,
        query: str,
        *,
        filters: Mapping[str, str | Sequence[str]] | None = None,
        top_k: int | None = None,
    ) -> RagRetrievalTrace:
        started = perf_counter()
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query no puede estar vacía.")
        limit = top_k if top_k is not None else self.top_k
        if limit < 1:
            raise ValueError("top_k debe ser positivo.")
        effective_filters = dict(self.default_filters)
        effective_filters.update(self._normalize_filters(filters or {}))
        self.vector_store.load()
        query_vectors = self.embedding_provider.embed((query.strip(),))
        if len(query_vectors) != 1:
            raise ValueError("EmbeddingProvider debe devolver un vector por consulta.")
        query_vector = query_vectors[0]
        scored: list[RetrievedChunk] = []
        for chunk in self.vector_store.all_chunks():
            if not self._matches(chunk, effective_filters):
                continue
            score = self._cosine(query_vector, chunk.embedding)
            if score >= self.relevance_threshold:
                scored.append(RetrievedChunk(chunk.chunk_id, chunk.content, chunk.metadata, score))
        retrieved = tuple(sorted(scored, key=lambda item: (-item.score, item.chunk_id)))
        used: list[RetrievedChunk] = []
        seen_hashes: set[str] = set()
        for item in retrieved:
            if item.metadata.content_hash in seen_hashes:
                continue
            seen_hashes.add(item.metadata.content_hash)
            used.append(item)
            if len(used) >= limit:
                break
        documents = tuple(dict.fromkeys(item.metadata.document_id for item in used))
        sufficiency = self._evaluate_sufficiency(used, documents)
        conclusions = self._conclusions(retrieved, used, sufficiency)
        trace = RagRetrievalTrace(
            query.strip(), effective_filters, retrieved, tuple(used), documents,
            conclusions, sufficiency,
        )
        self.last_trace = trace
        emit_observation(
            self.observability,
            ObservabilityEvent(
                "rag", "rag-retrieval",
                payload={"query": trace.query, "top_k": limit,
                         "filters": trace.filters,
                         "retrieved_chunks": [item.chunk_id for item in trace.retrieved_chunks],
                         "used_chunks": [item.chunk_id for item in trace.used_chunks],
                         "scores": {item.chunk_id: item.score for item in trace.retrieved_chunks},
                         "documents": trace.documents,
                         "sections": [item.metadata.section for item in trace.used_chunks],
                         "sufficiency": {"sufficient": sufficiency.sufficient,
                                         "confidence": sufficiency.confidence,
                                         "reasons": sufficiency.reasons}},
                latency_ms=(perf_counter() - started) * 1000,
            ),
        )
        return trace

    def retrieval_audit(self) -> Mapping[str, Any] | None:
        return self.last_trace.to_dict() if self.last_trace else None

    def _evaluate_sufficiency(
        self, chunks: Sequence[RetrievedChunk], documents: Sequence[str]
    ) -> RetrievalSufficiency:
        reasons = []
        if len(chunks) < self.min_chunks:
            reasons.append(f"Se requieren al menos {self.min_chunks} chunks relevantes.")
        if len(documents) < self.min_documents:
            reasons.append(f"Se requieren al menos {self.min_documents} documentos distintos.")
        sufficient = not reasons
        average = sum(item.score for item in chunks) / len(chunks) if chunks else 0.0
        coverage = min(1.0, len(chunks) / self.min_chunks) * 0.5
        confidence = min(1.0, average * 0.5 + coverage)
        return RetrievalSufficiency(sufficient, confidence, tuple(reasons))

    @staticmethod
    def _conclusions(
        retrieved: Sequence[RetrievedChunk],
        used: Sequence[RetrievedChunk],
        sufficiency: RetrievalSufficiency,
    ) -> tuple[str, ...]:
        if not retrieved:
            return ("No se recuperaron chunks que superen filtros y umbral.",)
        status = "suficiente" if sufficiency.sufficient else "insuficiente"
        return (
            f"Se recuperaron {len(retrieved)} chunks y se utilizaron {len(used)}.",
            f"La evidencia vectorial se evaluó como {status}.",
        )

    @staticmethod
    def _normalize_filters(
        filters: Mapping[str, str | Sequence[str]],
    ) -> dict[str, tuple[str, ...]]:
        normalized = {}
        for key, raw in filters.items():
            values = (raw,) if isinstance(raw, str) else tuple(raw)
            clean = tuple(str(value).strip() for value in values if str(value).strip())
            if clean:
                normalized[str(key).strip()] = clean
        return normalized

    @staticmethod
    def _matches(
        chunk: DocumentChunk, filters: Mapping[str, Sequence[str]]
    ) -> bool:
        metadata = chunk.metadata.to_dict()
        aliases = {
            "language": "detected_language",
            "source": "source_name",
            "type": "source_type",
        }
        tags = {tag.casefold() for tag in chunk.metadata.tags}
        for key, expected in filters.items():
            metadata_key = aliases.get(key, key)
            actual = metadata.get(metadata_key)
            actual_values = actual if isinstance(actual, list) else [actual]
            normalized_actual = {str(value).casefold() for value in actual_values if value is not None}
            matches = False
            for value in expected:
                normalized = value.casefold()
                if normalized in normalized_actual or normalized in tags:
                    matches = True
                    break
                if f"{key.casefold()}:{normalized}" in tags:
                    matches = True
                    break
            if not matches:
                return False
        return True

    @staticmethod
    def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right):
            raise ValueError("La dimensión del índice no coincide con el embedding de consulta.")
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if not left_norm or not right_norm:
            return 0.0
        raw = sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
        return max(0.0, min(1.0, raw))
