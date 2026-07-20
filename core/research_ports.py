"""Contratos de investigación compartidos entre agents/ y rag/, sin ciclo entre ambos.

Estos tipos vivían en agents/researcher.py, pero tanto rag/retriever.py como
agents/project_memory.py y agents/web_research.py necesitan importarlos, lo que
generaba una dependencia circular entre los paquetes rag y agents (rag, pensado
como infraestructura independiente, terminaba dependiendo del paquete de
aplicación agents para sus propios tipos de valor).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from core.task_state import SourceOrigin, SourceReference


@dataclass(frozen=True)
class EvidenceFragment:
    """Fragmento técnico recuperado con origen, referencia y relevancia."""

    origin: SourceOrigin
    reference: str
    content: str
    relevance: float = 1.0

    def __post_init__(self) -> None:
        for field_name in ("reference", "content"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} no puede estar vacío.")
            object.__setattr__(self, field_name, value.strip())
        if (
            isinstance(self.relevance, bool)
            or not isinstance(self.relevance, (int, float))
            or not 0 <= self.relevance <= 1
        ):
            raise ValueError("relevance debe estar entre 0 y 1.")
        object.__setattr__(self, "relevance", float(self.relevance))

    def to_source(self) -> SourceReference:
        return SourceReference(self.origin, self.reference, self.content[:300])

    def to_dict(self) -> dict[str, object]:
        return {
            "origin": self.origin,
            "reference": self.reference,
            "content": self.content,
            "relevance": self.relevance,
        }


class ProjectMemoryProvider(ABC):
    @abstractmethod
    def search(self, query: str, *, limit: int = 5) -> Sequence[EvidenceFragment]:
        """Recupera decisiones y conocimiento persistido del proyecto."""


class KnowledgeRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, *, limit: int = 5) -> Sequence[EvidenceFragment]:
        """Recupera fragmentos desde el índice RAG configurado."""

    def retrieve_filtered(
        self,
        query: str,
        *,
        filters: Mapping[str, str | Sequence[str]] | None = None,
        limit: int = 5,
    ) -> Sequence[EvidenceFragment]:
        """Compatibilidad para proveedores sin soporte de filtros dinámicos."""
        return self.retrieve(query, limit=limit)

    def retrieval_audit(self) -> Mapping[str, Any] | None:
        """Devuelve la última traza cuando el proveedor la soporta."""
        return None


class WebSearchProvider(ABC):
    @abstractmethod
    def search(self, query: str, *, limit: int = 5) -> Sequence[EvidenceFragment]:
        """Busca evidencia externa únicamente cuando memoria y RAG no alcanzan."""

    def search_context(
        self,
        query: str,
        *,
        limit: int = 5,
        technologies: Sequence[str] = (),
        rag_metadata: Sequence[Mapping[str, Any]] = (),
    ) -> Sequence[EvidenceFragment]:
        """Compatibilidad para proveedores sin priorización contextual."""
        return self.search(query, limit=limit)

    def search_audit(self) -> Mapping[str, Any] | None:
        return None
