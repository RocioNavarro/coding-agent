"""Recuperación RAG con filtros, umbral, deduplicación y auditoría."""

from collections.abc import Sequence
from pathlib import Path

from rag.embeddings import EmbeddingProvider
from rag.models import ChunkMetadata, DocumentChunk
from rag.retriever import RagRetriever
from rag.vector_store import JsonVectorStore
from core.observability import ObservabilityEvent


class FakeObservability:
    def __init__(self) -> None:
        self.events: list[ObservabilityEvent] = []

    def record(self, event: ObservabilityEvent) -> None:
        self.events.append(event)

    def flush(self) -> None:
        return None


class QueryEmbedding(EmbeddingProvider):
    def __init__(self, vectors: dict[str, tuple[float, ...]]) -> None:
        self.vectors = vectors
        self.queries: list[str] = []

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        self.queries.extend(texts)
        return tuple(self.vectors[text] for text in texts)


def chunk(
    chunk_id: str,
    document_id: str,
    content: str,
    vector: tuple[float, ...],
    *,
    source_type: str = "documentation",
    language: str | None = None,
    tags: tuple[str, ...] = (),
    content_hash: str | None = None,
    source_name: str | None = None,
    section: str = "section",
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id,
        content,
        vector,
        ChunkMetadata(
            document_id=document_id,
            source_name=source_name or f"source-{document_id}",
            source_type=source_type,
            path_or_url=f"docs/{document_id}.txt",
            title=f"Document {document_id}",
            section=section,
            detected_language=language,
            tags=tags,
            chunk_index=0,
            content_hash=content_hash or f"hash-{chunk_id}",
            indexed_at="2026-01-01T00:00:00+00:00",
        ),
    )


def store_with(tmp_path: Path, *chunks: DocumentChunk) -> JsonVectorStore:
    store = JsonVectorStore(tmp_path / "index.json")
    for item in chunks:
        store.replace_document(
            item.metadata.document_id,
            f"document-hash-{item.metadata.document_id}",
            (item,),
        )
    store.save()
    return store


def test_retrieves_top_k_with_scores_and_metadata(tmp_path: Path) -> None:
    store = store_with(
        tmp_path,
        chunk("a", "doc-a", "Highly relevant", (1.0, 0.0)),
        chunk("b", "doc-b", "Partially relevant", (0.8, 0.6)),
        chunk("c", "doc-c", "Not relevant", (0.0, 1.0)),
    )
    embeddings = QueryEmbedding({"target query": (1.0, 0.0)})
    retriever = RagRetriever(
        embedding_provider=embeddings,
        vector_store=store,
        top_k=2,
        relevance_threshold=0.1,
    )

    evidence = retriever.retrieve("target query", limit=2)

    assert [item.content for item in evidence] == ["Highly relevant", "Partially relevant"]
    assert [item.relevance for item in evidence] == [1.0, 0.8]
    assert evidence[0].reference == "rag://doc-a/0"
    assert embeddings.queries == ["target query"]


def test_records_compact_rag_observation(tmp_path: Path) -> None:
    observed = FakeObservability()
    retriever = RagRetriever(
        embedding_provider=QueryEmbedding({"query": (1.0, 0.0)}),
        vector_store=store_with(
            tmp_path, chunk("chunk-a", "doc-a", "full content", (1.0, 0.0))
        ),
        observability=observed,
    )

    retriever.retrieve_context("query", top_k=1)

    assert len(observed.events) == 1
    payload = observed.events[0].payload
    assert payload["retrieved_chunks"] == ["chunk-a"]
    assert payload["documents"] == ("doc-a",)
    assert "full content" not in str(payload)


def test_applies_dynamic_metadata_and_tag_filters(tmp_path: Path) -> None:
    matching = chunk(
        "match", "doc-match", "Filtered evidence", (1.0, 0.0),
        source_type="official", language="ConfiguredLang",
        tags=("technology:runtime-x", "framework:framework-y", "module:billing", "stable"),
    )
    other = chunk(
        "other", "doc-other", "Other evidence", (1.0, 0.0),
        source_type="community", language="OtherLang", tags=("module:other",),
    )
    retriever = RagRetriever(
        embedding_provider=QueryEmbedding({"query": (1.0, 0.0)}),
        vector_store=store_with(tmp_path, matching, other),
        relevance_threshold=0.0,
    )

    trace = retriever.retrieve_context(
        "query",
        filters={
            "technology": "runtime-x",
            "language": "ConfiguredLang",
            "framework": "framework-y",
            "source_type": "official",
            "module": "billing",
            "tags": "stable",
        },
    )

    assert [item.chunk_id for item in trace.used_chunks] == ["match"]
    assert trace.filters["framework"] == ("framework-y",)


def test_relevance_threshold_excludes_low_score_chunks(tmp_path: Path) -> None:
    retriever = RagRetriever(
        embedding_provider=QueryEmbedding({"query": (1.0, 0.0)}),
        vector_store=store_with(
            tmp_path,
            chunk("high", "doc-high", "above", (0.9, 0.1)),
            chunk("low", "doc-low", "below", (0.1, 0.9)),
        ),
        relevance_threshold=0.7,
    )

    trace = retriever.retrieve_context("query")

    assert [item.chunk_id for item in trace.retrieved_chunks] == ["high"]
    assert all(item.score >= 0.7 for item in trace.used_chunks)


def test_deduplicates_equal_content_hashes_after_retrieval(tmp_path: Path) -> None:
    retriever = RagRetriever(
        embedding_provider=QueryEmbedding({"query": (1.0, 0.0)}),
        vector_store=store_with(
            tmp_path,
            chunk("best", "doc-a", "duplicated", (1.0, 0.0), content_hash="same"),
            chunk("copy", "doc-b", "duplicated", (0.9, 0.1), content_hash="same"),
        ),
        relevance_threshold=0.0,
    )

    trace = retriever.retrieve_context("query", top_k=5)

    assert len(trace.retrieved_chunks) == 2
    assert [item.chunk_id for item in trace.used_chunks] == ["best"]
    assert trace.documents == ("doc-a",)


def test_returns_insufficient_trace_when_no_results_pass(tmp_path: Path) -> None:
    retriever = RagRetriever(
        embedding_provider=QueryEmbedding({"query": (1.0, 0.0)}),
        vector_store=store_with(
            tmp_path, chunk("orthogonal", "doc", "unrelated", (0.0, 1.0))
        ),
        relevance_threshold=0.5,
        min_chunks_for_sufficiency=2,
    )

    evidence = retriever.retrieve("query")
    trace = retriever.last_trace

    assert evidence == ()
    assert trace is not None
    assert trace.sufficiency.sufficient is False
    assert trace.sufficiency.confidence == 0.0
    assert "No se recuperaron" in trace.conclusions[0]


def test_audit_differentiates_retrieved_used_and_inferred_conclusions(tmp_path: Path) -> None:
    retriever = RagRetriever(
        embedding_provider=QueryEmbedding({"query": (1.0, 0.0)}),
        vector_store=store_with(
            tmp_path,
            chunk("a", "doc-a", "first", (1.0, 0.0)),
            chunk("b", "doc-b", "second", (0.8, 0.2)),
        ),
        top_k=1,
        relevance_threshold=0.0,
    )

    retriever.retrieve("query", limit=1)
    audit = retriever.retrieval_audit()

    assert audit is not None
    assert len(audit["retrieved_chunks"]) == 2
    assert len(audit["used_chunks"]) == 1
    assert set(audit["scores"]) == {"a", "b"}
    assert audit["documents"] == ["doc-a"]
    assert audit["conclusions"]


def test_low_scores_from_same_source_are_discarded_and_insufficient(
    tmp_path: Path,
) -> None:
    retriever = RagRetriever(
        embedding_provider=QueryEmbedding({"architecture modules": (1.0, 0.0)}),
        vector_store=store_with(
            tmp_path,
            chunk(
                "a", "doc-a", "Kotlin variables", (0.38, 0.925),
                source_name="kotlin-basic-syntax", section="Variables",
            ),
            chunk(
                "b", "doc-b", "More Kotlin variables", (0.32, 0.947),
                source_name="kotlin-basic-syntax", section="Variables",
            ),
        ),
    )

    trace = retriever.retrieve_context("architecture modules")

    assert len(trace.retrieved_chunks) == 2
    assert trace.used_chunks == ()
    assert {item.chunk_id for item in trace.discarded_chunks} == {"a", "b"}
    assert trace.sufficiency.sufficient is False


def test_fragment_preserves_rag_metadata_and_query(tmp_path: Path) -> None:
    retriever = RagRetriever(
        embedding_provider=QueryEmbedding({"PrintScript specification": (1.0, 0.0)}),
        vector_store=store_with(
            tmp_path,
            chunk(
                "spec", "doc-spec", "PrintScript processing architecture", (1.0, 0.0),
                source_name="printscript-language-spec",
                section="Architecture",
                tags=("language-spec",),
            ),
        ),
    )

    fragment = retriever.retrieve("PrintScript specification")[0]

    assert fragment.metadata["source_name"] == "printscript-language-spec"
    assert fragment.metadata["section"] == "Architecture"
    assert fragment.metadata["tags"] == ["language-spec"]
    assert fragment.metadata["query"] == "PrintScript specification"
