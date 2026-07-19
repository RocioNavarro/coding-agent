"""Pipeline RAG local, configurable e independiente del ecosistema."""

from rag.embeddings import EmbeddingProvider, HashEmbeddingProvider
from rag.index_manager import IndexManager, IndexingResult
from rag.models import ChunkMetadata, DocumentChunk, RawDocument, SourceConfig
from rag.processing import (
    Chunker,
    ConfigurableChunker,
    DocumentParser,
    SectionDocumentParser,
    TextNormalizer,
    WhitespaceNormalizer,
)
from rag.sources import ConfiguredSourceLoader, SourceLoader, SourceManifest
from rag.vector_store import JsonVectorStore, VectorStore

__all__ = [
    "ChunkMetadata",
    "Chunker",
    "ConfigurableChunker",
    "ConfiguredSourceLoader",
    "DocumentChunk",
    "DocumentParser",
    "EmbeddingProvider",
    "HashEmbeddingProvider",
    "IndexManager",
    "IndexingResult",
    "JsonVectorStore",
    "RawDocument",
    "SectionDocumentParser",
    "SourceConfig",
    "SourceLoader",
    "SourceManifest",
    "TextNormalizer",
    "VectorStore",
    "WhitespaceNormalizer",
]
