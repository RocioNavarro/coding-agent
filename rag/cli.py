"""CLI independiente para construir un índice RAG desde un manifiesto."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from rag.embeddings import HashEmbeddingProvider
from rag.index_manager import IndexManager, IndexingResult
from rag.processing import (
    ConfigurableChunker,
    HtmlDocumentParser,
    SectionDocumentParser,
    WhitespaceNormalizer,
)
from rag.sources import ConfiguredSourceLoader, SourceManifest
from rag.vector_store import JsonVectorStore


def index_manifest(path: str, *, prune: bool = False) -> IndexingResult:
    manifest = SourceManifest.load(path)
    provider = manifest.embedding.get("provider", "hash")
    if provider != "hash":
        raise ValueError(f"Embedding provider no configurado en esta instalación: {provider}")
    dimensions = int(manifest.embedding.get("dimensions", 128))
    chunking = manifest.chunking
    manager = IndexManager(
        source_loader=ConfiguredSourceLoader(),
        parsers={
            "sections": SectionDocumentParser(),
            "plain": SectionDocumentParser(),
            "html": HtmlDocumentParser(),
        },
        normalizer=WhitespaceNormalizer(),
        chunker=ConfigurableChunker(
            max_characters=int(chunking.get("max_characters", 1_500)),
            overlap_characters=int(chunking.get("overlap_characters", 150)),
            respect_sections=bool(chunking.get("respect_sections", True)),
        ),
        embedding_provider=HashEmbeddingProvider(dimensions),
        vector_store=JsonVectorStore(manifest.resolved_index_path()),
    )
    return manager.index(manifest.resolved_sources(), prune=prune)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Indexa fuentes RAG configuradas.")
    parser.add_argument("manifest", help="Ruta al manifiesto JSON de fuentes.")
    parser.add_argument(
        "--prune", action="store_true",
        help="Elimina del índice documentos que ya no figuran en el manifiesto.",
    )
    arguments = parser.parse_args(argv)
    try:
        result = index_manifest(arguments.manifest, prune=arguments.prune)
    except Exception as error:
        parser.exit(1, f"Error de indexación: {error}\n")
    print(
        "Indexación completada: "
        f"{result.indexed_documents} nuevos, {result.updated_documents} actualizados, "
        f"{result.skipped_documents} sin cambios, {result.chunks_written} chunks."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
