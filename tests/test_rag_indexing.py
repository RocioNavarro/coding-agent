"""Indexación RAG genérica, incremental y persistente."""

import json
from pathlib import Path

from rag.cli import index_manifest, main
from rag.embeddings import HashEmbeddingProvider
from rag.index_manager import IndexManager
from rag.models import SourceConfig
from rag.processing import ConfigurableChunker, SectionDocumentParser, WhitespaceNormalizer
from rag.sources import ConfiguredSourceLoader, SourceManifest
from rag.vector_store import JsonVectorStore


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_manager(
    index_path: Path,
    *,
    fetcher=None,
    max_characters: int = 80,
    overlap: int = 10,
) -> tuple[IndexManager, JsonVectorStore]:
    store = JsonVectorStore(index_path)
    return (
        IndexManager(
            source_loader=ConfiguredSourceLoader(fetcher),
            parsers={"sections": SectionDocumentParser(), "plain": SectionDocumentParser()},
            normalizer=WhitespaceNormalizer(),
            chunker=ConfigurableChunker(
                max_characters=max_characters,
                overlap_characters=overlap,
                respect_sections=True,
            ),
            embedding_provider=HashEmbeddingProvider(32),
            vector_store=store,
        ),
        store,
    )


def test_loads_local_documentation_readmes_text_and_configured_url(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    write(docs / "README", "Project overview")
    write(docs / "guide.md", "# Guide\nConfigured guide")
    write(docs / "notes.txt", "Downloaded documentation stored locally")
    write(docs / "ignored.bin", "not selected")
    sources = (
        SourceConfig(
            "workspace-docs", "local", "local_documentation", str(docs),
            patterns=("README*", "**/*.md", "**/*.txt"),
        ),
        SourceConfig(
            "official", "url", "official_documentation",
            "https://docs.example.invalid/reference", tags=("configured",),
        ),
    )
    loader = ConfiguredSourceLoader(
        lambda url, encoding: "# Official\nReference downloaded by configured provider"
    )

    documents = loader.load(sources)

    assert len(documents) == 4
    assert {Path(item.path_or_url).name for item in documents if item.path_or_url.startswith("/")} == {
        "README", "guide.md", "notes.txt"
    }
    assert any(item.path_or_url.startswith("https://") for item in documents)
    assert {item.source.source_type for item in documents} == {
        "local_documentation", "official_documentation"
    }


def test_chunking_respects_sections_and_configured_size(tmp_path: Path) -> None:
    document = tmp_path / "guide.md"
    write(
        document,
        "# First\n" + "alpha " * 20 + "\n\n# Second\n" + "beta " * 20,
    )
    manager, store = build_manager(tmp_path / "index.json", max_characters=45, overlap=5)

    result = manager.index(
        (SourceConfig("guide", "local", "markdown", str(document)),)
    )
    chunks = store.all_chunks()

    assert result.chunks_written >= 4
    assert {chunk.metadata.section for chunk in chunks} == {"First", "Second"}
    assert all(len(chunk.content) <= 45 for chunk in chunks)
    assert all(chunk.metadata.chunk_index >= 0 for chunk in chunks)


def test_chunks_include_complete_auditable_metadata(tmp_path: Path) -> None:
    document = tmp_path / "reference.txt"
    write(document, "# API\nStable reference content")
    manager, store = build_manager(tmp_path / "index.json")
    source = SourceConfig(
        "reference-project", "local", "configured_example", str(document),
        title="Reference", detected_language="configured-language",
        tags=("example", "user-provided"),
    )

    manager.index((source,))
    metadata = store.all_chunks()[0].metadata

    assert metadata.document_id
    assert metadata.source_name == "reference-project"
    assert metadata.source_type == "configured_example"
    assert metadata.path_or_url == document.resolve().as_posix()
    assert metadata.title == "Reference"
    assert metadata.section == "API"
    assert metadata.detected_language == "configured-language"
    assert metadata.tags == ("example", "user-provided")
    assert metadata.content_hash
    assert metadata.indexed_at.endswith("+00:00")


def test_duplicate_sources_and_duplicate_chunks_are_not_indexed_twice(tmp_path: Path) -> None:
    document = tmp_path / "duplicate.txt"
    write(document, "# One\nsame content\n# Two\nsame content")
    manager, store = build_manager(tmp_path / "index.json", max_characters=100, overlap=0)
    source = SourceConfig("same", "local", "text", str(document))

    first = manager.index((source, source))
    second = manager.index((source, source))

    assert first.loaded_documents == 1
    assert len(store.all_chunks()) == 1
    assert second.indexed_documents == 0
    assert second.skipped_documents == 1
    assert second.chunks_written == 0


def test_incremental_update_replaces_only_changed_document(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    write(first, "first version")
    write(second, "stable document")
    sources = (
        SourceConfig("first", "local", "text", str(first)),
        SourceConfig("second", "local", "text", str(second)),
    )
    manager, store = build_manager(tmp_path / "index.json")
    initial = manager.index(sources)
    stable_id = next(
        chunk.chunk_id for chunk in store.all_chunks()
        if chunk.metadata.source_name == "second"
    )
    write(first, "second version with changed content")

    updated = manager.index(sources)

    assert initial.indexed_documents == 2
    assert updated.updated_documents == 1
    assert updated.skipped_documents == 1
    assert stable_id in {chunk.chunk_id for chunk in store.all_chunks()}
    assert any("second version" in chunk.content for chunk in store.all_chunks())
    assert not any("first version" in chunk.content for chunk in store.all_chunks())


def test_vector_index_persists_and_manifest_cli_reuses_it(
    tmp_path: Path, capsys
) -> None:
    write(tmp_path / "docs" / "README.md", "# Persistent\nIndex content")
    manifest_path = tmp_path / "rag-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "index_path": "data/index.json",
                "embedding": {"provider": "hash", "dimensions": 24},
                "chunking": {
                    "max_characters": 100,
                    "overlap_characters": 5,
                    "respect_sections": True,
                },
                "sources": [
                    {
                        "name": "docs",
                        "loader": "local",
                        "source_type": "documentation",
                        "path": "docs",
                        "patterns": ["**/*.md"],
                        "tags": ["configured"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    first = index_manifest(str(manifest_path))
    exit_code = main([str(manifest_path)])
    reloaded = JsonVectorStore(tmp_path / "data" / "index.json")
    reloaded.load()

    assert first.indexed_documents == 1
    assert exit_code == 0
    assert "0 nuevos" in capsys.readouterr().out
    assert len(reloaded.all_chunks()) == 1
    persisted = json.loads((tmp_path / "data" / "index.json").read_text(encoding="utf-8"))
    assert persisted["schema_version"] == 1
    assert persisted["documents"]
