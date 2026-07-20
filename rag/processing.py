"""Parsing, normalización y chunking configurables de documentos."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from html.parser import HTMLParser
from pathlib import Path
from rag.models import ChunkDraft, DocumentSection, ParsedDocument, RawDocument


class DocumentParser(ABC):
    def fingerprint(self) -> str:
        return f"{type(self).__module__}.{type(self).__qualname__}"

    @abstractmethod
    def parse(self, document: RawDocument) -> ParsedDocument:
        ...


class SectionDocumentParser(DocumentParser):
    """Reconoce encabezados Markdown cuando existen y conserva secciones."""

    _HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$")

    def parse(self, document: RawDocument) -> ParsedDocument:
        title = document.source.title
        sections: list[DocumentSection] = []
        current_title = "document"
        current_lines: list[str] = []
        for line in document.content.splitlines():
            heading = self._HEADING.match(line)
            if heading:
                if current_lines:
                    sections.append(DocumentSection(current_title, "\n".join(current_lines)))
                    current_lines = []
                current_title = heading.group(1).strip()
                if title is None:
                    title = current_title
                continue
            current_lines.append(line)
        if current_lines or not sections:
            sections.append(DocumentSection(current_title, "\n".join(current_lines)))
        resolved_title = title or Path(document.path_or_url).name or document.source.name
        return ParsedDocument(document, resolved_title, tuple(sections))


class _HtmlTextExtractor(HTMLParser):
    """Agrupa el texto de un HTML en secciones delimitadas por <h1>-<h6>."""

    _HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
    _SKIP_TAGS = frozenset({"script", "style"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sections: list[DocumentSection] = []
        self._title = "document"
        self._buffer: list[str] = []
        self._skip_depth = 0
        self._in_heading = False
        self._heading_buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        elif tag in self._HEADING_TAGS:
            self._flush_section()
            self._in_heading = True
            self._heading_buffer = []

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in self._HEADING_TAGS and self._in_heading:
            self._title = "".join(self._heading_buffer).strip() or self._title
            self._in_heading = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        (self._heading_buffer if self._in_heading else self._buffer).append(data)

    def _flush_section(self) -> None:
        content = "".join(self._buffer).strip()
        if content:
            self.sections.append(DocumentSection(self._title, content))
        self._buffer = []

    def close(self) -> None:
        super().close()
        self._flush_section()


class HtmlDocumentParser(DocumentParser):
    """Extrae texto plano de HTML, sin scripts/estilos, agrupado por encabezados."""

    def parse(self, document: RawDocument) -> ParsedDocument:
        extractor = _HtmlTextExtractor()
        extractor.feed(document.content)
        extractor.close()
        sections = tuple(extractor.sections) or (DocumentSection("document", ""),)
        resolved_title = (
            document.source.title
            or sections[0].title
            or Path(document.path_or_url).name
            or document.source.name
        )
        return ParsedDocument(document, resolved_title, sections)


class TextNormalizer(ABC):
    def fingerprint(self) -> str:
        return f"{type(self).__module__}.{type(self).__qualname__}"

    @abstractmethod
    def normalize(self, text: str) -> str:
        ...


class WhitespaceNormalizer(TextNormalizer):
    """Normaliza saltos/espacios preservando párrafos y contenido textual."""

    def normalize(self, text: str) -> str:
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.replace("\r\n", "\n").split("\n")]
        result: list[str] = []
        blank = False
        for line in lines:
            if not line:
                if result and not blank:
                    result.append("")
                blank = True
            else:
                result.append(line)
                blank = False
        return "\n".join(result).strip()


class Chunker(ABC):
    def fingerprint(self) -> str:
        return f"{type(self).__module__}.{type(self).__qualname__}"

    @abstractmethod
    def chunk(self, document: ParsedDocument, normalizer: TextNormalizer) -> tuple[ChunkDraft, ...]:
        ...


class ConfigurableChunker(Chunker):
    """Divide por secciones/párrafos con límites y solapamiento configurables."""

    def __init__(
        self,
        *,
        max_characters: int = 1_500,
        overlap_characters: int = 150,
        respect_sections: bool = True,
    ) -> None:
        if max_characters < 1:
            raise ValueError("max_characters debe ser positivo.")
        if not 0 <= overlap_characters < max_characters:
            raise ValueError("overlap_characters debe estar entre 0 y max_characters.")
        self.max_characters = max_characters
        self.overlap_characters = overlap_characters
        self.respect_sections = respect_sections

    def fingerprint(self) -> str:
        return (
            f"{super().fingerprint()}:{self.max_characters}:"
            f"{self.overlap_characters}:{self.respect_sections}"
        )

    def chunk(
        self, document: ParsedDocument, normalizer: TextNormalizer
    ) -> tuple[ChunkDraft, ...]:
        sections = document.sections
        if not self.respect_sections:
            combined = "\n\n".join(section.content for section in sections)
            sections = (DocumentSection("document", combined),)
        drafts: list[ChunkDraft] = []
        for section in sections:
            normalized = normalizer.normalize(section.content)
            if not normalized:
                continue
            for content in self._split(normalized):
                drafts.append(
                    ChunkDraft(
                        document.raw.document_id,
                        document.raw.source.name,
                        document.raw.source.source_type,
                        document.raw.path_or_url,
                        document.title,
                        section.title,
                        document.raw.source.detected_language,
                        document.raw.source.tags,
                        len(drafts),
                        content,
                    )
                )
        return tuple(drafts)

    def _split(self, text: str) -> tuple[str, ...]:
        chunks: list[str] = []
        cursor = 0
        while cursor < len(text):
            end = min(len(text), cursor + self.max_characters)
            if end < len(text):
                paragraph = text.rfind("\n\n", cursor, end)
                space = text.rfind(" ", cursor, end)
                boundary = paragraph if paragraph > cursor else space
                if boundary > cursor:
                    end = boundary
            content = text[cursor:end].strip()
            if content and (not chunks or content != chunks[-1]):
                chunks.append(content)
            if end >= len(text):
                break
            next_cursor = end - self.overlap_characters
            cursor = next_cursor if next_cursor > cursor else end
        return tuple(chunks)
