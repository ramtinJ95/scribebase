from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from scribebase.config import ChunkingConfig
from scribebase.markdown.normalize import PAGE_MARKER_RE
from scribebase.models import Chunk, PageMetadata, SourceManifest
from scribebase.source_registry import slugify

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def chunk_markdown(
    markdown_path: Path,
    manifest: SourceManifest,
    pages: list[PageMetadata],
    config: ChunkingConfig | None = None,
) -> list[Chunk]:
    config = config or ChunkingConfig()
    text = markdown_path.read_text()
    units = _units(text)
    chunks: list[Chunk] = []
    current = ""
    current_pages: list[int] = []
    current_section: str | None = None
    page_methods = {page.page_number: page.extraction_method for page in pages}
    page_ocr_models = {page.page_number: page.ocr_model for page in pages if page.ocr_model}

    for unit_text, page, section in units:
        if section:
            current_section = section
        if len(unit_text) > config.target_chars:
            if current.strip():
                chunks.append(
                    _build_chunk(
                        current,
                        current_pages,
                        current_section,
                        markdown_path,
                        manifest,
                        page_methods,
                        page_ocr_models,
                        len(chunks),
                        config.chunker_version,
                    )
                )
                current = ""
                current_pages = []
            for split_text in _split_long_text(unit_text, config.target_chars, config.overlap_chars):
                chunks.append(
                    _build_chunk(
                        split_text,
                        [page] if page else [],
                        current_section,
                        markdown_path,
                        manifest,
                        page_methods,
                        page_ocr_models,
                        len(chunks),
                        config.chunker_version,
                    )
                )
            continue
        projected = f"{current}\n\n{unit_text}".strip() if current else unit_text.strip()
        if current and len(projected) > config.target_chars and len(current) >= config.min_chars:
            chunks.append(
                _build_chunk(
                    current,
                    current_pages,
                    current_section,
                    markdown_path,
                    manifest,
                    page_methods,
                    page_ocr_models,
                    len(chunks),
                    config.chunker_version,
                )
            )
            overlap = current[-config.overlap_chars :].strip() if config.overlap_chars else ""
            current = f"{overlap}\n\n{unit_text}".strip() if overlap else unit_text.strip()
            current_pages = ([current_pages[-1]] if current_pages and overlap else []) + ([page] if page else [])
        else:
            current = projected
            if page:
                current_pages.append(page)

    if current.strip():
        chunks.append(
            _build_chunk(
                current,
                current_pages,
                current_section,
                markdown_path,
                manifest,
                page_methods,
                page_ocr_models,
                len(chunks),
                config.chunker_version,
            )
        )
    return chunks


def _units(text: str) -> list[tuple[str, int | None, str | None]]:
    parts = re.split(r"(\n\s*\n)", text)
    units: list[tuple[str, int | None, str | None]] = []
    page: int | None = None
    section: str | None = None
    buffer = ""
    for part in parts:
        if not part.strip():
            if buffer.strip():
                units.append((buffer.strip(), page, section))
                buffer = ""
            continue
        marker = PAGE_MARKER_RE.search(part)
        if marker:
            page = int(marker.group(1))
        heading = HEADING_RE.search(part)
        heading_text = heading.group(2).strip() if heading else None
        if heading_text and not heading_text.lower().startswith("page "):
            section = heading_text
        buffer = f"{buffer}\n{part}".strip() if buffer else part
    if buffer.strip():
        units.append((buffer.strip(), page, section))
    return units


def _split_long_text(text: str, target_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= target_chars:
        return [text.strip()]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + target_chars, len(text))
        if end < len(text):
            boundary = max(
                text.rfind("\n", start, end),
                text.rfind(". ", start, end),
                text.rfind(" ", start, end),
            )
            if boundary > start + max(200, target_chars // 2):
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap_chars, start + 1)
    return chunks


def _build_chunk(
    text: str,
    pages: list[int],
    section: str | None,
    markdown_path: Path,
    manifest: SourceManifest,
    page_methods: dict[int, str],
    page_ocr_models: dict[int, str],
    chunk_index: int,
    chunker_version: str,
) -> Chunk:
    unique_pages = sorted({p for p in pages if p is not None})
    page_start = unique_pages[0] if unique_pages else None
    page_end = unique_pages[-1] if unique_pages else None
    methods = {page_methods.get(p, "unknown") for p in unique_pages}
    extraction_method = next(iter(methods)) if len(methods) == 1 else "mixed"
    ocr_models = {page_ocr_models[p] for p in unique_pages if p in page_ocr_models}
    chapter_part = f"ch{manifest.chapter}" if manifest.chapter else "doc"
    page_part = f"p{page_start:04d}" if page_start else "p0000"
    chunk_id = f"{manifest.source_id}_{slugify(chapter_part)}_{page_part}_{chunk_index:04d}"
    return Chunk(
        chunk_id=chunk_id,
        source_id=manifest.source_id,
        source_type=manifest.source_type,
        title=manifest.title,
        course=manifest.course,
        chapter=manifest.chapter,
        section=section,
        page_start=page_start,
        page_end=page_end,
        chunk_index=chunk_index,
        text=text.strip(),
        file_path=str(markdown_path),
        extraction_method=extraction_method,
        ocr_model=next(iter(ocr_models)) if len(ocr_models) == 1 else ("mixed" if ocr_models else None),
        language=manifest.language,
        chunker_version=chunker_version,
        created_at=datetime.now(timezone.utc),
    )
