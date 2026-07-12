from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypeVar

from scribebase.config import AppConfig
from scribebase.extractors.image_renderer import render_pdf_page
from scribebase.extractors.pymupdf_extractor import (
    extract_page_markdown,
    extract_page_text,
    page_has_images,
    pdf_page_count,
)
from scribebase.markdown.frontmatter import read_markdown_with_frontmatter
from scribebase.markdown.normalize import combine_pages, normalize_page_markdown
from scribebase.models import PageMetadata, SourceManifest, SourceMetadataInput, TextQuality
from scribebase.ocr.shell_provider import ShellOCRProvider
from scribebase.paths import chapter_file_name, source_subdirs
from scribebase.pdf_router import evaluate_text_quality
from scribebase.source_registry import create_manifest, write_manifest

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}
MARKDOWN_EXTS = {".md", ".markdown"}
TEXT_EXTS = {".txt"}
T = TypeVar("T")


@dataclass(frozen=True)
class PDFPageRoute:
    raw_text: str
    quality: TextQuality
    has_images: bool


def extract_source(
    input_path: Path,
    title: str | None,
    source_type: str | None,
    course: str | None,
    chapter: str | None,
    language: str | None,
    ocr: str,
    config: AppConfig,
    logger,
    continue_on_ocr_error: bool = False,
    tags: list[str] | str | None = None,
    origin: str | None = None,
    publisher: str | None = None,
    author: str | None = None,
    created_at_source: datetime | str | None = None,
    updated_at_source: datetime | str | None = None,
    retrieved_at: datetime | str | None = None,
    url: str | None = None,
    canonical_url: str | None = None,
    external_id: str | None = None,
    collection: str | None = None,
    summary: str | None = None,
    source_id: str | None = None,
) -> SourceManifest:
    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    frontmatter = _frontmatter_metadata(input_path)
    title = _resolve_field(title, frontmatter.title)
    if not title:
        raise ValueError("title is required unless provided by Markdown frontmatter")
    source_type = _resolve_field(source_type, frontmatter.source_type) or "other"
    course = _resolve_field(course, frontmatter.course)
    chapter = _resolve_field(chapter, frontmatter.chapter)
    language = _resolve_field(language, frontmatter.language) or "unknown"
    manifest = create_manifest(
        config.data_dir,
        input_path,
        title,
        source_type,
        course,
        chapter,
        language,
        tags=tags if tags is not None else frontmatter.tags,
        origin=_resolve_field(origin, frontmatter.origin),
        publisher=_resolve_field(publisher, frontmatter.publisher),
        author=_resolve_field(author, frontmatter.author),
        created_at_source=_resolve_field(created_at_source, frontmatter.created_at_source),
        updated_at_source=_resolve_field(updated_at_source, frontmatter.updated_at_source),
        retrieved_at=_resolve_field(retrieved_at, frontmatter.retrieved_at),
        url=_resolve_field(url, frontmatter.url),
        canonical_url=_resolve_field(canonical_url, frontmatter.canonical_url),
        external_id=_resolve_field(external_id, frontmatter.external_id),
        collection=_resolve_field(collection, frontmatter.collection),
        summary=_resolve_field(summary, frontmatter.summary),
        source_id=source_id,
    )
    logger.info("Ingest source: %s (%s)", manifest.title, manifest.source_id)
    paths = source_subdirs(config.data_dir, manifest.source_id)
    original = Path(manifest.original_path)
    if original.is_dir():
        pages = _extract_images(original, manifest, ocr, config, logger, continue_on_ocr_error)
    elif original.suffix.lower() == ".pdf":
        pages = _extract_pdf(original, manifest, ocr, config, logger, continue_on_ocr_error)
    elif original.suffix.lower() in IMAGE_EXTS:
        pages = _extract_images(original, manifest, ocr, config, logger, continue_on_ocr_error)
    elif original.suffix.lower() in MARKDOWN_EXTS:
        pages = _extract_text_document(original, manifest, config, logger, input_type="markdown")
    elif original.suffix.lower() in TEXT_EXTS:
        pages = _extract_text_document(original, manifest, config, logger, input_type="text")
    else:
        raise ValueError(f"Unsupported input type: {original}")

    page_markdowns = [
        Path(page.markdown_path).read_text() for page in pages if Path(page.markdown_path).exists()
    ]
    document_path = paths["markdown"] / "document.md"
    document_path.write_text(combine_pages(page_markdowns))
    if chapter:
        chapter_path = paths["chapters"] / chapter_file_name(chapter)
        chapter_path.write_text(document_path.read_text())

    manifest.extraction_summary.pages_total = len(pages)
    manifest.extraction_summary.pages_extracted_with_pymupdf4llm = sum(
        1 for page in pages if page.extraction_method in {"pymupdf4llm", "pymupdf"}
    )
    manifest.extraction_summary.pages_ocr = sum(
        1 for page in pages if page.extraction_method == "ocr"
    )
    manifest.extraction_summary.ocr_provider = config.ocr.default_provider
    provider_cfg = config.ocr.providers.get(config.ocr.default_provider)
    manifest.extraction_summary.ocr_model = provider_cfg.model_name if provider_cfg else None
    manifest.updated_at = datetime.now(timezone.utc)
    write_manifest(manifest)
    if not document_path.read_text().strip():
        raise RuntimeError(
            "Empty extraction. Try --ocr always or check OCR provider configuration."
        )
    return manifest


def _extract_pdf(
    pdf_path: Path,
    manifest: SourceManifest,
    ocr: str,
    config: AppConfig,
    logger,
    continue_on_ocr_error: bool,
) -> list[PageMetadata]:
    paths = source_subdirs(config.data_dir, manifest.source_id)
    provider: ShellOCRProvider | None = None
    pages: list[PageMetadata] = []
    routes = _pdf_page_routes(pdf_path, config)
    likely_true_text_pdf = _likely_true_text_pdf(routes, config)
    if ocr == "auto":
        logger.info(
            "PDF auto route: %s pages, %s text-layer pages, true_text_pdf=%s",
            len(routes),
            sum(1 for route in routes if _has_usable_text_layer(route.quality, config)),
            likely_true_text_pdf,
        )
    for page_index, route in enumerate(routes):
        page_number = page_index + 1
        md_path = paths["markdown"] / f"page_{page_number:04d}.md"
        image_path = paths["pages"] / f"page_{page_number:04d}.png"
        use_ocr = _should_ocr_pdf_page(ocr, route, likely_true_text_pdf)
        if not use_ocr:
            logger.info("Page %s: using PyMuPDF4LLM", page_number)
            meta = _extract_text_pdf_page(
                pdf_path,
                page_index,
                md_path,
                manifest,
                route.quality,
            )
        else:
            if ocr == "never":
                raise RuntimeError(f"Page {page_number} has insufficient text and OCR is disabled")
            provider = provider or _ocr_provider(ocr, config)
            render_dpi = provider.config.render_dpi or config.ocr.render_dpi
            logger.info("Page %s: insufficient text, rendering at %s DPI", page_number, render_dpi)
            render_pdf_page(pdf_path, page_index, image_path, render_dpi)
            meta = _run_ocr_page(
                image_path,
                md_path,
                manifest,
                page_number,
                page_index,
                "pdf_page",
                provider,
                logger,
                continue_on_ocr_error,
                route.quality.flags,
            )
        _write_page_metadata(paths["metadata"], meta)
        pages.append(meta)
    return pages


def _pdf_page_routes(pdf_path: Path, config: AppConfig) -> list[PDFPageRoute]:
    routes = []
    for page_index in range(pdf_page_count(pdf_path)):
        raw_text = extract_page_text(pdf_path, page_index)
        routes.append(
            PDFPageRoute(
                raw_text=raw_text,
                quality=evaluate_text_quality(raw_text, config.pdf_detection),
                has_images=page_has_images(pdf_path, page_index),
            )
        )
    return routes


def _likely_true_text_pdf(routes: list[PDFPageRoute], config: AppConfig) -> bool:
    if not routes:
        return False
    text_layer_pages = sum(1 for route in routes if _has_usable_text_layer(route.quality, config))
    true_text_pages = sum(1 for route in routes if route.quality.is_true_text)
    return true_text_pages > 0 and (text_layer_pages / len(routes) >= 0.5 or true_text_pages >= 3)


def _has_usable_text_layer(quality: TextQuality, config: AppConfig) -> bool:
    min_chars = max(20, config.pdf_detection.min_chars_per_page // 4)
    return (
        quality.char_count >= min_chars
        and quality.alpha_ratio >= config.pdf_detection.min_alpha_ratio
        and "replacement_chars" not in quality.flags
        and "long_average_word" not in quality.flags
    )


def _should_ocr_pdf_page(ocr: str, route: PDFPageRoute, likely_true_text_pdf: bool) -> bool:
    if ocr == "always":
        return True
    if ocr == "never":
        return False
    if ocr != "auto":
        return not route.quality.is_true_text
    if route.quality.is_true_text:
        return False
    if likely_true_text_pdf and not route.has_images:
        return False
    if route.quality.char_count > 0 and not route.has_images:
        return False
    return route.has_images


def _extract_text_pdf_page(
    pdf_path: Path,
    page_index: int,
    md_path: Path,
    manifest: SourceManifest,
    quality: TextQuality,
) -> PageMetadata:
    page_number = page_index + 1
    page_md, method = extract_page_markdown(pdf_path, page_index)
    text = normalize_page_markdown(page_md, page_number)
    md_path.write_text(text)
    return PageMetadata(
        source_id=manifest.source_id,
        page_number=page_number,
        page_index=page_index,
        input_type="pdf_page",
        text_layer_detected=quality.is_true_text,
        extraction_method=method,  # type: ignore[arg-type]
        image_path=None,
        markdown_path=str(md_path),
        char_count=len(text.strip()),
        word_count=len(text.split()),
        quality_flags=quality.flags,
    )


def _extract_images(
    image_input: Path,
    manifest: SourceManifest,
    ocr: str,
    config: AppConfig,
    logger,
    continue_on_ocr_error: bool,
) -> list[PageMetadata]:
    if ocr == "never":
        raise RuntimeError("Image inputs require OCR; remove --ocr never")
    paths = source_subdirs(config.data_dir, manifest.source_id)
    provider = _ocr_provider(ocr, config)
    image_paths = _image_files(image_input)
    pages: list[PageMetadata] = []
    for page_index, image_path in enumerate(image_paths):
        page_number = page_index + 1
        dest = paths["pages"] / f"page_{page_number:04d}{image_path.suffix.lower()}"
        if image_path.resolve() != dest.resolve():
            shutil.copy2(image_path, dest)
        md_path = paths["markdown"] / f"page_{page_number:04d}.md"
        meta = _run_ocr_page(
            dest,
            md_path,
            manifest,
            page_number,
            page_index,
            "image",
            provider,
            logger,
            continue_on_ocr_error,
            [],
        )
        _write_page_metadata(paths["metadata"], meta)
        pages.append(meta)
    return pages


def _extract_text_document(
    text_path: Path,
    manifest: SourceManifest,
    config: AppConfig,
    logger,
    input_type: Literal["markdown", "text"],
) -> list[PageMetadata]:
    paths = source_subdirs(config.data_dir, manifest.source_id)
    method: Literal["markdown", "text"] = input_type
    logger.info("Document: using %s extraction", method)
    if input_type == "markdown":
        _, raw_text = read_markdown_with_frontmatter(text_path)
    else:
        raw_text = text_path.read_text(encoding="utf-8-sig")
    if not raw_text.strip():
        raise RuntimeError(f"Empty text document: {text_path}")
    md_path = paths["markdown"] / "page_0001.md"
    text = normalize_page_markdown(raw_text, 1)
    md_path.write_text(text)
    meta = PageMetadata(
        source_id=manifest.source_id,
        page_number=1,
        page_index=0,
        input_type=input_type,
        text_layer_detected=True,
        extraction_method=method,
        image_path=None,
        markdown_path=str(md_path),
        char_count=len(text.strip()),
        word_count=len(text.split()),
        quality_flags=[],
    )
    _write_page_metadata(paths["metadata"], meta)
    return [meta]


def _frontmatter_metadata(input_path: Path) -> SourceMetadataInput:
    if input_path.suffix.lower() not in MARKDOWN_EXTS:
        return SourceMetadataInput()
    metadata, _ = read_markdown_with_frontmatter(input_path)
    return metadata


def _resolve_field(explicit: T | None, default: T | None) -> T | None:
    return explicit if explicit is not None else default


def _run_ocr_page(
    image_path: Path,
    md_path: Path,
    manifest: SourceManifest,
    page_number: int,
    page_index: int,
    input_type: str,
    provider: ShellOCRProvider,
    logger,
    continue_on_ocr_error: bool,
    quality_flags: list[str],
) -> PageMetadata:
    logger.info("Page %s: OCR with %s provider", page_number, provider.name)
    try:
        result = provider.ocr_image(
            image_path,
            md_path,
            {"page_number": page_number, "source_id": manifest.source_id},
        )
        text = normalize_page_markdown(result.text, page_number)
        md_path.write_text(text)
        return PageMetadata(
            source_id=manifest.source_id,
            page_number=page_number,
            page_index=page_index,
            input_type=input_type,  # type: ignore[arg-type]
            text_layer_detected=False,
            extraction_method="ocr",
            ocr_provider=result.provider,
            ocr_model=result.model,
            image_path=str(image_path),
            markdown_path=str(md_path),
            char_count=len(text.strip()),
            word_count=len(text.split()),
            quality_flags=quality_flags + result.warnings,
        )
    except Exception as exc:
        if not continue_on_ocr_error:
            raise
        text = normalize_page_markdown(f"[OCR failed: {exc}]", page_number)
        md_path.write_text(text)
        return PageMetadata(
            source_id=manifest.source_id,
            page_number=page_number,
            page_index=page_index,
            input_type=input_type,  # type: ignore[arg-type]
            text_layer_detected=False,
            extraction_method="failed",
            image_path=str(image_path),
            markdown_path=str(md_path),
            char_count=len(text.strip()),
            word_count=len(text.split()),
            quality_flags=quality_flags + ["ocr_failed"],
        )


def _ocr_provider(ocr: str, config: AppConfig) -> ShellOCRProvider:
    provider_name = config.ocr.default_provider if ocr in {"auto", "always"} else ocr
    provider_cfg = config.ocr.providers.get(provider_name)
    if provider_cfg is None:
        raise ValueError(f"OCR provider not configured: {provider_name}")
    return ShellOCRProvider(provider_cfg, name=provider_name)


def _image_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = [p for p in sorted(path.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    if not files:
        raise FileNotFoundError(f"No supported image files found in {path}")
    return files


def _write_page_metadata(metadata_dir: Path, metadata: PageMetadata) -> None:
    path = metadata_dir / f"page_{metadata.page_number:04d}.json"
    path.write_text(metadata.model_dump_json(indent=2))


def read_page_metadata(source_root: Path) -> list[PageMetadata]:
    rows = []
    for path in sorted((source_root / "metadata").glob("page_*.json")):
        rows.append(PageMetadata.model_validate(json.loads(path.read_text())))
    return rows
