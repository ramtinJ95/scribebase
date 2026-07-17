from __future__ import annotations

import json
import shutil
import threading
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypeVar
from uuid import uuid4

from scribebase.config import AppConfig
from scribebase.durable_fs import durable_replace, durable_rmtree, sync_tree
from scribebase.extractors.pymupdf_extractor import PDFDocument
from scribebase.markdown.frontmatter import read_markdown_with_frontmatter
from scribebase.markdown.normalize import combine_pages, normalize_page_markdown
from scribebase.models import PageMetadata, SourceManifest, SourceMetadataInput, TextQuality
from scribebase.ocr.shell_provider import ShellOCRProvider
from scribebase.ocr.health import ensure_ocr_provider_ready
from scribebase.paths import chapter_file_name, source_dir, source_root_subdirs
from scribebase.pdf_router import evaluate_text_quality
from scribebase.source_registry import (
    create_manifest,
    find_source,
    generate_source_id,
    identity_reservation_owned,
    manifest_path,
    prepare_source_identity,
    refresh_source_identity_reservation,
    release_source_identity,
    reserve_source_identity,
    source_registry_lock,
    write_manifest,
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}
MARKDOWN_EXTS = {".md", ".markdown"}
TEXT_EXTS = {".txt"}
T = TypeVar("T")


@dataclass(frozen=True)
class PDFPageRoute:
    raw_text: str
    quality: TextQuality
    has_images: bool
    has_visual_content: bool


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
    duplicate_policy: str = "reject",
    identity_owner: str | None = None,
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
    origin = _resolve_field(origin, frontmatter.origin)
    canonical_url = _resolve_field(canonical_url, frontmatter.canonical_url)
    url = _resolve_field(url, frontmatter.url)
    external_id = _resolve_field(external_id, frontmatter.external_id)
    source_id = source_id or generate_source_id(title)
    identity_key, content_sha256 = prepare_source_identity(
        config.data_dir,
        input_path,
        origin=origin,
        canonical_url=canonical_url,
        url=url,
        external_id=external_id,
        source_id=source_id,
        duplicate_policy=duplicate_policy,
    )
    owner_id = identity_owner or uuid4().hex
    reserve_source_identity(
        config.data_dir,
        identity_key,
        owner_id=owner_id,
        source_id=source_id,
        duplicate_policy=duplicate_policy,
        owner_type="job" if identity_owner else "direct",
    )
    heartbeat_stop, heartbeat_thread = _start_identity_heartbeat(
        config,
        identity_key,
        owner_id,
        enabled=duplicate_policy == "reject",
    )
    live_root = source_dir(config.data_dir, source_id)
    staging_data = config.data_dir / ".source-staging" / uuid4().hex
    try:
        with source_registry_lock(config.data_dir):
            _recover_source_publication(live_root, logger)
        if live_root.exists():
            try:
                existing = find_source(config.data_dir, source_id)
            except (FileNotFoundError, ValueError):
                existing = None
            if existing is not None:
                if existing.identity_key != identity_key:
                    raise ValueError(
                        f"Source id already exists with different content: {source_id}"
                    )
                if existing.content_sha256 != content_sha256:
                    raise ValueError(
                        f"Source id {source_id} has changed content; submit with a new source id"
                    )
                if (live_root / "markdown" / "document.md").exists():
                    return existing
        manifest = create_manifest(
            staging_data,
            input_path,
            title,
            source_type,
            course,
            chapter,
            language,
            tags=tags if tags is not None else frontmatter.tags,
            origin=origin,
            publisher=_resolve_field(publisher, frontmatter.publisher),
            author=_resolve_field(author, frontmatter.author),
            created_at_source=_resolve_field(created_at_source, frontmatter.created_at_source),
            updated_at_source=_resolve_field(updated_at_source, frontmatter.updated_at_source),
            retrieved_at=_resolve_field(retrieved_at, frontmatter.retrieved_at),
            url=url,
            canonical_url=canonical_url,
            external_id=external_id,
            collection=_resolve_field(collection, frontmatter.collection),
            summary=_resolve_field(summary, frontmatter.summary),
            source_id=source_id,
            identity_key=identity_key,
            content_sha256=content_sha256,
        )
        manifest = _extract_manifest(
            manifest,
            chapter,
            ocr,
            config,
            logger,
            continue_on_ocr_error,
        )
        return _publish_staged_source(
            manifest,
            live_root,
            config.data_dir,
            identity_key,
            owner_id,
            duplicate_policy,
            logger,
        )
    finally:
        if heartbeat_stop is not None and heartbeat_thread is not None:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=config.server.identity_reservation_heartbeat_seconds + 1)
        shutil.rmtree(staging_data, ignore_errors=True)
        release_source_identity(config.data_dir, identity_key, owner_id)


def _extract_manifest(
    manifest: SourceManifest,
    chapter: str | None,
    ocr: str,
    config: AppConfig,
    logger,
    continue_on_ocr_error: bool,
) -> SourceManifest:
    logger.info("Ingest source: %s (%s)", manifest.title, manifest.source_id)
    paths = source_root_subdirs(Path(manifest.data_dir))
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
        1 for page in pages if page.extraction_method == "pymupdf4llm"
    )
    ocr_pages = [page for page in pages if page.extraction_method == "ocr"]
    manifest.extraction_summary.pages_ocr = len(ocr_pages)
    providers = {page.ocr_provider for page in ocr_pages if page.ocr_provider}
    models = {page.ocr_model for page in ocr_pages if page.ocr_model}
    manifest.extraction_summary.ocr_provider = _one_or_mixed(providers)
    manifest.extraction_summary.ocr_model = _one_or_mixed(models)
    manifest.updated_at = datetime.now(timezone.utc)
    write_manifest(manifest)
    if not document_path.read_text().strip():
        raise RuntimeError(
            "Empty extraction. Try --ocr always or check OCR provider configuration."
        )
    return manifest


def _publish_staged_source(
    manifest: SourceManifest,
    live_root: Path,
    data_dir: Path,
    identity_key: str,
    owner_id: str,
    duplicate_policy: str,
    logger,
) -> SourceManifest:
    staged_root = Path(manifest.data_dir)
    for path in (staged_root / "metadata").glob("page_*.json"):
        payload = json.loads(path.read_text())
        for field in ["image_path", "markdown_path"]:
            value = payload.get(field)
            if value:
                payload[field] = str(live_root / Path(value).relative_to(staged_root))
        path.write_text(json.dumps(payload, indent=2))
    manifest.original_path = str(live_root / Path(manifest.original_path).relative_to(staged_root))
    manifest.data_dir = str(live_root)
    manifest_path(staged_root).write_text(manifest.model_dump_json(indent=2))
    sync_tree(staged_root)
    with source_registry_lock(data_dir):
        if duplicate_policy == "reject" and not identity_reservation_owned(
            data_dir, identity_key, owner_id
        ):
            raise RuntimeError(f"Source identity reservation was lost: {identity_key}")
        try:
            duplicate = find_source(data_dir, manifest.source_id) if live_root.exists() else None
        except (FileNotFoundError, ValueError):
            duplicate = None
        if duplicate is not None and (
            duplicate.identity_key != identity_key
            or duplicate.content_sha256 != manifest.content_sha256
        ):
            raise ValueError(
                f"Source id already exists with different content: {manifest.source_id}"
            )
        backup = live_root.with_name(f".{live_root.name}.backup.{uuid4().hex}")
        if live_root.exists():
            durable_replace(live_root, backup)
        live_root.parent.mkdir(parents=True, exist_ok=True)
        try:
            durable_replace(staged_root, live_root)
        except Exception:
            if backup.exists() and not live_root.exists():
                durable_replace(backup, live_root)
            raise
        else:
            try:
                durable_rmtree(backup)
            except OSError as exc:
                logger.warning(
                    "Published source %s but could not remove old backup %s: %s",
                    manifest.source_id,
                    backup,
                    exc,
                )
    return manifest


def _recover_source_publication(live_root: Path, logger=None) -> None:  # noqa: ANN001
    backups = sorted(live_root.parent.glob(f".{live_root.name}.backup.*"))
    if live_root.exists():
        for backup in backups:
            _remove_redundant_source_backup(backup, live_root, logger)
        return
    if backups:
        durable_replace(backups[-1], live_root)
        for backup in backups[:-1]:
            _remove_redundant_source_backup(backup, live_root, logger)


def _remove_redundant_source_backup(backup: Path, live_root: Path, logger=None) -> None:  # noqa: ANN001
    try:
        durable_rmtree(backup)
    except OSError as exc:
        message = f"Live source {live_root.name} is valid but old backup {backup} remains: {exc}"
        if logger is not None:
            logger.warning(message)
        else:
            warnings.warn(message, stacklevel=2)


def recover_source_publications(data_dir: Path, logger=None) -> int:  # noqa: ANN001
    """Settle source-directory renames interrupted by process or machine loss."""
    sources = data_dir / "sources"
    if not sources.exists():
        return 0
    recovered = 0
    with source_registry_lock(data_dir):
        source_ids = {
            backup.name[1:].split(".backup.", 1)[0]
            for backup in sources.glob(".*.backup.*")
            if ".backup." in backup.name
        }
        for source_id in sorted(source_ids):
            try:
                live_root = source_dir(data_dir, source_id)
            except ValueError:
                continue
            _recover_source_publication(live_root, logger)
            recovered += 1
    return recovered


def _start_identity_heartbeat(
    config: AppConfig,
    identity_key: str,
    owner_id: str,
    *,
    enabled: bool,
) -> tuple[threading.Event | None, threading.Thread | None]:
    if not enabled:
        return None, None
    stop = threading.Event()

    def refresh() -> None:
        while not stop.wait(config.server.identity_reservation_heartbeat_seconds):
            if not refresh_source_identity_reservation(config.data_dir, identity_key, owner_id):
                return

    thread = threading.Thread(target=refresh, name="scribebase-identity-heartbeat", daemon=True)
    thread.start()
    return stop, thread


def _extract_pdf(
    pdf_path: Path,
    manifest: SourceManifest,
    ocr: str,
    config: AppConfig,
    logger,
    continue_on_ocr_error: bool,
) -> list[PageMetadata]:
    paths = source_root_subdirs(Path(manifest.data_dir))
    provider: ShellOCRProvider | None = None
    pages: list[PageMetadata] = []
    with PDFDocument(pdf_path) as pdf:
        routes = _pdf_page_routes(pdf, config)
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
                    pdf,
                    page_index,
                    md_path,
                    manifest,
                    route,
                    logger,
                )
            else:
                if ocr == "never":
                    raise RuntimeError(
                        f"Page {page_number} has insufficient text and OCR is disabled"
                    )
                selected_provider = provider or _ocr_provider(ocr, config)
                render_dpi = selected_provider.config.render_dpi or config.ocr.render_dpi
                logger.info(
                    "Page %s: insufficient text, rendering at %s DPI", page_number, render_dpi
                )
                pdf.render_page(page_index, image_path, render_dpi)
                if pdf.page_is_visually_blank(page_index):
                    logger.info("Page %s: rendered page is blank; skipping OCR", page_number)
                    meta = _blank_pdf_page_metadata(
                        image_path,
                        md_path,
                        manifest,
                        page_number,
                        page_index,
                        route.quality.flags,
                    )
                else:
                    if provider is None:
                        ensure_ocr_provider_ready(
                            selected_provider.name, selected_provider.config
                        )
                        provider = selected_provider
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


def _pdf_page_routes(pdf: PDFDocument, config: AppConfig) -> list[PDFPageRoute]:
    routes = []
    for page_index in range(pdf.page_count):
        raw_text = pdf.extract_page_text(page_index)
        has_images = pdf.page_has_images(page_index)
        quality = evaluate_text_quality(raw_text, config.pdf_detection)
        routes.append(
            PDFPageRoute(
                raw_text=raw_text,
                quality=quality,
                has_images=has_images,
                has_visual_content=(
                    True
                    if has_images or quality.is_true_text
                    else pdf.page_has_visual_content(page_index)
                ),
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
    if likely_true_text_pdf and not route.has_visual_content:
        return False
    serious_quality_flags = set(route.quality.flags) - {"too_few_chars"}
    if route.quality.char_count > 0 and not route.has_images and not serious_quality_flags:
        return False
    if route.quality.char_count > 0 and not route.has_visual_content:
        return False
    return route.has_visual_content


def _extract_text_pdf_page(
    pdf: PDFDocument,
    page_index: int,
    md_path: Path,
    manifest: SourceManifest,
    route: PDFPageRoute,
    logger,
) -> PageMetadata:
    page_number = page_index + 1
    page_md, method, fallback_warning = pdf.extract_page_markdown(page_index, route.raw_text)
    if fallback_warning:
        logger.warning("Page %s: %s; using PyMuPDF text fallback", page_number, fallback_warning)
    text = normalize_page_markdown(page_md, page_number)
    md_path.write_text(text)
    return PageMetadata(
        source_id=manifest.source_id,
        page_number=page_number,
        page_index=page_index,
        input_type="pdf_page",
        text_layer_detected=route.quality.is_true_text,
        extraction_method=method,  # type: ignore[arg-type]
        image_path=None,
        markdown_path=str(md_path),
        char_count=len(text.strip()),
        word_count=len(text.split()),
        quality_flags=route.quality.flags + ([fallback_warning] if fallback_warning else []),
    )


def _blank_pdf_page_metadata(
    image_path: Path,
    md_path: Path,
    manifest: SourceManifest,
    page_number: int,
    page_index: int,
    quality_flags: list[str],
) -> PageMetadata:
    text = normalize_page_markdown("", page_number)
    md_path.write_text(text)
    return PageMetadata(
        source_id=manifest.source_id,
        page_number=page_number,
        page_index=page_index,
        input_type="pdf_page",
        text_layer_detected=False,
        extraction_method="skipped",
        image_path=str(image_path),
        markdown_path=str(md_path),
        char_count=0,
        word_count=0,
        quality_flags=quality_flags + ["blank_page"],
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
    paths = source_root_subdirs(Path(manifest.data_dir))
    provider = _ready_ocr_provider(ocr, config)
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
    paths = source_root_subdirs(Path(manifest.data_dir))
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
        if not result.text.strip():
            raise RuntimeError(
                f"OCR provider {provider.name!r} returned empty output for nonblank "
                f"page {page_number}: {image_path}"
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


def _ready_ocr_provider(ocr: str, config: AppConfig) -> ShellOCRProvider:
    provider = _ocr_provider(ocr, config)
    ensure_ocr_provider_ready(provider.name, provider.config)
    return provider


def _image_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = [p for p in sorted(path.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    if not files:
        raise FileNotFoundError(f"No supported image files found in {path}")
    return files


def _one_or_mixed(values: set[str]) -> str | None:
    if len(values) == 1:
        return next(iter(values))
    return "mixed" if values else None


def _write_page_metadata(metadata_dir: Path, metadata: PageMetadata) -> None:
    path = metadata_dir / f"page_{metadata.page_number:04d}.json"
    path.write_text(metadata.model_dump_json(indent=2))


def read_page_metadata(source_root: Path) -> list[PageMetadata]:
    rows = []
    for path in sorted((source_root / "metadata").glob("page_*.json")):
        rows.append(PageMetadata.model_validate(json.loads(path.read_text())))
    return rows
