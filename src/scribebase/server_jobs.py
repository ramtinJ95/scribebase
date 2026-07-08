from __future__ import annotations

import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Literal, TypeVar
from uuid import uuid4

from scribebase.config import AppConfig
from scribebase.extraction import extract_source
from scribebase.indexing import index_source
from scribebase.logging_utils import setup_logging
from scribebase.markdown.frontmatter import read_markdown_with_frontmatter
from scribebase.models import (
    GenericMetadata,
    Language,
    SourceMetadataInput,
    SourceType,
    normalize_tags,
)
from scribebase.paths import ensure_data_layout

JobStatus = Literal["queued", "running", "succeeded", "failed"]
MARKDOWN_EXTS = {".md", ".markdown"}
T = TypeVar("T")


class IngestJob(GenericMetadata):
    job_id: str
    status: JobStatus
    filename: str
    upload_path: str
    title: str
    source_type: SourceType = "other"
    course: str | None = None
    chapter: str | None = None
    language: Language = "unknown"
    ocr: str = "auto"
    no_index: bool = False
    continue_on_ocr_error: bool = False
    source_id: str | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


class IngestJobResponse(GenericMetadata):
    job_id: str
    status: JobStatus
    filename: str
    title: str
    source_type: SourceType = "other"
    course: str | None = None
    chapter: str | None = None
    language: Language = "unknown"
    ocr: str = "auto"
    no_index: bool = False
    continue_on_ocr_error: bool = False
    source_id: str | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


def public_job(job: IngestJob) -> IngestJobResponse:
    return IngestJobResponse.model_validate(job.model_dump(exclude={"upload_path"}))


def create_ingest_job(
    config: AppConfig,
    filename: str,
    fileobj: BinaryIO,
    title: str | None,
    source_type: SourceType | None,
    course: str | None,
    chapter: str | None,
    language: Language | None,
    ocr: str,
    no_index: bool,
    continue_on_ocr_error: bool,
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
) -> IngestJob:
    ensure_data_layout(config.data_dir)
    job_id = uuid4().hex
    safe_name = _safe_filename(filename)
    upload_path = config.data_dir / "uploads" / f"{job_id}_{safe_name}"
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    with upload_path.open("wb") as out:
        shutil.copyfileobj(fileobj, out)

    frontmatter = _frontmatter_metadata(upload_path)
    title = _resolve_field(title, frontmatter.title)
    if not title:
        raise ValueError("title is required unless provided by Markdown frontmatter")
    source_type = _resolve_field(source_type, frontmatter.source_type) or "other"
    course = _resolve_field(course, frontmatter.course)
    chapter = _resolve_field(chapter, frontmatter.chapter)
    language = _resolve_field(language, frontmatter.language) or "unknown"

    now = _now()
    job = IngestJob(
        job_id=job_id,
        status="queued",
        filename=safe_name,
        upload_path=str(upload_path),
        title=title,
        source_type=source_type,
        course=course,
        chapter=chapter,
        language=language,
        tags=normalize_tags(tags) if tags is not None else frontmatter.tags,
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
        ocr=ocr,
        no_index=no_index,
        continue_on_ocr_error=continue_on_ocr_error,
        created_at=now,
        updated_at=now,
    )
    write_job(config.data_dir, job)
    return job


def run_ingest_job(job_id: str, config: AppConfig) -> None:
    job = read_job(config.data_dir, job_id)
    logger = setup_logging(config.data_dir)
    job.status = "running"
    job.started_at = _now()
    job.updated_at = job.started_at
    job.error = None
    write_job(config.data_dir, job)

    try:
        manifest = extract_source(
            Path(job.upload_path),
            job.title,
            job.source_type,
            job.course,
            job.chapter,
            job.language,
            job.ocr,
            config,
            logger,
            job.continue_on_ocr_error,
            tags=job.tags,
            origin=job.origin,
            publisher=job.publisher,
            author=job.author,
            created_at_source=job.created_at_source,
            updated_at_source=job.updated_at_source,
            retrieved_at=job.retrieved_at,
            url=job.url,
            canonical_url=job.canonical_url,
            external_id=job.external_id,
            collection=job.collection,
            summary=job.summary,
        )
        job.source_id = manifest.source_id
        job.updated_at = _now()
        write_job(config.data_dir, job)
        if not job.no_index:
            index_source(manifest.source_id, config, logger)
        job.status = "succeeded"
    except Exception as exc:
        job.status = "failed"
        job.error = str(exc).strip() or exc.__class__.__name__
    finally:
        job.finished_at = _now()
        job.updated_at = job.finished_at
        write_job(config.data_dir, job)


def read_job(data_dir: Path, job_id: str) -> IngestJob:
    path = job_path(data_dir, job_id)
    if not path.exists():
        raise FileNotFoundError(f"Job not found: {job_id}")
    return IngestJob.model_validate_json(path.read_text())


def _frontmatter_metadata(path: Path) -> SourceMetadataInput:
    if path.suffix.lower() not in MARKDOWN_EXTS:
        return SourceMetadataInput()
    metadata, _ = read_markdown_with_frontmatter(path)
    return metadata


def _resolve_field(explicit: T | None, default: T | None) -> T | None:
    return explicit if explicit is not None else default


def write_job(data_dir: Path, job: IngestJob) -> Path:
    path = job_path(data_dir, job.job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(job.model_dump_json(indent=2))
    return path


def job_path(data_dir: Path, job_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
        raise ValueError(f"Invalid job id: {job_id}")
    return data_dir / "jobs" / f"{job_id}.json"


def _safe_filename(filename: str) -> str:
    name = Path(filename or "upload").name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name[:120] or "upload"


def _now() -> datetime:
    return datetime.now(timezone.utc)
