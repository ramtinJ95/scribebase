from __future__ import annotations

import re
import fcntl
import time
from contextlib import contextmanager
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
from scribebase.source_registry import find_source

JobStatus = Literal["queued", "running", "succeeded", "failed"]
MARKDOWN_EXTS = {".md", ".markdown"}
SUPPORTED_UPLOAD_EXTS = MARKDOWN_EXTS | {
    ".pdf",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".webp",
    ".bmp",
}
T = TypeVar("T")


class UploadTooLargeError(ValueError):
    pass


class QueueFullError(RuntimeError):
    pass


class UnsupportedUploadError(ValueError):
    pass


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
    attempts: int = 0


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
    attempts: int = 0


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
    suffix = Path(safe_name).suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_EXTS:
        raise UnsupportedUploadError(f"Unsupported upload type: {suffix or 'no extension'}")
    upload_path = config.data_dir / "uploads" / f"{job_id}_{safe_name}"
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _copy_limited(fileobj, upload_path, config.server.max_upload_bytes)
    except Exception:
        upload_path.unlink(missing_ok=True)
        raise

    try:
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
        with _queue_lock(config.data_dir):
            active = sum(
                1
                for existing in list_jobs(config.data_dir)
                if existing.status in {"queued", "running"}
            )
            if active >= config.server.max_active_jobs:
                raise QueueFullError(
                    f"Ingestion queue is full ({active}/{config.server.max_active_jobs})"
                )
            write_job(config.data_dir, job)
        return job
    except Exception:
        upload_path.unlink(missing_ok=True)
        raise


def run_ingest_job(job_id: str, config: AppConfig) -> None:
    job = read_job(config.data_dir, job_id)
    logger = setup_logging(config.data_dir)
    if job.status != "running":
        job.status = "running"
        job.started_at = _now()
        job.updated_at = job.started_at
        job.error = None
        job.attempts += 1
        write_job(config.data_dir, job)

    try:
        if job.source_id:
            manifest = find_source(config.data_dir, job.source_id)
        else:
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
        if job.status == "succeeded":
            Path(job.upload_path).unlink(missing_ok=True)


def list_jobs(data_dir: Path) -> list[IngestJob]:
    jobs_dir = data_dir / "jobs"
    if not jobs_dir.exists():
        return []
    return sorted(
        (IngestJob.model_validate_json(path.read_text()) for path in jobs_dir.glob("*.json")),
        key=lambda job: (job.created_at, job.job_id),
    )


def recover_interrupted_jobs(data_dir: Path) -> int:
    recovered = 0
    with _queue_lock(data_dir):
        for job in list_jobs(data_dir):
            if job.status != "running":
                continue
            job.status = "queued"
            job.error = "Worker interrupted; queued for recovery"
            job.updated_at = _now()
            job.finished_at = None
            write_job(data_dir, job)
            recovered += 1
    return recovered


def claim_next_job(data_dir: Path) -> IngestJob | None:
    with _queue_lock(data_dir):
        job = next(
            (candidate for candidate in list_jobs(data_dir) if candidate.status == "queued"), None
        )
        if job is None:
            return None
        job.status = "running"
        job.started_at = _now()
        job.updated_at = job.started_at
        job.error = None
        job.attempts += 1
        write_job(data_dir, job)
        return job


def run_worker(config: AppConfig, once: bool = False, poll_seconds: float | None = None) -> None:
    ensure_data_layout(config.data_dir)
    delay = poll_seconds if poll_seconds is not None else config.server.worker_poll_seconds
    with _worker_lock(config.data_dir):
        recover_interrupted_jobs(config.data_dir)
        while True:
            job = claim_next_job(config.data_dir)
            if job is not None:
                run_ingest_job(job.job_id, config)
                if once:
                    return
                continue
            if once:
                return
            time.sleep(delay)


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
    temporary = path.with_suffix(f"{path.suffix}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(job.model_dump_json(indent=2))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
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


def worker_is_running(data_dir: Path) -> bool:
    path = data_dir / "jobs" / ".worker.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        return False


def _copy_limited(fileobj: BinaryIO, destination: Path, limit: int) -> None:
    written = 0
    with destination.open("wb") as output:
        while data := fileobj.read(1024 * 1024):
            written += len(data)
            if written > limit:
                raise UploadTooLargeError(f"Upload exceeds {limit} bytes")
            output.write(data)


@contextmanager
def _queue_lock(data_dir: Path):  # noqa: ANN202
    with _file_lock(data_dir / "jobs" / ".queue.lock", blocking=True):
        yield


@contextmanager
def _worker_lock(data_dir: Path):  # noqa: ANN202
    try:
        with _file_lock(data_dir / "jobs" / ".worker.lock", blocking=False):
            yield
    except BlockingIOError as exc:
        raise RuntimeError("Another ScribeBase ingestion worker is already running") from exc


@contextmanager
def _file_lock(path: Path, blocking: bool):  # noqa: ANN202
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lock_file:
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        fcntl.flock(lock_file.fileno(), flags)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
