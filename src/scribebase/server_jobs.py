from __future__ import annotations

import fcntl
import json
import os
import re
import socket
import threading
import time
import warnings
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import BinaryIO, Literal, TypeVar
from uuid import uuid4

from scribebase.config import AppConfig
from scribebase.durable_fs import atomic_write, atomic_write_text, durable_replace, durable_unlink
from scribebase.embeddings.llamacpp_client import LlamaCppEmbeddingClient
from scribebase.errors import DependencyUnavailableError
from scribebase.extraction import extract_source, recover_source_publications
from scribebase.indexing import index_recovery_pending, index_source, recover_index_transactions
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
from scribebase.source_registry import (
    DuplicateSourceError,
    find_source,
    prepare_source_identity,
    reconcile_source_identity_reservations,
    release_source_identity,
    reserve_source_identity,
    slugify,
)
from scribebase.vectorstores.weaviate_store import WeaviateStore

JobStatus = Literal["queued", "running", "succeeded", "failed"]
JobPhase = Literal["queued", "extracting", "extracted", "indexing", "completed"]
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
    phase: JobPhase = "queued"
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
    claim_token: str | None = None
    worker_id: str | None = None
    duplicate_policy: Literal["reject", "create"] = "reject"
    identity_key: str | None = None
    content_sha256: str | None = None
    next_attempt_at: datetime | None = None


class IngestJobResponse(GenericMetadata):
    job_id: str
    status: JobStatus
    phase: JobPhase = "queued"
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
    duplicate_policy: Literal["reject", "create"] = "reject"
    identity_key: str | None = None
    content_sha256: str | None = None
    next_attempt_at: datetime | None = None


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
    expected_size: int | None = None,
    duplicate_policy: Literal["reject", "create"] = "reject",
) -> IngestJob:
    ensure_data_layout(config.data_dir)
    job_id = uuid4().hex
    safe_name = _safe_filename(filename)
    suffix = Path(safe_name).suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_EXTS:
        raise UnsupportedUploadError(f"Unsupported upload type: {suffix or 'no extension'}")
    reserved_bytes = min(
        expected_size or config.server.max_upload_bytes,
        config.server.max_upload_bytes,
    )
    _reserve_upload(config, job_id, reserved_bytes)
    upload_path = config.data_dir / "uploads" / f"{job_id}_{safe_name}"
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _copy_limited(fileobj, upload_path, config.server.max_upload_bytes)
    except Exception:
        durable_unlink(upload_path, missing_ok=True)
        _release_reservation(config.data_dir, job_id)
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
        planned_source_id = f"{slugify(title)}_{now.year}_{job_id[:6]}"
        identity_key, content_sha256 = prepare_source_identity(
            config.data_dir,
            upload_path,
            origin=_resolve_field(origin, frontmatter.origin),
            canonical_url=_resolve_field(canonical_url, frontmatter.canonical_url),
            url=_resolve_field(url, frontmatter.url),
            external_id=_resolve_field(external_id, frontmatter.external_id),
            source_id=planned_source_id,
            duplicate_policy=duplicate_policy,
        )
        for existing in list_jobs(config.data_dir):
            if existing.identity_key == identity_key and not _job_blocks_duplicate(
                config.data_dir, existing
            ):
                release_source_identity(config.data_dir, identity_key, existing.job_id)
        reserve_source_identity(
            config.data_dir,
            identity_key,
            owner_id=job_id,
            source_id=planned_source_id,
            duplicate_policy=duplicate_policy,
            owner_type="job",
        )
        job = IngestJob(
            job_id=job_id,
            status="queued",
            phase="queued",
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
            source_id=planned_source_id,
            duplicate_policy=duplicate_policy,
            identity_key=identity_key,
            content_sha256=content_sha256,
            created_at=now,
            updated_at=now,
        )
        with _queue_lock(config.data_dir):
            _validate_upload_storage(config)
            if duplicate_policy == "reject":
                duplicate_job = next(
                    (
                        existing
                        for existing in list_jobs(config.data_dir)
                        if existing.identity_key == identity_key
                        and existing.source_id != planned_source_id
                        and _job_blocks_duplicate(config.data_dir, existing)
                    ),
                    None,
                )
                if duplicate_job is not None:
                    raise DuplicateSourceError(duplicate_job.source_id or "unknown", identity_key)
            write_job(config.data_dir, job)
            try:
                _release_reservation_unlocked(config.data_dir, job_id)
            except OSError as exc:
                _warn_nonfatal(
                    f"Queued job {job_id}, but upload reservation cleanup could not be "
                    f"confirmed: {exc}",
                )
        return job
    except Exception:
        durable_unlink(upload_path, missing_ok=True)
        _release_reservation(config.data_dir, job_id)
        if "identity_key" in locals():
            release_source_identity(config.data_dir, identity_key, job_id)
        raise


def run_ingest_job(job_id: str, claim_token: str, config: AppConfig) -> None:
    with _queue_lock(config.data_dir):
        job = read_job(config.data_dir, job_id)
        if job.status in {"succeeded", "failed"}:
            return
        if job.status != "running" or job.claim_token != claim_token:
            raise RuntimeError(f"Job claim is not owned by this worker: {job_id}")
    logger = setup_logging(config.data_dir)

    try:
        if job.phase in {"queued", "extracting"}:
            job.phase = "extracting"
            _persist_claimed_job(config.data_dir, job, claim_token)
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
                source_id=job.source_id,
                duplicate_policy=job.duplicate_policy,
                identity_owner=job.job_id,
            )
            if manifest.source_id != job.source_id:
                raise RuntimeError(
                    f"Extractor returned unexpected source id {manifest.source_id!r}; "
                    f"expected {job.source_id!r}"
                )
            job.phase = "extracted"
            job.updated_at = _now()
            _persist_claimed_job(config.data_dir, job, claim_token)
        else:
            manifest = find_source(config.data_dir, job.source_id or "")

        if job.no_index:
            job.phase = "completed"
        elif job.phase in {"extracted", "indexing"}:
            job.phase = "indexing"
            job.updated_at = _now()
            _persist_claimed_job(config.data_dir, job, claim_token)
            ready, message = _index_dependencies_ready(config)
            if not ready:
                raise DependencyUnavailableError(message)
            # Replaying an interrupted operation is deliberate. A matching local
            # operation ID cannot prove that Weaviate survived the same outage.
            index_source(manifest.source_id, config, logger, operation_id=job.job_id)
            job.phase = "completed"
        job.status = "succeeded"
    except Exception as exc:
        if isinstance(exc, DependencyUnavailableError):
            job.status = "queued"
            job.error = f"Dependency unavailable; retry scheduled: {exc}"
            job.next_attempt_at = _now() + timedelta(
                seconds=config.server.worker_dependency_retry_seconds
            )
        else:
            job.status = "failed"
            job.error = str(exc).strip() or exc.__class__.__name__
    finally:
        now = _now()
        job.finished_at = now if job.status in {"succeeded", "failed"} else None
        job.updated_at = now
        job.claim_token = None
        job.worker_id = None
        persisted = False
        with _queue_lock(config.data_dir):
            current = read_job(config.data_dir, job.job_id)
            if current.claim_token == claim_token:
                write_job(config.data_dir, job)
                persisted = True
        if persisted:
            _cleanup_terminal_job(job, config, logger)


def _cleanup_terminal_job(job: IngestJob, config: AppConfig, logger) -> None:  # noqa: ANN001
    try:
        if job.status == "succeeded":
            durable_unlink(Path(job.upload_path), missing_ok=True)
        elif job.status == "failed" and job.identity_key:
            release_source_identity(config.data_dir, job.identity_key, job.job_id)
    except Exception as exc:
        logger.warning(
            "Job %s is durably %s but terminal cleanup failed: %s",
            job.job_id,
            job.status,
            exc,
        )


def list_jobs(data_dir: Path) -> list[IngestJob]:
    jobs_dir = data_dir / "jobs"
    if not jobs_dir.exists():
        return []
    jobs = []
    for path in jobs_dir.glob("*.json"):
        try:
            jobs.append(IngestJob.model_validate_json(path.read_text()))
        except Exception as exc:
            quarantine = path.with_suffix(f".corrupt.{int(time.time())}")
            durable_replace(path, quarantine)
            warnings.warn(f"Quarantined corrupt job {path}: {exc}", stacklevel=2)
    return sorted(jobs, key=lambda job: (job.created_at, job.job_id))


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
            job.claim_token = None
            job.worker_id = None
            job.next_attempt_at = None
            write_job(data_dir, job)
            recovered += 1
    return recovered


def claim_next_job(data_dir: Path, worker_id: str) -> IngestJob | None:
    with _queue_lock(data_dir):
        job = next(
            (
                candidate
                for candidate in list_jobs(data_dir)
                if candidate.status == "queued"
                and (candidate.next_attempt_at is None or candidate.next_attempt_at <= _now())
            ),
            None,
        )
        if job is None:
            return None
        job.status = "running"
        job.started_at = _now()
        job.updated_at = job.started_at
        job.error = None
        job.next_attempt_at = None
        job.attempts += 1
        job.claim_token = uuid4().hex
        job.worker_id = worker_id
        write_job(data_dir, job)
        return job


def retry_job(config: AppConfig, job_id: str) -> IngestJob:
    job = read_job(config.data_dir, job_id)
    reservation_created = False
    if job.identity_key and job.source_id:
        reservation_created = reserve_source_identity(
            config.data_dir,
            job.identity_key,
            owner_id=job.job_id,
            source_id=job.source_id,
            duplicate_policy=job.duplicate_policy,
            owner_type="job",
        )
    try:
        with _queue_lock(config.data_dir):
            job = read_job(config.data_dir, job_id)
            if job.status != "failed":
                raise ValueError(f"Only failed jobs can be retried: {job_id}")
            if not Path(job.upload_path).exists() and job.phase in {"queued", "extracting"}:
                raise FileNotFoundError(f"Upload is no longer available for retry: {job_id}")
            active = sum(
                1
                for existing in list_jobs(config.data_dir)
                if existing.status in {"queued", "running"}
            )
            if active >= config.server.max_active_jobs:
                raise QueueFullError(
                    f"Ingestion queue is full ({active}/{config.server.max_active_jobs})"
                )
            job.status = "queued"
            job.error = None
            job.finished_at = None
            job.claim_token = None
            job.worker_id = None
            job.updated_at = _now()
            job.next_attempt_at = None
            write_job(config.data_dir, job)
            return job
    except Exception:
        if job.identity_key and reservation_created:
            release_source_identity(config.data_dir, job.identity_key, job.job_id)
        raise


def run_worker(config: AppConfig, once: bool = False, poll_seconds: float | None = None) -> None:
    ensure_data_layout(config.data_dir)
    delay = poll_seconds if poll_seconds is not None else config.server.worker_poll_seconds
    worker_id = f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex[:8]}"
    logger = setup_logging(config.data_dir)
    with _worker_lock(config.data_dir):
        recovered_sources = recover_source_publications(config.data_dir, logger)
        if recovered_sources:
            logger.warning("Recovered %s interrupted source publications", recovered_sources)
        while True:
            while index_recovery_pending(config.data_dir):
                ready, message = _weaviate_ready(config)
                if not ready:
                    logger.warning("Index recovery waiting for Weaviate: %s", message)
                    if once:
                        return
                    time.sleep(config.server.worker_dependency_retry_seconds)
                    continue
                # Journal/artifact errors are permanent until an operator intervenes.
                # Let them fail startup instead of advertising a healthy idle worker.
                try:
                    recover_index_transactions(config, logger)
                except DependencyUnavailableError as exc:
                    logger.warning("Index recovery lost Weaviate; waiting to retry: %s", exc)
                    if once:
                        return
                    time.sleep(config.server.worker_dependency_retry_seconds)
                    continue
            with _worker_heartbeat(config, worker_id):
                reconcile_queue_storage(config)
                recover_interrupted_jobs(config.data_dir)
                last_reconcile = time.monotonic()
                while True:
                    if index_recovery_pending(config.data_dir):
                        break
                    if time.monotonic() - last_reconcile >= 60:
                        reconcile_queue_storage(config)
                        last_reconcile = time.monotonic()
                    job = claim_next_job(config.data_dir, worker_id)
                    if job is not None:
                        run_ingest_job(job.job_id, job.claim_token or "", config)
                        if index_recovery_pending(config.data_dir):
                            break
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
    _write_text_atomic(path, job.model_dump_json(indent=2))
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


def _warn_nonfatal(message: str) -> None:
    """Report post-commit cleanup trouble without rolling back committed state."""
    try:
        warnings.warn(message, stacklevel=3)
    except Exception:
        pass


def worker_status(config: AppConfig) -> tuple[bool, str]:
    if not worker_is_running(config.data_dir):
        return False, "worker is not running"
    path = _heartbeat_path(config.data_dir)
    try:
        payload = json.loads(path.read_text())
        updated = datetime.fromisoformat(payload["updated_at"])
        age = (_now() - updated).total_seconds()
    except Exception as exc:
        return False, f"worker heartbeat unavailable: {exc}"
    if age > config.server.worker_stale_seconds:
        return False, f"worker heartbeat stale ({age:.1f}s)"
    return True, f"worker healthy ({payload['worker_id']}, heartbeat {age:.1f}s ago)"


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
    def copy(output: BinaryIO) -> None:
        written = 0
        while data := fileobj.read(min(1024 * 1024, limit - written + 1)):
            written += len(data)
            if written > limit:
                raise UploadTooLargeError(f"Upload exceeds {limit} bytes")
            output.write(data)

    atomic_write(destination, copy)


def _index_dependencies_ready(config: AppConfig) -> tuple[bool, str]:
    ready, message = _weaviate_ready(config)
    if not ready:
        return ready, message
    embedding_ok, message = LlamaCppEmbeddingClient(config.embedding).check_health()
    if not embedding_ok:
        return False, f"Embedding service is unavailable: {message}"
    return True, "index dependencies ready"


def _weaviate_ready(config: AppConfig) -> tuple[bool, str]:
    store = WeaviateStore(config.weaviate)
    try:
        if not store.is_ready():
            return False, f"Weaviate is not ready at {config.weaviate.url}"
    except DependencyUnavailableError as exc:
        return False, f"Weaviate is unavailable: {exc}"
    finally:
        store.close()
    return True, "Weaviate ready"


def _persist_claimed_job(data_dir: Path, job: IngestJob, claim_token: str) -> None:
    with _queue_lock(data_dir):
        current = read_job(data_dir, job.job_id)
        if current.status != "running" or current.claim_token != claim_token:
            raise RuntimeError(f"Job claim was lost: {job.job_id}")
        write_job(data_dir, job)


def _job_blocks_duplicate(data_dir: Path, job: IngestJob) -> bool:
    if job.status in {"queued", "running"}:
        return True
    if job.status != "succeeded" or not job.source_id:
        return False
    try:
        find_source(data_dir, job.source_id)
        return True
    except FileNotFoundError:
        return False


def _reservation_dir(data_dir: Path) -> Path:
    return data_dir / "jobs" / "reservations"


def _reserve_upload(config: AppConfig, job_id: str, reserved_bytes: int) -> None:
    with _queue_lock(config.data_dir):
        reservations = list(_reservation_dir(config.data_dir).glob("*.json"))
        active = sum(
            1 for job in list_jobs(config.data_dir) if job.status in {"queued", "running"}
        ) + len(reservations)
        if active >= config.server.max_active_jobs:
            raise QueueFullError(
                f"Ingestion queue is full ({active}/{config.server.max_active_jobs})"
            )
        used = _upload_bytes(config.data_dir) + sum(
            int(json.loads(path.read_text())["reserved_bytes"]) for path in reservations
        )
        if used + reserved_bytes > config.server.max_upload_storage_bytes:
            raise QueueFullError(
                "Upload storage budget exceeded "
                f"({used + reserved_bytes}/{config.server.max_upload_storage_bytes} bytes)"
            )
        path = _reservation_dir(config.data_dir) / f"{job_id}.json"
        _write_text_atomic(
            path,
            json.dumps(
                {
                    "job_id": job_id,
                    "reserved_bytes": reserved_bytes,
                    "created_at": _now().isoformat(),
                }
            ),
        )


def _release_reservation(data_dir: Path, job_id: str) -> None:
    with _queue_lock(data_dir):
        _release_reservation_unlocked(data_dir, job_id)


def _release_reservation_unlocked(data_dir: Path, job_id: str) -> None:
    durable_unlink(_reservation_dir(data_dir) / f"{job_id}.json", missing_ok=True)


def _validate_upload_storage(config: AppConfig) -> None:
    if _upload_bytes(config.data_dir) > config.server.max_upload_storage_bytes:
        raise QueueFullError(
            f"Upload storage exceeds {config.server.max_upload_storage_bytes} bytes"
        )


def _upload_bytes(data_dir: Path) -> int:
    return sum(path.stat().st_size for path in (data_dir / "uploads").glob("*") if path.is_file())


def reconcile_queue_storage(config: AppConfig) -> None:
    now = time.time()
    active_job_ids: set[str] = set()
    with _queue_lock(config.data_dir):
        reservations = list(_reservation_dir(config.data_dir).glob("*.json"))
        for path in reservations:
            if now - path.stat().st_mtime > config.server.upload_reservation_timeout_seconds:
                durable_unlink(path, missing_ok=True)
        jobs = list_jobs(config.data_dir)
        active_job_ids = {job.job_id for job in jobs if job.status in {"queued", "running"}}
        referenced = {Path(job.upload_path).resolve() for job in jobs}
        for job in jobs:
            upload = Path(job.upload_path)
            if job.status == "succeeded":
                durable_unlink(upload, missing_ok=True)
            elif job.status == "failed" and job.finished_at is not None:
                age = (_now() - job.finished_at).total_seconds()
                if age >= config.server.failed_upload_retention_seconds:
                    durable_unlink(upload, missing_ok=True)
        reserved_ids = {path.stem for path in _reservation_dir(config.data_dir).glob("*.json")}
        for upload in (config.data_dir / "uploads").glob("*"):
            job_id = upload.name.split("_", 1)[0]
            if upload.resolve() in referenced or job_id in reserved_ids:
                continue
            if now - upload.stat().st_mtime >= config.server.failed_upload_retention_seconds:
                durable_unlink(upload, missing_ok=True)
    reconcile_source_identity_reservations(
        config.data_dir,
        active_job_ids,
        orphan_job_seconds=config.server.identity_orphan_job_seconds,
        direct_seconds=config.server.identity_direct_reservation_seconds,
    )


@contextmanager
def _worker_heartbeat(config: AppConfig, worker_id: str):  # noqa: ANN202
    stop = threading.Event()
    path = _heartbeat_path(config.data_dir)

    def update() -> None:
        while not stop.is_set():
            _write_heartbeat(
                path,
                json.dumps({"worker_id": worker_id, "updated_at": _now().isoformat()}),
            )
            stop.wait(config.server.worker_heartbeat_seconds)

    _write_heartbeat(
        path,
        json.dumps({"worker_id": worker_id, "updated_at": _now().isoformat()}),
    )
    thread = threading.Thread(target=update, name="scribebase-worker-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=config.server.worker_heartbeat_seconds + 1)
        durable_unlink(path, missing_ok=True)


def _heartbeat_path(data_dir: Path) -> Path:
    return data_dir / "jobs" / ".worker-heartbeat"


def _write_heartbeat(path: Path, content: str) -> None:
    """Atomically publish ephemeral liveness state without forcing storage flushes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_text_atomic(path: Path, content: str) -> None:
    atomic_write_text(path, content)


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
