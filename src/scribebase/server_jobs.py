from __future__ import annotations

import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Literal
from uuid import uuid4

from pydantic import BaseModel

from scribebase.config import AppConfig
from scribebase.extraction import extract_source
from scribebase.indexing import index_source
from scribebase.logging_utils import setup_logging
from scribebase.models import Language, SourceType
from scribebase.paths import ensure_data_layout

JobStatus = Literal["queued", "running", "succeeded", "failed"]


class IngestJob(BaseModel):
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


class IngestJobResponse(BaseModel):
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
    title: str,
    source_type: SourceType,
    course: str | None,
    chapter: str | None,
    language: Language,
    ocr: str,
    no_index: bool,
    continue_on_ocr_error: bool,
) -> IngestJob:
    ensure_data_layout(config.data_dir)
    job_id = uuid4().hex
    safe_name = _safe_filename(filename)
    upload_path = config.data_dir / "uploads" / f"{job_id}_{safe_name}"
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    with upload_path.open("wb") as out:
        shutil.copyfileobj(fileobj, out)

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
