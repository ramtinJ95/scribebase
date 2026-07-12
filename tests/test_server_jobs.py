from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO

import pytest

from scribebase.config import default_config
from scribebase.models import SourceManifest
from scribebase.server_jobs import (
    QueueFullError,
    UnsupportedUploadError,
    UploadTooLargeError,
    _worker_lock,
    claim_next_job,
    create_ingest_job,
    read_job,
    recover_interrupted_jobs,
    run_ingest_job,
    run_worker,
    write_job,
)
from scribebase.source_registry import write_manifest


def test_run_ingest_job_marks_success_and_indexes(tmp_path, monkeypatch) -> None:
    config = default_config()
    config.data_dir = tmp_path
    job = create_ingest_job(
        config,
        "../unsafe name.pdf",
        BytesIO(b"pdf"),
        "Uploaded PDF",
        "paper",
        None,
        None,
        "en",
        "auto",
        False,
        False,
    )
    indexed: list[str] = []

    def fake_extract(
        input_path,
        title,
        source_type,
        course,
        chapter,
        language,
        ocr,
        config,
        logger,
        cont,
        **metadata,
    ):
        assert input_path.name == f"{job.job_id}_unsafe_name.pdf"
        assert title == "Uploaded PDF"
        assert source_type == "paper"
        assert language == "en"
        assert metadata["tags"] == []
        now = datetime.now(timezone.utc)
        return SourceManifest(
            source_id="source-1",
            title=title,
            source_type=source_type,
            original_path=str(input_path),
            data_dir=str(tmp_path / "sources" / "source-1"),
            created_at=now,
            updated_at=now,
        )

    def fake_index(source_id, config, logger):
        indexed.append(source_id)
        return None

    monkeypatch.setattr("scribebase.server_jobs.extract_source", fake_extract)
    monkeypatch.setattr("scribebase.server_jobs.index_source", fake_index)

    run_ingest_job(job.job_id, config)

    saved = read_job(tmp_path, job.job_id)
    assert saved.status == "succeeded"
    assert saved.source_id == "source-1"
    assert saved.error is None
    assert saved.started_at is not None
    assert saved.finished_at is not None
    assert indexed == ["source-1"]


def test_run_ingest_job_marks_failure(tmp_path, monkeypatch) -> None:
    config = default_config()
    config.data_dir = tmp_path
    job = create_ingest_job(
        config,
        "paper.pdf",
        BytesIO(b"pdf"),
        "Uploaded PDF",
        "paper",
        None,
        None,
        "en",
        "auto",
        False,
        False,
    )
    monkeypatch.setattr(
        "scribebase.server_jobs.extract_source",
        lambda *_, **__: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    run_ingest_job(job.job_id, config)

    saved = read_job(tmp_path, job.job_id)
    assert saved.status == "failed"
    assert saved.error == "boom"


def test_run_ingest_job_passes_generic_metadata(tmp_path, monkeypatch) -> None:
    config = default_config()
    config.data_dir = tmp_path
    job = create_ingest_job(
        config,
        "notes.txt",
        BytesIO(b"note"),
        "Uploaded Notes",
        "notes",
        None,
        None,
        "en",
        "auto",
        True,
        False,
        tags="kubernetes, notes",
        origin="manual",
        publisher="Personal",
        author="Ramtin",
        created_at_source="2026-07-08",
        retrieved_at="2026-07-08T12:00:00Z",
        url="https://example.com/note",
        external_id="note-1",
        collection="kubernetes-reading",
        summary="A note.",
    )

    def fake_extract(
        input_path,
        title,
        source_type,
        course,
        chapter,
        language,
        ocr,
        config,
        logger,
        cont,
        **metadata,
    ):
        assert metadata["tags"] == ["kubernetes", "notes"]
        assert metadata["origin"] == "manual"
        assert metadata["publisher"] == "Personal"
        assert metadata["created_at_source"].isoformat().startswith("2026-07-08")
        assert metadata["url"] == "https://example.com/note"
        assert metadata["external_id"] == "note-1"
        assert metadata["collection"] == "kubernetes-reading"
        assert metadata["summary"] == "A note."
        now = datetime.now(timezone.utc)
        return SourceManifest(
            source_id="source-1",
            title=title,
            source_type=source_type,
            original_path=str(input_path),
            data_dir=str(tmp_path / "sources" / "source-1"),
            created_at=now,
            updated_at=now,
        )

    monkeypatch.setattr("scribebase.server_jobs.extract_source", fake_extract)

    run_ingest_job(job.job_id, config)

    saved = read_job(tmp_path, job.job_id)
    assert saved.status == "succeeded"
    assert saved.tags == ["kubernetes", "notes"]


def test_create_job_rejects_oversized_upload_and_removes_partial_file(tmp_path) -> None:
    config = default_config()
    config.data_dir = tmp_path
    config.server.max_upload_bytes = 3

    with pytest.raises(UploadTooLargeError, match="exceeds 3 bytes"):
        create_ingest_job(
            config,
            "notes.txt",
            BytesIO(b"four"),
            "Notes",
            "notes",
            None,
            None,
            "en",
            "auto",
            False,
            False,
        )

    assert not list((tmp_path / "uploads").glob("*"))


def test_create_job_rejects_unsupported_upload_before_writing(tmp_path) -> None:
    config = default_config()
    config.data_dir = tmp_path

    with pytest.raises(UnsupportedUploadError, match=".zip"):
        create_ingest_job(
            config,
            "archive.zip",
            BytesIO(b"zip"),
            "Archive",
            "other",
            None,
            None,
            "en",
            "auto",
            False,
            False,
        )

    assert not list((tmp_path / "uploads").glob("*"))


def test_create_job_enforces_active_queue_capacity(tmp_path) -> None:
    config = default_config()
    config.data_dir = tmp_path
    config.server.max_active_jobs = 1
    create_ingest_job(
        config, "one.txt", BytesIO(b"one"), "One", "notes", None, None, "en", "auto", False, False
    )

    with pytest.raises(QueueFullError, match="1/1"):
        create_ingest_job(
            config,
            "two.txt",
            BytesIO(b"two"),
            "Two",
            "notes",
            None,
            None,
            "en",
            "auto",
            False,
            False,
        )

    assert len(list((tmp_path / "uploads").glob("*"))) == 1


def test_recover_and_claim_interrupted_job(tmp_path) -> None:
    config = default_config()
    config.data_dir = tmp_path
    job = create_ingest_job(
        config,
        "notes.txt",
        BytesIO(b"note"),
        "Notes",
        "notes",
        None,
        None,
        "en",
        "auto",
        False,
        False,
    )
    job.status = "running"
    job.attempts = 1
    write_job(tmp_path, job)

    assert recover_interrupted_jobs(tmp_path) == 1
    claimed = claim_next_job(tmp_path)

    assert claimed is not None
    assert claimed.job_id == job.job_id
    assert claimed.status == "running"
    assert claimed.attempts == 2


def test_worker_once_recovers_and_processes_one_job(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    config = default_config()
    config.data_dir = tmp_path
    job = create_ingest_job(
        config,
        "notes.txt",
        BytesIO(b"note"),
        "Notes",
        "notes",
        None,
        None,
        "en",
        "auto",
        False,
        False,
    )
    processed = []
    monkeypatch.setattr(
        "scribebase.server_jobs.run_ingest_job", lambda job_id, _config: processed.append(job_id)
    )

    run_worker(config, once=True)

    assert processed == [job.job_id]
    assert read_job(tmp_path, job.job_id).status == "running"


def test_worker_lock_rejects_second_worker(tmp_path) -> None:
    with _worker_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already running"):
            with _worker_lock(tmp_path):
                pass


def test_recovered_job_with_source_id_resumes_at_indexing(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    config = default_config()
    config.data_dir = tmp_path
    job = create_ingest_job(
        config,
        "notes.txt",
        BytesIO(b"note"),
        "Notes",
        "notes",
        None,
        None,
        "en",
        "auto",
        False,
        False,
    )
    now = datetime.now(timezone.utc)
    source_root = tmp_path / "sources" / "source-1"
    manifest = SourceManifest(
        source_id="source-1",
        title="Notes",
        source_type="notes",
        original_path=str(source_root / "original" / "notes.txt"),
        data_dir=str(source_root),
        created_at=now,
        updated_at=now,
    )
    write_manifest(manifest)
    job.source_id = manifest.source_id
    job.status = "running"
    write_job(tmp_path, job)
    indexed = []
    monkeypatch.setattr(
        "scribebase.server_jobs.extract_source",
        lambda *_args, **_kwargs: pytest.fail("extraction should not be repeated"),
    )
    monkeypatch.setattr(
        "scribebase.server_jobs.index_source", lambda source_id, *_args: indexed.append(source_id)
    )

    run_ingest_job(job.job_id, config)

    assert indexed == ["source-1"]
    assert read_job(tmp_path, job.job_id).status == "succeeded"
