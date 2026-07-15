from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import pytest

from scribebase.config import default_config
from scribebase.errors import DependencyUnavailableError
from scribebase.extraction import extract_source
from scribebase.logging_utils import setup_logging
from scribebase.models import SourceManifest
from scribebase.server_jobs import (
    QueueFullError,
    UnsupportedUploadError,
    UploadTooLargeError,
    _worker_heartbeat,
    _worker_lock,
    claim_next_job,
    create_ingest_job,
    list_jobs,
    read_job,
    reconcile_queue_storage,
    recover_interrupted_jobs,
    retry_job,
    run_ingest_job,
    run_worker,
    write_job,
)
from scribebase.source_registry import list_manifests, write_manifest
from scribebase.source_registry import identity_reservation_owned


def _claim(config, job):  # noqa: ANN001, ANN202
    claimed = claim_next_job(config.data_dir, "test-worker")
    assert claimed is not None
    assert claimed.job_id == job.job_id
    return claimed


def _extracted_index_job(config, tmp_path):  # noqa: ANN001, ANN202
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
    source_root = tmp_path / "sources" / (job.source_id or "")
    write_manifest(
        SourceManifest(
            source_id=job.source_id or "",
            title="Notes",
            source_type="notes",
            original_path=str(source_root / "original" / "notes.txt"),
            data_dir=str(source_root),
            created_at=now,
            updated_at=now,
        )
    )
    job.phase = "extracted"
    write_job(tmp_path, job)
    return job


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
            source_id=metadata["source_id"],
            title=title,
            source_type=source_type,
            original_path=str(input_path),
            data_dir=str(tmp_path / "sources" / "source-1"),
            created_at=now,
            updated_at=now,
        )

    def fake_index(source_id, config, logger, operation_id=None):
        assert operation_id == job.job_id
        indexed.append(source_id)
        return None

    monkeypatch.setattr("scribebase.server_jobs.extract_source", fake_extract)
    monkeypatch.setattr("scribebase.server_jobs.index_source", fake_index)
    monkeypatch.setattr(
        "scribebase.server_jobs._index_dependencies_ready", lambda _config: (True, "ready")
    )

    claimed = _claim(config, job)
    run_ingest_job(job.job_id, claimed.claim_token or "", config)

    saved = read_job(tmp_path, job.job_id)
    assert saved.status == "succeeded"
    assert saved.source_id == job.source_id
    assert saved.error is None
    assert saved.started_at is not None
    assert saved.finished_at is not None
    assert indexed == [job.source_id]


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

    claimed = _claim(config, job)
    run_ingest_job(job.job_id, claimed.claim_token or "", config)

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
            source_id=metadata["source_id"],
            title=title,
            source_type=source_type,
            original_path=str(input_path),
            data_dir=str(tmp_path / "sources" / "source-1"),
            created_at=now,
            updated_at=now,
        )

    monkeypatch.setattr("scribebase.server_jobs.extract_source", fake_extract)

    claimed = _claim(config, job)
    run_ingest_job(job.job_id, claimed.claim_token or "", config)

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
    claimed = claim_next_job(tmp_path, "test-worker")

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
        "scribebase.server_jobs.run_ingest_job",
        lambda job_id, _claim_token, _config: processed.append(job_id),
    )

    run_worker(config, once=True)

    assert processed == [job.job_id]
    assert read_job(tmp_path, job.job_id).status == "running"


def test_worker_fails_startup_on_permanent_index_recovery_error(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    config = default_config()
    config.data_dir = tmp_path
    (tmp_path / ".index-transaction.json").write_text("{")
    monkeypatch.setattr(
        "scribebase.server_jobs._weaviate_ready", lambda _config: (True, "ready")
    )

    with pytest.raises(RuntimeError, match="Unreadable index recovery journal"):
        run_worker(config, once=True)

    assert not (tmp_path / "jobs" / ".worker-heartbeat").exists()


def test_worker_waits_without_heartbeat_while_recovery_dependency_is_down(
    tmp_path, monkeypatch
) -> None:  # noqa: ANN001
    config = default_config()
    config.data_dir = tmp_path
    (tmp_path / ".index-transaction.json").write_text("pending")
    monkeypatch.setattr(
        "scribebase.server_jobs._weaviate_ready",
        lambda _config: (False, "Weaviate is starting"),
    )
    monkeypatch.setattr(
        "scribebase.server_jobs.recover_index_transactions",
        lambda *_args: pytest.fail("recovery must wait for Weaviate"),
    )

    run_worker(config, once=True)

    assert not (tmp_path / "jobs" / ".worker-heartbeat").exists()


def test_worker_retries_when_weaviate_drops_during_recovery(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    config = default_config()
    config.data_dir = tmp_path
    pending = iter([True, True, False])
    recovery_attempts = []
    sleeps = []

    monkeypatch.setattr(
        "scribebase.server_jobs.index_recovery_pending", lambda _data_dir: next(pending)
    )
    monkeypatch.setattr(
        "scribebase.server_jobs._weaviate_ready", lambda _config: (True, "ready")
    )

    def recover(*_args) -> None:  # noqa: ANN002
        recovery_attempts.append(True)
        if len(recovery_attempts) == 1:
            raise DependencyUnavailableError("connection dropped")

    monkeypatch.setattr("scribebase.server_jobs.recover_index_transactions", recover)
    monkeypatch.setattr("scribebase.server_jobs.time.sleep", sleeps.append)
    monkeypatch.setattr(
        "scribebase.server_jobs.claim_next_job",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("worker loop reached")),
    )

    with pytest.raises(RuntimeError, match="worker loop reached"):
        run_worker(config, once=False)

    assert len(recovery_attempts) == 2
    assert sleeps == [config.server.worker_dependency_retry_seconds]


def test_worker_lock_rejects_second_worker(tmp_path) -> None:
    with _worker_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already running"):
            with _worker_lock(tmp_path):
                pass


def test_worker_heartbeat_does_not_use_durable_state_writes(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    config = default_config()
    config.data_dir = tmp_path
    config.server.worker_heartbeat_seconds = 60
    monkeypatch.setattr(
        "scribebase.server_jobs.atomic_write_text",
        lambda *_args: pytest.fail("heartbeat must not force a durable storage flush"),
    )

    with _worker_heartbeat(config, "test-worker"):
        heartbeat = tmp_path / "jobs" / ".worker-heartbeat"
        assert json.loads(heartbeat.read_text())["worker_id"] == "test-worker"

    assert not heartbeat.exists()


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
    source_root = tmp_path / "sources" / (job.source_id or "")
    manifest = SourceManifest(
        source_id=job.source_id or "",
        title="Notes",
        source_type="notes",
        original_path=str(source_root / "original" / "notes.txt"),
        data_dir=str(source_root),
        created_at=now,
        updated_at=now,
    )
    write_manifest(manifest)
    job.phase = "extracted"
    write_job(tmp_path, job)
    indexed = []
    monkeypatch.setattr(
        "scribebase.server_jobs.extract_source",
        lambda *_args, **_kwargs: pytest.fail("extraction should not be repeated"),
    )
    monkeypatch.setattr(
        "scribebase.server_jobs.index_source",
        lambda source_id, *_args, **_kwargs: indexed.append(source_id),
    )
    monkeypatch.setattr(
        "scribebase.server_jobs._index_dependencies_ready", lambda _config: (True, "ready")
    )

    claimed = _claim(config, job)
    run_ingest_job(job.job_id, claimed.claim_token or "", config)

    assert indexed == [job.source_id]
    assert read_job(tmp_path, job.job_id).status == "succeeded"


def test_indexing_job_waits_for_dependencies_without_losing_upload(tmp_path, monkeypatch) -> None:  # noqa: ANN001
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
    source_root = tmp_path / "sources" / (job.source_id or "")
    write_manifest(
        SourceManifest(
            source_id=job.source_id or "",
            title="Notes",
            source_type="notes",
            original_path=str(source_root / "original" / "notes.txt"),
            data_dir=str(source_root),
            created_at=now,
            updated_at=now,
        )
    )
    job.phase = "extracted"
    write_job(tmp_path, job)
    monkeypatch.setattr(
        "scribebase.server_jobs._index_dependencies_ready",
        lambda _config: (False, "Weaviate is starting"),
    )
    monkeypatch.setattr(
        "scribebase.server_jobs.index_source",
        lambda *_args, **_kwargs: pytest.fail("indexing must wait for dependencies"),
    )

    claimed = _claim(config, job)
    run_ingest_job(job.job_id, claimed.claim_token or "", config)

    saved = read_job(tmp_path, job.job_id)
    assert saved.status == "queued"
    assert saved.phase == "indexing"
    assert saved.next_attempt_at is not None
    assert saved.finished_at is None
    assert "Weaviate is starting" in (saved.error or "")
    assert Path(saved.upload_path).read_bytes() == b"note"
    assert claim_next_job(tmp_path, "too-early") is None


def test_deterministic_index_error_is_not_reclassified_by_later_health(
    tmp_path, monkeypatch
) -> None:  # noqa: ANN001
    config = default_config()
    config.data_dir = tmp_path
    job = _extracted_index_job(config, tmp_path)
    readiness_checks = 0

    def ready_once(_config):  # noqa: ANN001, ANN202
        nonlocal readiness_checks
        readiness_checks += 1
        if readiness_checks > 1:
            pytest.fail("index errors must not be reclassified by a second health probe")
        return True, "ready"

    monkeypatch.setattr("scribebase.server_jobs._index_dependencies_ready", ready_once)
    monkeypatch.setattr(
        "scribebase.server_jobs.index_source",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("invalid chunks")),
    )

    claimed = _claim(config, job)
    run_ingest_job(job.job_id, claimed.claim_token or "", config)

    saved = read_job(tmp_path, job.job_id)
    assert saved.status == "failed"
    assert saved.error == "invalid chunks"
    assert readiness_checks == 1


def test_typed_index_dependency_failure_requeues_even_after_preflight(
    tmp_path, monkeypatch
) -> None:  # noqa: ANN001
    config = default_config()
    config.data_dir = tmp_path
    job = _extracted_index_job(config, tmp_path)
    monkeypatch.setattr(
        "scribebase.server_jobs._index_dependencies_ready", lambda _config: (True, "ready")
    )
    monkeypatch.setattr(
        "scribebase.server_jobs.index_source",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            DependencyUnavailableError("connection reset")
        ),
    )

    claimed = _claim(config, job)
    run_ingest_job(job.job_id, claimed.claim_token or "", config)

    saved = read_job(tmp_path, job.job_id)
    assert saved.status == "queued"
    assert "connection reset" in (saved.error or "")


def test_recovery_reuses_preassigned_source_after_extraction_crash(tmp_path) -> None:
    config = default_config()
    config.data_dir = tmp_path
    job = create_ingest_job(
        config,
        "notes.txt",
        BytesIO(b"durable note"),
        "Durable Notes",
        "notes",
        None,
        None,
        "en",
        "auto",
        True,
        False,
    )
    extract_source(
        Path(job.upload_path),
        job.title,
        job.source_type,
        job.course,
        job.chapter,
        job.language,
        job.ocr,
        config,
        setup_logging(tmp_path),
        source_id=job.source_id,
        identity_owner=job.job_id,
    )
    job.status = "running"
    job.phase = "extracting"
    write_job(tmp_path, job)

    assert recover_interrupted_jobs(tmp_path) == 1
    claimed = _claim(config, job)
    run_ingest_job(job.job_id, claimed.claim_token or "", config)

    assert [manifest.source_id for manifest in list_manifests(tmp_path)] == [job.source_id]
    assert read_job(tmp_path, job.job_id).phase == "completed"


def test_recovery_replays_completed_index_operation_to_verify_weaviate(tmp_path, monkeypatch) -> None:  # noqa: ANN001
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
    source_root = tmp_path / "sources" / (job.source_id or "")
    manifest = SourceManifest(
        source_id=job.source_id or "",
        title="Notes",
        source_type="notes",
        original_path=str(source_root / "original" / "notes.txt"),
        data_dir=str(source_root),
        created_at=now,
        updated_at=now,
    )
    manifest.embedding_summary.index_operation_id = job.job_id
    write_manifest(manifest)
    job.status = "running"
    job.phase = "indexing"
    write_job(tmp_path, job)
    replayed = []
    monkeypatch.setattr(
        "scribebase.server_jobs.index_source",
        lambda source_id, *_args, **_kwargs: replayed.append(source_id),
    )
    monkeypatch.setattr(
        "scribebase.server_jobs._index_dependencies_ready", lambda _config: (True, "ready")
    )

    recover_interrupted_jobs(tmp_path)
    claimed = _claim(config, job)
    run_ingest_job(job.job_id, claimed.claim_token or "", config)

    saved = read_job(tmp_path, job.job_id)
    assert saved.status == "succeeded"
    assert saved.phase == "completed"
    assert replayed == [job.source_id]


def test_corrupt_job_is_quarantined_without_blocking_queue(tmp_path) -> None:
    config = default_config()
    config.data_dir = tmp_path
    valid = create_ingest_job(
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
    (tmp_path / "jobs" / "broken.json").write_text("{")

    with pytest.warns(UserWarning, match="Quarantined corrupt job"):
        jobs = list_jobs(tmp_path)

    assert [job.job_id for job in jobs] == [valid.job_id]
    assert not (tmp_path / "jobs" / "broken.json").exists()
    assert list((tmp_path / "jobs").glob("broken.corrupt.*"))


def test_queue_reservation_rejects_before_reading_upload(tmp_path) -> None:
    config = default_config()
    config.data_dir = tmp_path
    config.server.max_active_jobs = 1
    create_ingest_job(
        config,
        "one.txt",
        BytesIO(b"one"),
        "One",
        "notes",
        None,
        None,
        "en",
        "auto",
        False,
        False,
    )

    class MustNotRead:
        def read(self, _size):  # noqa: ANN001, ANN201
            raise AssertionError("upload was read before capacity rejection")

    with pytest.raises(QueueFullError):
        create_ingest_job(
            config,
            "two.txt",
            MustNotRead(),
            "Two",
            "notes",
            None,
            None,
            "en",
            "auto",
            False,
            False,
        )


def test_queue_reconciliation_removes_expired_failed_upload(tmp_path) -> None:
    config = default_config()
    config.data_dir = tmp_path
    config.server.failed_upload_retention_seconds = 0
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
    job.status = "failed"
    job.finished_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    write_job(tmp_path, job)

    reconcile_queue_storage(config)

    assert not Path(job.upload_path).exists()


def test_losing_concurrent_retry_keeps_winner_identity_reservation(tmp_path) -> None:
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
    job.status = "failed"
    write_job(tmp_path, job)
    from scribebase.source_registry import release_source_identity

    release_source_identity(tmp_path, job.identity_key or "", job.job_id)

    def retry():  # noqa: ANN202
        try:
            return retry_job(config, job.job_id).status
        except ValueError:
            return "lost"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: retry(), range(2)))

    assert sorted(results) == ["lost", "queued"]
    assert identity_reservation_owned(tmp_path, job.identity_key or "", job.job_id)
