from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO

from scribebase.config import default_config
from scribebase.models import SourceManifest
from scribebase.server_jobs import create_ingest_job, read_job, run_ingest_job


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
    ):
        assert input_path.name == f"{job.job_id}_unsafe_name.pdf"
        assert title == "Uploaded PDF"
        assert source_type == "paper"
        assert language == "en"
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
