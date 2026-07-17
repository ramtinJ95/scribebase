from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from scribebase.config import default_config
from scribebase.models import Chunk, SearchResult, SourceManifest
from scribebase.paths import ensure_data_layout
from scribebase.source_registry import write_manifest
from scribebase.server import ServiceHealth, create_app
from scribebase.server_jobs import _worker_heartbeat, _worker_lock, read_job, write_job


TOKEN = "test-token"


def _client(tmp_path, monkeypatch) -> TestClient:
    config = default_config()
    config.data_dir = tmp_path
    ensure_data_layout(config.data_dir)
    monkeypatch.setattr(
        "scribebase.server._weaviate_health",
        lambda _: ServiceHealth(ok=True, message="ready"),
    )
    monkeypatch.setattr(
        "scribebase.server._embedding_health",
        lambda _: ServiceHealth(ok=True, message="ready"),
    )
    monkeypatch.setattr(
        "scribebase.server._ocr_health",
        lambda _: ServiceHealth(ok=True, message="GLM-OCR ready"),
    )
    return TestClient(create_app(config, api_token=TOKEN))


def _auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def test_health_reports_readiness(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["data_dir"] == str(tmp_path)
    assert body["auth_required"] is True
    assert body["weaviate"] == {"ok": True, "message": "ready"}
    assert body["ocr"] == {"ok": True, "message": "GLM-OCR ready"}
    assert body["worker"] == {"ok": False, "message": "worker is not running"}


def test_health_reports_running_worker(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    with _worker_lock(tmp_path):
        with _worker_heartbeat(client.app.state.config, "test-worker"):
            response = client.get("/health")

    assert response.json()["worker"]["ok"] is True
    assert "test-worker" in response.json()["worker"]["message"]
    assert response.json()["status"] == "ok"


def test_health_is_degraded_when_glm_ocr_is_unavailable(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "scribebase.server._ocr_health",
        lambda _: ServiceHealth(ok=False, message="GLM-OCR unavailable; no fallback"),
    )

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["ocr"] == {
        "ok": False,
        "message": "GLM-OCR unavailable; no fallback",
    }


def test_sources_requires_bearer_auth(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.get("/sources")

    assert response.status_code == 401


def test_sources_lists_manifests(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    now = datetime.now(timezone.utc)
    write_manifest(
        SourceManifest(
            source_id="source-1",
            title="Source One",
            source_type="book",
            original_path=str(tmp_path / "source.pdf"),
            data_dir=str(tmp_path / "sources" / "source-1"),
            created_at=now,
            updated_at=now,
        )
    )

    response = client.get("/sources", headers=_auth())

    assert response.status_code == 200
    assert response.json()[0]["source_id"] == "source-1"


def test_search_returns_results(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    def fake_search(query, filters, config, top_k, alpha, allow_model_mismatch):
        assert query == "kubelet eviction"
        assert filters.source_type == "book"
        assert filters.tags == ["kubernetes", "ops"]
        assert filters.origin == "company_blog"
        assert filters.collection == "infra-reading"
        assert filters.created_at_source_after.isoformat().startswith("2026-07-01")
        assert top_k == 3
        assert alpha == 0.5
        assert allow_model_mismatch is True
        return [_result()]

    monkeypatch.setattr("scribebase.server.search_chunks", fake_search)

    response = client.post(
        "/search",
        headers=_auth(),
        json={
            "query": "kubelet eviction",
            "filters": {
                "source_type": "book",
                "tags": "kubernetes, ops",
                "origin": "company_blog",
                "collection": "infra-reading",
                "created_at_source_after": "2026-07-01T00:00:00Z",
            },
            "top_k": 3,
            "alpha": 0.5,
            "allow_model_mismatch": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "kubelet eviction"
    assert body["results"][0]["chunk"]["chunk_id"] == "chunk-1"


def test_context_returns_context_pack(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    monkeypatch.setattr("scribebase.server.search_chunks", lambda *_, **__: [_result()])

    response = client.post(
        "/context",
        headers=_auth(),
        json={"query": "what is eviction?", "task": "answer", "top_k": 1},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["task"] == "answer"
    assert "# Context Pack" in body["context_pack"]
    assert "Chunk ID: chunk-1" in body["context_pack"]


def test_ingest_upload_creates_job(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/ingest",
        headers=_auth(),
        data={"title": "Uploaded PDF", "source_type": "paper", "language": "en"},
        files={"file": ("paper.pdf", b"%PDF-1.7 test", "application/pdf")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "queued"
    assert body["title"] == "Uploaded PDF"
    assert body["source_type"] == "paper"
    assert body["language"] == "en"
    assert "upload_path" not in body
    assert (tmp_path / "jobs" / f"{body['job_id']}.json").exists()
    assert (tmp_path / "uploads" / f"{body['job_id']}_paper.pdf").read_bytes() == b"%PDF-1.7 test"


def test_ingest_upload_accepts_text_file(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/ingest",
        headers=_auth(),
        data={
            "title": "Uploaded Notes",
            "source_type": "notes",
            "language": "en",
            "tags": "kubernetes, notes",
            "origin": "manual",
            "collection": "kubernetes-reading",
        },
        files={"file": ("notes.txt", b"plain text notes", "text/plain")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "queued"
    assert body["title"] == "Uploaded Notes"
    assert body["source_type"] == "notes"
    assert body["tags"] == ["kubernetes", "notes"]
    assert body["origin"] == "manual"
    assert body["collection"] == "kubernetes-reading"
    assert (
        tmp_path / "uploads" / f"{body['job_id']}_notes.txt"
    ).read_bytes() == b"plain text notes"


def test_ingest_upload_uses_markdown_frontmatter_defaults(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/ingest",
        headers=_auth(),
        files={
            "file": (
                "article.md",
                b"---\n"
                b"title: Frontmatter Article\n"
                b"source_type: article\n"
                b"language: en\n"
                b"tags: [kubernetes, gitops]\n"
                b"origin: company_blog\n"
                b"collection: infra-reading\n"
                b"---\n\n"
                b"# Body\n",
                "text/markdown",
            )
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["title"] == "Frontmatter Article"
    assert body["source_type"] == "article"
    assert body["language"] == "en"
    assert body["tags"] == ["kubernetes", "gitops"]
    assert body["origin"] == "company_blog"
    assert body["collection"] == "infra-reading"


def test_ingest_upload_without_title_or_frontmatter_returns_400(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/ingest",
        headers=_auth(),
        files={"file": ("notes.txt", b"plain text notes", "text/plain")},
    )

    assert response.status_code == 400
    assert "title is required" in response.json()["detail"]
    assert list((tmp_path / "uploads").iterdir()) == []
    assert not list((tmp_path / "jobs").glob("*.json"))


def test_ingest_upload_rejects_oversized_file(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    client.app.state.config.server.max_upload_bytes = 3

    response = client.post(
        "/ingest",
        headers=_auth(),
        data={"title": "Too large"},
        files={"file": ("notes.txt", b"four", "text/plain")},
    )

    assert response.status_code == 413
    assert not list((tmp_path / "uploads").glob("*"))


def test_ingest_upload_rejects_unsupported_type(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/ingest",
        headers=_auth(),
        data={"title": "Archive"},
        files={"file": ("archive.zip", b"zip", "application/zip")},
    )

    assert response.status_code == 415
    assert not list((tmp_path / "uploads").glob("*"))


def test_ingest_rejects_when_queue_is_full(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    client.app.state.config.server.max_active_jobs = 1
    first = client.post(
        "/ingest",
        headers=_auth(),
        data={"title": "One"},
        files={"file": ("one.txt", b"one", "text/plain")},
    )

    second = client.post(
        "/ingest",
        headers=_auth(),
        data={"title": "Two"},
        files={"file": ("two.txt", b"two", "text/plain")},
    )

    assert first.status_code == 200
    assert second.status_code == 429
    assert len(list((tmp_path / "uploads").glob("*"))) == 1


def test_article_ingest_json_creates_markdown_job(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/articles",
        headers=_auth(),
        json={
            "title": "GitOps Article",
            "body": "# GitOps\n\nArgo CD reconciles declared state.",
            "language": "en",
            "tags": ["kubernetes", "gitops"],
            "origin": "company_blog",
            "publisher": "Example Blog",
            "author": "Author",
            "created_at_source": "2026-07-08T00:00:00Z",
            "url": "https://example.com/gitops",
            "external_id": "article-1",
            "collection": "infra-reading",
            "summary": "GitOps article.",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "queued"
    assert body["title"] == "GitOps Article"
    assert body["filename"] == "gitops_article.md"
    assert body["source_type"] == "article"
    assert body["language"] == "en"
    assert body["tags"] == ["kubernetes", "gitops"]
    assert body["origin"] == "company_blog"
    assert body["publisher"] == "Example Blog"
    assert body["collection"] == "infra-reading"
    assert (tmp_path / "uploads" / f"{body['job_id']}_gitops_article.md").read_text() == (
        "# GitOps\n\nArgo CD reconciles declared state."
    )


def test_article_duplicate_is_rejected_while_first_job_is_queued(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    payload = {"title": "Same Article", "body": "# Same\n\nContent."}

    first = client.post("/articles", headers=_auth(), json=payload)
    duplicate = client.post("/articles", headers=_auth(), json=payload)
    explicit_copy = client.post(
        "/articles",
        headers=_auth(),
        json={**payload, "duplicate_policy": "create"},
    )

    assert first.status_code == 200
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"]["source_id"] == first.json()["source_id"]
    assert explicit_copy.status_code == 200
    assert explicit_copy.json()["source_id"] != first.json()["source_id"]


def test_failed_unpublished_job_does_not_block_resubmission(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    payload = {"title": "Retry Article", "body": "# Retry\n\nContent."}
    first = client.post("/articles", headers=_auth(), json=payload)
    job = read_job(tmp_path, first.json()["job_id"])
    job.status = "failed"
    job.error = "extraction failed"
    job.finished_at = datetime.now(timezone.utc)
    write_job(tmp_path, job)

    second = client.post("/articles", headers=_auth(), json=payload)

    assert second.status_code == 200
    assert second.json()["source_id"] != first.json()["source_id"]


def test_article_ingest_json_uses_markdown_frontmatter_defaults(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/articles",
        headers=_auth(),
        json={
            "body": "---\n"
            "title: Frontmatter Article\n"
            "language: en\n"
            "tags: [kubernetes, gitops]\n"
            "origin: company_blog\n"
            "collection: infra-reading\n"
            "---\n\n"
            "# Body\n",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["title"] == "Frontmatter Article"
    assert body["source_type"] == "article"
    assert body["language"] == "en"
    assert body["tags"] == ["kubernetes", "gitops"]
    assert body["origin"] == "company_blog"
    assert body["collection"] == "infra-reading"


def test_article_ingest_json_rejects_empty_body(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/articles",
        headers=_auth(),
        json={"title": "Empty", "body": "\n\t  "},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "body must not be empty"
    assert list((tmp_path / "uploads").iterdir()) == []
    assert not list((tmp_path / "jobs").glob("*.json"))


def test_article_ingest_json_without_title_or_frontmatter_returns_400(
    tmp_path, monkeypatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/articles",
        headers=_auth(),
        json={"body": "# Untitled\n\nBody."},
    )

    assert response.status_code == 400
    assert "title is required" in response.json()["detail"]
    assert list((tmp_path / "uploads").iterdir()) == []
    assert not list((tmp_path / "jobs").glob("*.json"))


def test_job_status_returns_persisted_job(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    created = client.post(
        "/ingest",
        headers=_auth(),
        data={"title": "Uploaded PDF"},
        files={"file": ("paper.pdf", b"pdf", "application/pdf")},
    ).json()

    response = client.get(f"/jobs/{created['job_id']}", headers=_auth())

    assert response.status_code == 200
    assert response.json()["job_id"] == created["job_id"]


def test_missing_job_returns_404(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.get("/jobs/missing", headers=_auth())

    assert response.status_code == 404


def test_failed_job_can_be_retried(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    created = client.post(
        "/ingest",
        headers=_auth(),
        data={"title": "Retry me"},
        files={"file": ("notes.txt", b"note", "text/plain")},
    ).json()
    job = read_job(tmp_path, created["job_id"])
    job.status = "failed"
    job.error = "temporary failure"
    job.finished_at = datetime.now(timezone.utc)
    write_job(tmp_path, job)

    response = client.post(f"/jobs/{job.job_id}/retry", headers=_auth())

    assert response.status_code == 200
    assert response.json()["status"] == "queued"
    assert response.json()["error"] is None


def test_known_oversized_request_is_rejected_before_body_parsing(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    client.app.state.config.server.max_upload_bytes = 3

    response = client.post(
        "/articles",
        headers={**_auth(), "Content-Length": str(2 * 1024 * 1024)},
        content=b"not-json",
    )

    assert response.status_code == 413


def _result() -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id="chunk-1",
            source_id="source-1",
            source_type="book",
            title="Kubernetes Book",
            page_start=10,
            page_end=11,
            chunk_index=0,
            text="Kubelet eviction text.",
            file_path="document.md",
            extraction_method="pymupdf4llm",
            language="en",
        ),
        score=0.9,
    )
