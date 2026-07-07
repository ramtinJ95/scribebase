from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from scribebase.config import default_config
from scribebase.models import Chunk, SearchResult, SourceManifest
from scribebase.paths import ensure_data_layout
from scribebase.source_registry import write_manifest
from scribebase.server import ServiceHealth, create_app


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
    return TestClient(create_app(config, api_token=TOKEN))


def _auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def test_health_reports_readiness(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["data_dir"] == str(tmp_path)
    assert body["auth_required"] is True
    assert body["weaviate"] == {"ok": True, "message": "ready"}


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
            "filters": {"source_type": "book"},
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
