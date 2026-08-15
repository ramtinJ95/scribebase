from datetime import datetime, timezone

import pytest

from scribebase.config import default_config
from scribebase.embeddings.profile import embedding_profile_fingerprint
from scribebase.models import Chunk, SearchFilters, SearchResult, SourceManifest
from scribebase.retrieval.search import _validate_manifest_profiles, search_chunks


def _manifest(tmp_path) -> SourceManifest:  # noqa: ANN001
    now = datetime.now(timezone.utc)
    return SourceManifest(
        source_id="source-1",
        title="Source",
        original_path="source.md",
        data_dir=str(tmp_path / "sources" / "source-1"),
        created_at=now,
        updated_at=now,
    )


def _chunk(profile: str | None) -> Chunk:
    return Chunk(
        chunk_id="chunk-1",
        source_id="source-1",
        source_type="notes",
        title="Source",
        chunk_index=0,
        text="text",
        file_path="document.md",
        extraction_method="markdown",
        embedding_profile_fingerprint=profile,
    )


def test_manifest_profile_validation_fails_closed_on_missing_metadata(
    tmp_path, monkeypatch
) -> None:  # noqa: ANN001
    config = default_config()
    config.data_dir = tmp_path
    manifest = _manifest(tmp_path)
    manifest.embedding_summary.indexed_in_weaviate = True
    manifest.embedding_summary.weaviate_collection = config.weaviate.collection
    manifest.embedding_summary.embedding_dimension = 2048
    monkeypatch.setattr("scribebase.retrieval.search.list_manifests", lambda _: [manifest])

    with pytest.raises(RuntimeError, match="profile metadata is missing"):
        _validate_manifest_profiles(
            config,
            embedding_profile_fingerprint(config.embedding, 2048),
            2048,
        )


def test_search_rejects_chunk_with_different_profile(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    config = default_config()
    config.data_dir = tmp_path

    class Embedder:
        def __init__(self, _config) -> None:  # noqa: ANN001
            pass

        def embed_query(self, _query):  # noqa: ANN001
            return [1.0, 0.0]

    class Store:
        def __init__(self, _config) -> None:  # noqa: ANN001
            pass

        def hybrid_search(self, **_kwargs):  # noqa: ANN003
            return [SearchResult(chunk=_chunk(None))]

        def close(self) -> None:
            pass

    monkeypatch.setattr("scribebase.retrieval.search.LlamaCppEmbeddingClient", Embedder)
    monkeypatch.setattr("scribebase.retrieval.search.WeaviateStore", Store)
    monkeypatch.setattr("scribebase.retrieval.search.list_manifests", lambda _: [])

    with pytest.raises(RuntimeError, match="profile mismatch in search results"):
        search_chunks("query", SearchFilters(), config)
