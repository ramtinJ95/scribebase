from datetime import datetime, timezone

import pytest

from scribebase.config import default_config
from scribebase.indexing import index_source, rebuild_index
from scribebase.models import Chunk, SourceManifest


class Logger:
    def info(self, *args) -> None:  # noqa: ANN002
        pass

    def warning(self, *args) -> None:  # noqa: ANN002
        pass


def _manifest(tmp_path, source_id: str = "source-1") -> SourceManifest:  # noqa: ANN001
    root = tmp_path / "sources" / source_id
    (root / "metadata").mkdir(parents=True)
    return SourceManifest(
        source_id=source_id,
        title="Source",
        source_type="book",
        original_path=str(root / "original" / "source.pdf"),
        data_dir=str(root),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def _chunk(source_id: str, index: int, chunk_id: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id or f"{source_id}-{index}",
        source_id=source_id,
        source_type="book",
        title="Source",
        chunk_index=index,
        text=f"chunk {index}",
        file_path="document.md",
        extraction_method="markdown",
    )


def test_index_source_streams_batches_before_removing_stale_chunks(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    config = default_config()
    config.data_dir = tmp_path
    config.embedding.batch_size = 2
    manifest = _manifest(tmp_path)
    chunks = [_chunk(manifest.source_id, index) for index in range(3)]
    old = [_chunk(manifest.source_id, 0), _chunk(manifest.source_id, 9, "stale")]
    writes = []

    class Embedder:
        def __init__(self, _config) -> None:  # noqa: ANN001
            pass

        def embed_batches(self, texts):  # noqa: ANN001
            yield [[1.0, 0.0] for _ in texts[:2]]
            yield [[0.0, 1.0] for _ in texts[2:]]

    class Store:
        batches = []
        deleted = None

        def __init__(self, _config) -> None:  # noqa: ANN001
            pass

        def ensure_collection(self) -> None:
            pass

        def upsert_chunks(self, batch_chunks, vectors, collection_name=None) -> None:  # noqa: ANN001
            self.batches.append((list(batch_chunks), list(vectors), collection_name))

        def delete_chunks(self, chunk_ids) -> None:  # noqa: ANN001
            type(self).deleted = chunk_ids

        def close(self) -> None:
            pass

    monkeypatch.setattr("scribebase.indexing.find_source", lambda *_: manifest)
    monkeypatch.setattr("scribebase.indexing.chunk_source", lambda *_: chunks)
    monkeypatch.setattr("scribebase.indexing.load_chunks", lambda *_: old)
    monkeypatch.setattr("scribebase.indexing.LlamaCppEmbeddingClient", Embedder)
    monkeypatch.setattr("scribebase.indexing.WeaviateStore", Store)
    monkeypatch.setattr("scribebase.indexing.write_manifest", writes.append)

    result = index_source(manifest.source_id, config, Logger())

    assert [len(batch[0]) for batch in Store.batches] == [2, 1]
    assert Store.deleted == {"stale"}
    assert result.embedding_summary.embedding_dimension == 2
    assert len(writes) == 1
    assert (tmp_path / "sources" / manifest.source_id / "metadata" / "chunks.jsonl").exists()


def test_index_source_preserves_local_state_and_stale_vectors_on_batch_failure(
    tmp_path, monkeypatch
) -> None:  # noqa: ANN001
    config = default_config()
    config.data_dir = tmp_path
    config.embedding.batch_size = 1
    manifest = _manifest(tmp_path)
    chunks_path = tmp_path / "sources" / manifest.source_id / "metadata" / "chunks.jsonl"
    chunks_path.write_text("old local chunks\n")
    writes = []

    class Embedder:
        def __init__(self, _config) -> None:  # noqa: ANN001
            pass

        def embed_batches(self, texts):  # noqa: ANN001
            for _ in texts:
                yield [[1.0, 0.0]]

    class Store:
        calls = 0
        deleted = False

        def __init__(self, _config) -> None:  # noqa: ANN001
            pass

        def ensure_collection(self) -> None:
            pass

        def upsert_chunks(self, *_args, **_kwargs) -> None:
            type(self).calls += 1
            if type(self).calls == 2:
                raise RuntimeError("batch failed")

        def delete_chunks(self, _chunk_ids) -> None:  # noqa: ANN001
            type(self).deleted = True

        def close(self) -> None:
            pass

    monkeypatch.setattr("scribebase.indexing.find_source", lambda *_: manifest)
    monkeypatch.setattr(
        "scribebase.indexing.chunk_source",
        lambda *_: [_chunk(manifest.source_id, 0), _chunk(manifest.source_id, 1)],
    )
    monkeypatch.setattr(
        "scribebase.indexing.load_chunks", lambda *_: [_chunk(manifest.source_id, 9, "stale")]
    )
    monkeypatch.setattr("scribebase.indexing.LlamaCppEmbeddingClient", Embedder)
    monkeypatch.setattr("scribebase.indexing.WeaviateStore", Store)
    monkeypatch.setattr("scribebase.indexing.write_manifest", writes.append)

    with pytest.raises(RuntimeError, match="batch failed"):
        index_source(manifest.source_id, config, Logger())

    assert chunks_path.read_text() == "old local chunks\n"
    assert not Store.deleted
    assert writes == []


def test_full_rebuild_promotes_only_after_count_verification(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    config = default_config()
    config.data_dir = tmp_path
    manifests = [_manifest(tmp_path, "source-1"), _manifest(tmp_path, "source-2")]
    events = []

    class Store:
        def __init__(self, _config) -> None:  # noqa: ANN001
            pass

        def create_collection(self, name: str) -> None:
            events.append(("create", name))

        def object_count(self, name: str) -> int:
            events.append(("count", name))
            return 2

        def promote_collection(self, name: str):  # noqa: ANN201
            events.append(("promote", name))
            return "ChunkIndexOld"

        def delete_collection(self, name: str) -> None:
            events.append(("delete", name))

        def close(self) -> None:
            pass

    def fake_index(source_id, *_args, **kwargs):  # noqa: ANN001
        assert kwargs["collection_name"].startswith("ChunkBuild")
        assert not kwargs["write_manifest_summary"]
        events.append(("index", source_id))
        return next(manifest for manifest in manifests if manifest.source_id == source_id)

    monkeypatch.setattr("scribebase.source_registry.list_manifests", lambda *_: manifests)
    monkeypatch.setattr("scribebase.indexing.index_source", fake_index)
    monkeypatch.setattr("scribebase.indexing.load_chunks", lambda *_: [_chunk("source", 0)])
    monkeypatch.setattr("scribebase.indexing.WeaviateStore", Store)
    monkeypatch.setattr(
        "scribebase.indexing.write_manifest",
        lambda manifest: events.append(("manifest", manifest.source_id)),
    )

    rebuild_index(None, True, config, Logger())

    event_names = [event[0] for event in events]
    assert event_names.index("count") < event_names.index("promote")
    assert event_names.count("manifest") == 2
    assert events[-1] == ("delete", "ChunkIndexOld")
