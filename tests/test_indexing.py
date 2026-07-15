import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scribebase.config import default_config
from scribebase.indexing import (
    _index_lock,
    _install_staged_files,
    index_source,
    rebuild_index,
    recover_index_transactions,
)
from scribebase.models import Chunk, SourceManifest
from scribebase.vectorstores.weaviate_store import CollectionAliasMigrationError


class Logger:
    def info(self, *args) -> None:  # noqa: ANN002
        pass

    def warning(self, *args) -> None:  # noqa: ANN002
        pass

    def error(self, *args) -> None:  # noqa: ANN002
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

        def iter_source_chunks(self, *_args, **_kwargs):
            for chunk in old:
                yield chunk, [1.0, 0.0]

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
    result = index_source(manifest.source_id, config, Logger())

    assert [len(batch[0]) for batch in Store.batches] == [2, 1]
    assert Store.deleted == {"stale"}
    assert result.embedding_summary.embedding_dimension == 2
    saved_manifest = SourceManifest.model_validate_json(
        (tmp_path / "sources" / manifest.source_id / "metadata" / "manifest.json").read_text()
    )
    assert saved_manifest.embedding_summary.embedding_dimension == 2
    assert not (tmp_path / ".index-transaction.json").exists()
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
        source_deleted = False
        restored = []

        def __init__(self, _config) -> None:  # noqa: ANN001
            pass

        def ensure_collection(self) -> None:
            pass

        def iter_source_chunks(self, *_args, **_kwargs):
            yield _chunk(manifest.source_id, 9, "stale"), [0.5, 0.5]

        def upsert_chunks(self, *_args, **_kwargs) -> None:
            type(self).calls += 1
            if type(self).calls == 2:
                raise RuntimeError("batch failed")
            if type(self).source_deleted:
                type(self).restored.append(_args[0][0].chunk_id)

        def delete_chunks(self, _chunk_ids) -> None:  # noqa: ANN001
            pass

        def delete_source(self, _source_id) -> None:  # noqa: ANN001
            type(self).source_deleted = True

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
    assert Store.source_deleted
    assert Store.restored == ["stale"]
    assert writes == []


def test_failed_rollback_preserves_vector_snapshot(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    config = default_config()
    config.data_dir = tmp_path
    manifest = _manifest(tmp_path)

    class Embedder:
        def __init__(self, _config) -> None:  # noqa: ANN001
            pass

        def embed_batches(self, _texts):  # noqa: ANN001
            yield [[1.0, 0.0]]

    class Store:
        def __init__(self, _config) -> None:  # noqa: ANN001
            pass

        def ensure_collection(self) -> None:
            pass

        def iter_source_chunks(self, *_args, **_kwargs):
            yield _chunk(manifest.source_id, 9, "old"), [0.5, 0.5]

        def upsert_chunks(self, *_args, **_kwargs) -> None:
            raise RuntimeError("weaviate unavailable")

        def delete_source(self, _source_id) -> None:  # noqa: ANN001
            pass

        def close(self) -> None:
            pass

    monkeypatch.setattr("scribebase.indexing.find_source", lambda *_: manifest)
    monkeypatch.setattr(
        "scribebase.indexing.chunk_source", lambda *_: [_chunk(manifest.source_id, 0)]
    )
    monkeypatch.setattr("scribebase.indexing.LlamaCppEmbeddingClient", Embedder)
    monkeypatch.setattr("scribebase.indexing.WeaviateStore", Store)

    with pytest.raises(RuntimeError, match="Recovery snapshot preserved"):
        index_source(manifest.source_id, config, Logger())

    snapshots = list(
        (tmp_path / "sources" / manifest.source_id / "metadata").glob("index_snapshot_*.jsonl")
    )
    assert len(snapshots) == 1
    assert '"chunk_id": "old"' in snapshots[0].read_text()


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
        kwargs["chunks_output_path"].write_text('{"chunk": true}\n')
        events.append(("index", source_id))
        return next(manifest for manifest in manifests if manifest.source_id == source_id)

    monkeypatch.setattr("scribebase.source_registry.list_manifests", lambda *_: manifests)
    monkeypatch.setattr(
        "scribebase.indexing.find_source",
        lambda _data_dir, source_id: next(
            manifest for manifest in manifests if manifest.source_id == source_id
        ),
    )
    monkeypatch.setattr("scribebase.indexing._index_source", fake_index)
    monkeypatch.setattr("scribebase.indexing.WeaviateStore", Store)
    monkeypatch.setattr(
        "scribebase.indexing.write_manifest",
        lambda manifest: events.append(("manifest", manifest.source_id)),
    )

    rebuild_index(None, True, config, Logger())

    event_names = [event[0] for event in events]
    assert event_names.index("count") < event_names.index("promote")
    assert events[-1] == ("delete", "ChunkIndexOld")
    for manifest in manifests:
        manifest_path = tmp_path / "sources" / manifest.source_id / "metadata" / "manifest.json"
        assert (
            SourceManifest.model_validate_json(manifest_path.read_text()).source_id
            == manifest.source_id
        )
    assert not list(tmp_path.glob("sources/*/metadata/*.ChunkBuild*.*"))


def test_failed_full_rebuild_preserves_live_chunk_files(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    config = default_config()
    config.data_dir = tmp_path
    manifests = [_manifest(tmp_path, "source-1"), _manifest(tmp_path, "source-2")]
    live_paths = []
    events = []
    for manifest in manifests:
        path = tmp_path / "sources" / manifest.source_id / "metadata" / "chunks.jsonl"
        path.write_text(f"old {manifest.source_id}\n")
        live_paths.append(path)

    class Store:
        def __init__(self, _config) -> None:  # noqa: ANN001
            pass

        def create_collection(self, name: str) -> None:
            events.append(("create", name))

        def delete_collection(self, name: str) -> None:
            events.append(("delete", name))

        def close(self) -> None:
            pass

    def fake_index(source_id, *_args, **kwargs):  # noqa: ANN001
        kwargs["chunks_output_path"].write_text('{"staged": true}\n')
        if source_id == "source-2":
            raise RuntimeError("embedding failed")
        return manifests[0]

    monkeypatch.setattr("scribebase.source_registry.list_manifests", lambda *_: manifests)
    monkeypatch.setattr(
        "scribebase.indexing.find_source",
        lambda _data_dir, source_id: next(
            manifest for manifest in manifests if manifest.source_id == source_id
        ),
    )
    monkeypatch.setattr("scribebase.indexing._index_source", fake_index)
    monkeypatch.setattr("scribebase.indexing.WeaviateStore", Store)

    with pytest.raises(RuntimeError, match="embedding failed"):
        rebuild_index(None, True, config, Logger())

    assert [path.read_text() for path in live_paths] == ["old source-1\n", "old source-2\n"]
    assert not list(tmp_path.glob("sources/*/metadata/chunks.ChunkBuild*.jsonl"))
    assert [event[0] for event in events] == ["create", "delete"]


def test_alias_migration_failure_preserves_verified_staging_collection(
    tmp_path, monkeypatch
) -> None:  # noqa: ANN001
    config = default_config()
    config.data_dir = tmp_path
    manifest = _manifest(tmp_path)
    deleted = []

    class Store:
        def __init__(self, _config) -> None:  # noqa: ANN001
            pass

        def create_collection(self, _name: str) -> None:
            pass

        def object_count(self, _name: str) -> int:
            return 1

        def promote_collection(self, name: str):  # noqa: ANN201
            raise CollectionAliasMigrationError("Chunk", name)

        def delete_collection(self, name: str) -> None:
            deleted.append(name)

        def close(self) -> None:
            pass

    def fake_index(_source_id, *_args, **kwargs):  # noqa: ANN001
        kwargs["chunks_output_path"].write_text('{"staged": true}\n')
        return manifest

    monkeypatch.setattr("scribebase.source_registry.list_manifests", lambda *_: [manifest])
    monkeypatch.setattr("scribebase.indexing.find_source", lambda *_: manifest)
    monkeypatch.setattr("scribebase.indexing._index_source", fake_index)
    monkeypatch.setattr("scribebase.indexing.WeaviateStore", Store)

    with pytest.raises(CollectionAliasMigrationError, match="Verified rebuilt data remains"):
        rebuild_index(None, True, config, Logger())

    assert deleted == []


def test_post_promotion_file_failure_preserves_all_staged_artifacts(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    config = default_config()
    config.data_dir = tmp_path
    manifest = _manifest(tmp_path)
    live_chunks = tmp_path / "sources" / manifest.source_id / "metadata" / "chunks.jsonl"
    live_chunks.write_text("old chunks\n")

    class Store:
        def __init__(self, _config) -> None:  # noqa: ANN001
            pass

        def create_collection(self, _name: str) -> None:
            pass

        def object_count(self, _name: str) -> int:
            return 1

        def promote_collection(self, _name: str):  # noqa: ANN201
            return "ChunkOld"

        def delete_collection(self, _name: str) -> None:
            pass

        def close(self) -> None:
            pass

    def fake_index(_source_id, *_args, **kwargs):  # noqa: ANN001
        kwargs["chunks_output_path"].write_text('{"staged": true}\n')
        return manifest

    monkeypatch.setattr("scribebase.source_registry.list_manifests", lambda *_: [manifest])
    monkeypatch.setattr("scribebase.indexing.find_source", lambda *_: manifest)
    monkeypatch.setattr("scribebase.indexing._index_source", fake_index)
    monkeypatch.setattr("scribebase.indexing.WeaviateStore", Store)
    monkeypatch.setattr(
        "scribebase.indexing._install_staged_files",
        lambda _files: (_ for _ in ()).throw(OSError("disk failure")),
    )

    with pytest.raises(OSError, match="disk failure"):
        rebuild_index(None, True, config, Logger())

    assert live_chunks.read_text() == "old chunks\n"
    staged = list(tmp_path.glob("sources/*/metadata/*.ChunkBuild*.*"))
    assert {path.name.split(".", 1)[0] for path in staged} == {"chunks", "manifest"}


def test_staged_file_install_preserves_forward_recovery_files(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    staged_chunks = tmp_path / "chunks.staged.jsonl"
    staged_manifest = tmp_path / "manifest.staged.json"
    live_chunks = tmp_path / "chunks.jsonl"
    live_manifest = tmp_path / "manifest.json"
    staged_chunks.write_text("new chunks\n")
    staged_manifest.write_text("new manifest\n")
    live_chunks.write_text("old chunks\n")
    live_manifest.write_text("old manifest\n")
    from scribebase.durable_fs import durable_copy as real_copy

    def fail_second_staged_copy(source, destination):  # noqa: ANN001, ANN202
        if Path(source) == staged_manifest:
            raise OSError("manifest install failed")
        return real_copy(source, destination)

    monkeypatch.setattr("scribebase.indexing.durable_copy", fail_second_staged_copy)

    with pytest.raises(OSError, match="manifest install failed"):
        _install_staged_files([(staged_chunks, live_chunks), (staged_manifest, live_manifest)])

    assert live_chunks.read_text() == "new chunks\n"
    assert live_manifest.read_text() == "old manifest\n"
    assert staged_chunks.read_text() == "new chunks\n"
    assert staged_manifest.read_text() == "new manifest\n"


def test_recovers_interrupted_incremental_vector_mutation(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    config = default_config()
    config.data_dir = tmp_path
    snapshot = tmp_path / "snapshot.jsonl"
    old_chunk = _chunk("source-1", 9, "old")
    snapshot.write_text(
        json.dumps({"chunk": old_chunk.model_dump(mode="json"), "vector": [0.5, 0.5]})
        + "\n"
    )
    journal = tmp_path / ".index-transaction.json"
    journal.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "incremental",
                "state": "mutating",
                "source_id": "source-1",
                "snapshot": str(snapshot),
                "staged_chunks": str(tmp_path / "staged-chunks"),
                "live_chunks": str(tmp_path / "live-chunks"),
                "staged_manifest": str(tmp_path / "staged-manifest"),
                "live_manifest": str(tmp_path / "live-manifest"),
            }
        )
    )

    class Store:
        deleted = []
        restored = []

        def __init__(self, _config) -> None:  # noqa: ANN001
            pass

        def delete_source(self, source_id: str) -> None:
            self.deleted.append(source_id)

        def upsert_chunks(self, chunks, vectors) -> None:  # noqa: ANN001
            self.restored.extend((chunk.chunk_id, vector) for chunk, vector in zip(chunks, vectors))

        def close(self) -> None:
            pass

    monkeypatch.setattr("scribebase.indexing.WeaviateStore", Store)

    recover_index_transactions(config, Logger())

    assert Store.deleted == ["source-1"]
    assert Store.restored == [("old", [0.5, 0.5])]
    assert not journal.exists()
    assert not snapshot.exists()


def test_recovery_validates_entire_snapshot_before_deleting_vectors(
    tmp_path, monkeypatch
) -> None:  # noqa: ANN001
    config = default_config()
    config.data_dir = tmp_path
    snapshot = tmp_path / "snapshot.jsonl"
    old_chunk = _chunk("source-1", 9, "old")
    snapshot.write_text(
        json.dumps({"chunk": old_chunk.model_dump(mode="json"), "vector": [0.5, 0.5]})
        + "\n{truncated"
    )
    journal = tmp_path / ".index-transaction.json"
    journal.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "incremental",
                "state": "mutating",
                "source_id": "source-1",
                "snapshot": str(snapshot),
                "staged_chunks": str(tmp_path / "staged-chunks"),
                "live_chunks": str(tmp_path / "live-chunks"),
                "staged_manifest": str(tmp_path / "staged-manifest"),
                "live_manifest": str(tmp_path / "live-manifest"),
            }
        )
    )

    class Store:
        deleted = []

        def __init__(self, _config) -> None:  # noqa: ANN001
            pass

        def delete_source(self, source_id: str) -> None:
            self.deleted.append(source_id)

        def close(self) -> None:
            pass

    monkeypatch.setattr("scribebase.indexing.WeaviateStore", Store)

    with pytest.raises(RuntimeError, match="Invalid index recovery snapshot"):
        recover_index_transactions(config, Logger())

    assert Store.deleted == []
    assert journal.exists()
    assert snapshot.exists()


def test_finishes_incremental_local_commit_after_remote_commit(tmp_path) -> None:
    config = default_config()
    config.data_dir = tmp_path
    staged_chunks = tmp_path / "staged-chunks.jsonl"
    staged_manifest = tmp_path / "staged-manifest.json"
    live_chunks = tmp_path / "live-chunks.jsonl"
    live_manifest = tmp_path / "live-manifest.json"
    snapshot = tmp_path / "snapshot.jsonl"
    chunk_content = json.dumps(_chunk("source-1", 0).model_dump(mode="json")) + "\n"
    manifest_content = _manifest(tmp_path).model_dump_json(indent=2)
    staged_chunks.write_text(chunk_content)
    staged_manifest.write_text(manifest_content)
    snapshot.write_text("")
    journal = tmp_path / ".index-transaction.json"
    journal.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "incremental",
                "state": "remote_committed",
                "source_id": "source-1",
                "snapshot": str(snapshot),
                "staged_chunks": str(staged_chunks),
                "live_chunks": str(live_chunks),
                "staged_manifest": str(staged_manifest),
                "live_manifest": str(live_manifest),
            }
        )
    )

    recover_index_transactions(config, Logger())

    assert live_chunks.read_text() == chunk_content
    assert live_manifest.read_text() == manifest_content
    assert not journal.exists()
    assert not staged_chunks.exists()
    assert not staged_manifest.exists()


def test_finishes_promoted_full_rebuild_after_restart(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    config = default_config()
    config.data_dir = tmp_path
    staged = tmp_path / "staged.json"
    live = tmp_path / "live.json"
    staged.write_text("new\n")
    live.write_text("old\n")
    journal = tmp_path / ".index-rebuild-transaction.json"
    journal.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "rebuild",
                "state": "promoted",
                "staging_collection": "ChunkBuild1",
                "expected_count": 1,
                "previous_collection": "ChunkOld",
                "files": [{"staged": str(staged), "live": str(live)}],
            }
        )
    )

    class Store:
        deleted = []

        def __init__(self, _config) -> None:  # noqa: ANN001
            pass

        def alias_target(self) -> str:
            return "ChunkBuild1"

        def object_count(self, _name: str) -> int:
            return 1

        def delete_collection(self, name: str) -> None:
            self.deleted.append(name)

        def close(self) -> None:
            pass

    monkeypatch.setattr("scribebase.indexing.WeaviateStore", Store)

    recover_index_transactions(config, Logger())

    assert live.read_text() == "new\n"
    assert Store.deleted == ["ChunkOld"]
    assert not journal.exists()
    assert not staged.exists()


def test_finishes_full_rebuild_if_power_fails_during_alias_promotion(
    tmp_path, monkeypatch
) -> None:  # noqa: ANN001
    config = default_config()
    config.data_dir = tmp_path
    staged = tmp_path / "staged.json"
    live = tmp_path / "live.json"
    staged.write_text("new\n")
    live.write_text("old\n")
    journal = tmp_path / ".index-rebuild-transaction.json"
    journal.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "rebuild",
                "state": "prepared",
                "staging_collection": "ChunkBuild1",
                "expected_count": 1,
                "previous_collection": None,
                "files": [{"staged": str(staged), "live": str(live)}],
            }
        )
    )

    class Store:
        promoted = []

        def __init__(self, _config) -> None:  # noqa: ANN001
            pass

        def alias_target(self) -> str:
            return "ChunkOld"

        def object_count(self, _name: str) -> int:
            return 1

        def promote_collection(self, name: str) -> str:
            self.promoted.append(name)
            return "ChunkOld"

        def delete_collection(self, _name: str) -> None:
            pass

        def close(self) -> None:
            pass

    monkeypatch.setattr("scribebase.indexing.WeaviateStore", Store)

    recover_index_transactions(config, Logger())

    assert Store.promoted == ["ChunkBuild1"]
    assert live.read_text() == "new\n"
    assert not journal.exists()


def test_does_not_promote_incomplete_rebuild_after_restart(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    config = default_config()
    config.data_dir = tmp_path
    staged = tmp_path / "staged.json"
    live = tmp_path / "live.json"
    staged.write_text("new\n")
    live.write_text("old\n")
    journal = tmp_path / ".index-rebuild-transaction.json"
    journal.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "rebuild",
                "state": "prepared",
                "staging_collection": "ChunkBuild1",
                "expected_count": 2,
                "previous_collection": None,
                "files": [{"staged": str(staged), "live": str(live)}],
            }
        )
    )

    class Store:
        promoted = []

        def __init__(self, _config) -> None:  # noqa: ANN001
            pass

        def object_count(self, _name: str) -> int:
            return 1

        def promote_collection(self, name: str) -> None:
            self.promoted.append(name)

        def close(self) -> None:
            pass

    monkeypatch.setattr("scribebase.indexing.WeaviateStore", Store)

    with pytest.raises(RuntimeError, match="Interrupted rebuild collection is incomplete"):
        recover_index_transactions(config, Logger())

    assert Store.promoted == []
    assert live.read_text() == "old\n"
    assert journal.exists()


def test_index_lock_rejects_concurrent_mutation(tmp_path) -> None:  # noqa: ANN001
    with _index_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another index operation"):
            with _index_lock(tmp_path):
                pass
