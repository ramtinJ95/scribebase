from __future__ import annotations

import fcntl
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from scribebase.chunking.chunker import chunk_markdown
from scribebase.config import AppConfig
from scribebase.durable_fs import (
    atomic_write,
    atomic_write_text,
    durable_copy,
    durable_unlink,
)
from scribebase.embeddings.llamacpp_client import LlamaCppEmbeddingClient
from scribebase.errors import DependencyUnavailableError, as_dependency_unavailable
from scribebase.extraction import read_page_metadata
from scribebase.models import Chunk, SourceManifest
from scribebase.paths import chapter_file_name
from scribebase.source_registry import find_source, read_jsonl, write_jsonl, write_manifest
from scribebase.vectorstores.weaviate_store import CollectionAliasMigrationError, WeaviateStore


def chunk_source(manifest: SourceManifest, config: AppConfig) -> list[Chunk]:
    root = Path(manifest.data_dir)
    markdown_path = root / "markdown" / "document.md"
    if manifest.chapter:
        chapter_path = root / "markdown" / "chapters" / chapter_file_name(manifest.chapter)
        if chapter_path.exists():
            markdown_path = chapter_path
    if not markdown_path.exists():
        raise FileNotFoundError(f"Missing extracted Markdown: {markdown_path}")
    pages = read_page_metadata(root)
    return chunk_markdown(markdown_path, manifest, pages, config.chunking)


def index_source(
    source_id: str,
    config: AppConfig,
    logger,
    no_create_collection: bool = False,
    allow_existing_model_mismatch: bool = False,
    operation_id: str | None = None,
) -> SourceManifest:
    with _index_lock(config.data_dir):
        _recover_index_transactions(config, logger)
        return _index_source(
            source_id,
            config,
            logger,
            no_create_collection=no_create_collection,
            allow_existing_model_mismatch=allow_existing_model_mismatch,
            operation_id=operation_id,
        )


def _index_source(
    source_id: str,
    config: AppConfig,
    logger,
    no_create_collection: bool = False,
    allow_existing_model_mismatch: bool = False,
    collection_name: str | None = None,
    write_manifest_summary: bool = True,
    chunks_output_path: Path | None = None,
    operation_id: str | None = None,
) -> SourceManifest:
    manifest = find_source(config.data_dir, source_id)
    chunks = chunk_source(manifest, config)
    if not chunks:
        raise RuntimeError(f"No chunks created for source: {source_id}")
    logger.info("Created %s chunks", len(chunks))

    chunks_path = chunks_output_path or Path(manifest.data_dir) / "metadata" / "chunks.jsonl"
    transaction_id = uuid4().hex
    snapshot_path = (
        Path(manifest.data_dir) / "metadata" / f"index_snapshot_{transaction_id}.jsonl"
    )
    staged_chunks = chunks_path.with_name(f"chunks.index-{transaction_id}.jsonl")
    staged_manifest = manifest_path = Path(manifest.data_dir) / "metadata" / "manifest.json"
    staged_manifest = manifest_path.with_name(f"manifest.index-{transaction_id}.json")
    journal_path = _incremental_journal_path(config.data_dir)
    embedder = LlamaCppEmbeddingClient(config.embedding)
    total_batches = (len(chunks) + config.embedding.batch_size - 1) // config.embedding.batch_size
    store = WeaviateStore(config.weaviate)
    dimension: int | None = None
    inserted = 0
    old_chunk_ids: set[str] = set()
    mutation_started = False
    preserve_snapshot = False
    transaction: dict | None = None
    try:
        if no_create_collection:
            client = store.connect()
            target = collection_name or config.weaviate.collection
            if not client.collections.exists(target):
                raise RuntimeError(f"Weaviate collection missing: {target}")
        else:
            store.ensure_collection()
        if collection_name is None:
            old_chunk_ids = _snapshot_source(store, source_id, snapshot_path)
            transaction = {
                "version": 1,
                "kind": "incremental",
                "state": "prepared",
                "source_id": source_id,
                "snapshot": str(snapshot_path),
                "staged_chunks": str(staged_chunks),
                "live_chunks": str(chunks_path),
                "staged_manifest": str(staged_manifest),
                "live_manifest": str(manifest_path),
            }
            _write_journal(journal_path, transaction)
            transaction["state"] = "mutating"
            _write_journal(journal_path, transaction)
        for batch_index, batch_vectors in enumerate(
            embedder.embed_batches([c.text for c in chunks]), start=1
        ):
            logger.info("Embedding batch %s/%s", batch_index, total_batches)
            start = (batch_index - 1) * config.embedding.batch_size
            batch_chunks = chunks[start : start + config.embedding.batch_size]
            if len(batch_chunks) != len(batch_vectors):
                raise RuntimeError("Embedding response count does not match chunk batch")
            batch_dimension = len(batch_vectors[0]) if batch_vectors else None
            if dimension is None:
                dimension = batch_dimension
                _validate_embedding_consistency(
                    config, source_id, dimension, allow_existing_model_mismatch
                )
            if batch_dimension != dimension or any(
                len(vector) != dimension for vector in batch_vectors
            ):
                raise RuntimeError("Embedding dimensions changed within source")
            for chunk in batch_chunks:
                chunk.embedding_model = config.embedding.model
                chunk.embedding_dimension = dimension
            if collection_name is None:
                mutation_started = True
            store.upsert_chunks(batch_chunks, batch_vectors, collection_name=collection_name)
            inserted += len(batch_chunks)
        if inserted != len(chunks):
            raise RuntimeError(f"Embedded {inserted} of {len(chunks)} chunks")
        if collection_name is None:
            store.delete_chunks(old_chunk_ids - {chunk.chunk_id for chunk in chunks})

            _set_embedding_summary(manifest, config, dimension, operation_id)
            _write_chunks_atomic(staged_chunks, chunks)
            atomic_write_text(staged_manifest, manifest.model_dump_json(indent=2))
            if transaction is None:
                raise RuntimeError("Missing incremental index transaction")
            transaction["state"] = "remote_committed"
            _write_journal(journal_path, transaction)
            _install_staged_files(
                [(staged_chunks, chunks_path), (staged_manifest, manifest_path)]
            )
            _finish_incremental_transaction(journal_path, transaction, logger)
    except Exception as exc:
        dependency_error = as_dependency_unavailable(exc)
        if (
            collection_name is None
            and mutation_started
            and transaction is not None
            and transaction["state"] == "mutating"
        ):
            try:
                _restore_source(store, source_id, snapshot_path, config.embedding.batch_size)
                _finish_incremental_transaction(journal_path, transaction, logger)
            except Exception as rollback_exc:
                preserve_snapshot = True
                message = (
                    f"Index update failed for {source_id}; restoring the previous vectors also failed: "
                    f"{rollback_exc}. Recovery snapshot preserved at {snapshot_path}"
                )
                if dependency_error or as_dependency_unavailable(rollback_exc):
                    raise DependencyUnavailableError(message) from exc
                raise RuntimeError(message) from exc
        elif collection_name is None and transaction is not None:
            if transaction["state"] == "prepared":
                _finish_incremental_transaction(journal_path, transaction, logger)
            else:
                preserve_snapshot = True
        if dependency_error is not None and dependency_error is not exc:
            raise dependency_error from exc
        raise
    finally:
        store.close()
        if collection_name is not None and not preserve_snapshot:
            snapshot_path.unlink(missing_ok=True)

    if collection_name is not None:
        _write_chunks_atomic(chunks_path, chunks)
        _set_embedding_summary(manifest, config, dimension, operation_id)
        if write_manifest_summary:
            write_manifest(manifest)
    logger.info(
        "Indexed %s chunks into Weaviate collection %s",
        len(chunks),
        collection_name or config.weaviate.collection,
    )
    return manifest


def load_chunks(data_dir: Path, source_id: str | None = None) -> list[Chunk]:
    roots = (
        [Path(find_source(data_dir, source_id).data_dir)]
        if source_id
        else sorted((data_dir / "sources").glob("*"))
    )
    chunks: list[Chunk] = []
    for root in roots:
        path = root / "metadata" / "chunks.jsonl"
        chunks.extend(Chunk.model_validate(row) for row in read_jsonl(path))
    return chunks


def rebuild_index(source_id: str | None, all_sources: bool, config: AppConfig, logger) -> None:
    with _index_lock(config.data_dir):
        _recover_index_transactions(config, logger)
        _rebuild_index(source_id, all_sources, config, logger)


def _rebuild_index(
    source_id: str | None,
    all_sources: bool,
    config: AppConfig,
    logger,
) -> None:
    from scribebase.source_registry import list_manifests

    if not all_sources and not source_id:
        raise ValueError("Provide --source-id or --all")
    ids = (
        [source_id]
        if source_id
        else [manifest.source_id for manifest in list_manifests(config.data_dir)]
    )
    if not all_sources:
        _index_source(source_id or "", config, logger)
        return

    staging = f"{config.weaviate.collection}Build{datetime.now(timezone.utc):%Y%m%d%H%M%S}{uuid4().hex[:6]}"
    store = WeaviateStore(config.weaviate)
    journal_path = _rebuild_journal_path(config.data_dir)
    transaction: dict | None = None
    try:
        logger.info("Building staged Weaviate collection %s", staging)
        store.create_collection(staging)
        expected = 0
        pending_files: list[tuple[Path, Path]] = []
        try:
            for sid in ids:
                if sid:
                    manifest = find_source(config.data_dir, sid)
                    live_chunks = Path(manifest.data_dir) / "metadata" / "chunks.jsonl"
                    staged_chunks = live_chunks.with_name(f"chunks.{staging}.jsonl")
                    pending_files.append((staged_chunks, live_chunks))
                    manifest = _index_source(
                        sid,
                        config,
                        logger,
                        no_create_collection=True,
                        allow_existing_model_mismatch=True,
                        collection_name=staging,
                        write_manifest_summary=False,
                        chunks_output_path=staged_chunks,
                    )
                    live_manifest = Path(manifest.data_dir) / "metadata" / "manifest.json"
                    staged_manifest = live_manifest.with_name(f"manifest.{staging}.json")
                    atomic_write_text(staged_manifest, manifest.model_dump_json(indent=2))
                    pending_files.append((staged_manifest, live_manifest))
                    expected += _jsonl_row_count(staged_chunks)
            actual = store.object_count(staging)
            if actual != expected:
                raise RuntimeError(
                    f"Staged index verification failed: expected {expected} chunks, found {actual}"
                )
            transaction = {
                "version": 1,
                "kind": "rebuild",
                "state": "prepared",
                "staging_collection": staging,
                "expected_count": expected,
                "previous_collection": None,
                "files": [
                    {"staged": str(staged), "live": str(live)}
                    for staged, live in pending_files
                ],
            }
            _write_journal(journal_path, transaction)
            previous = store.promote_collection(staging)
            transaction["state"] = "promoted"
            transaction["previous_collection"] = previous
            _write_journal(journal_path, transaction)
            logger.info("Promoted %s as alias %s", staging, config.weaviate.collection)
            _install_staged_files(pending_files)
            if previous and previous != staging:
                try:
                    store.delete_collection(previous)
                except Exception as exc:
                    logger.warning("Could not remove previous collection %s: %s", previous, exc)
            _finish_rebuild_transaction(journal_path, transaction, logger)
        except Exception as exc:
            preserve_staging = isinstance(exc, CollectionAliasMigrationError)
            if transaction is None and not preserve_staging:
                try:
                    store.delete_collection(staging)
                except Exception as exc:
                    logger.warning(
                        "Could not remove failed staging collection %s: %s", staging, exc
                    )
            if transaction is None:
                for staged_path, _ in pending_files:
                    staged_path.unlink(missing_ok=True)
            else:
                logger.error(
                    "Index transaction was preserved for restart recovery: %s",
                    ", ".join(str(staged) for staged, _ in pending_files),
                )
            raise
    finally:
        store.close()


def _validate_embedding_consistency(
    config: AppConfig,
    source_id: str,
    dimension: int | None,
    allow_existing_model_mismatch: bool,
) -> None:
    from scribebase.source_registry import list_manifests

    if dimension is None:
        return
    for manifest in list_manifests(config.data_dir):
        if manifest.source_id == source_id:
            continue
        summary = manifest.embedding_summary
        if not summary.indexed_in_weaviate:
            continue
        if summary.weaviate_collection != config.weaviate.collection:
            continue
        if summary.embedding_model != config.embedding.model and not allow_existing_model_mismatch:
            raise RuntimeError(
                "Embedding model mismatch for existing index: "
                f"configured model is {config.embedding.model!r}, but {manifest.source_id} stores "
                f"{summary.embedding_model!r}. Run `scribebase rebuild-index --all` after changing models."
            )
        if (
            summary.embedding_model == config.embedding.model
            and summary.embedding_dimension != dimension
        ):
            raise RuntimeError(
                "Embedding dimension mismatch for existing index: "
                f"configured model produced {dimension}, but {manifest.source_id} stores "
                f"{summary.embedding_dimension}. Rebuild the index."
            )


def _write_chunks_atomic(path: Path, chunks: list[Chunk]) -> None:
    write_jsonl(path, [chunk.model_dump(mode="json") for chunk in chunks])


def _snapshot_source(store: WeaviateStore, source_id: str, path: Path) -> set[str]:
    chunk_ids: set[str] = set()

    def write_snapshot(output) -> None:  # noqa: ANN001
        for chunk, vector in store.iter_source_chunks(source_id, include_vectors=True):
            chunk_ids.add(chunk.chunk_id)
            output.write(
                (
                json.dumps(
                    {"chunk": chunk.model_dump(mode="json"), "vector": vector},
                    ensure_ascii=False,
                )
                + "\n"
                ).encode()
            )

    atomic_write(path, write_snapshot)
    return chunk_ids


def _restore_source(
    store: WeaviateStore,
    source_id: str,
    snapshot_path: Path,
    batch_size: int,
) -> None:
    _validate_snapshot(snapshot_path, source_id)
    store.delete_source(source_id)
    chunks: list[Chunk] = []
    vectors: list[list[float]] = []
    for row in _iter_jsonl(snapshot_path):
        chunks.append(Chunk.model_validate(row["chunk"]))
        vectors.append(row["vector"])
        if len(chunks) == batch_size:
            store.upsert_chunks(chunks, vectors)
            chunks, vectors = [], []
    if chunks:
        store.upsert_chunks(chunks, vectors)


def _validate_snapshot(path: Path, source_id: str) -> None:
    chunk_ids: set[str] = set()
    try:
        for row in _iter_jsonl(path):
            chunk = Chunk.model_validate(row["chunk"])
            if chunk.source_id != source_id:
                raise ValueError(
                    f"snapshot chunk {chunk.chunk_id!r} belongs to {chunk.source_id!r}"
                )
            if chunk.chunk_id in chunk_ids:
                raise ValueError(f"duplicate snapshot chunk id: {chunk.chunk_id}")
            chunk_ids.add(chunk.chunk_id)
            vector = row["vector"]
            if not isinstance(vector, list) or not vector:
                raise ValueError(f"invalid vector for snapshot chunk: {chunk.chunk_id}")
            if any(not isinstance(value, (int, float)) for value in vector):
                raise ValueError(f"non-numeric vector for snapshot chunk: {chunk.chunk_id}")
    except Exception as exc:
        raise RuntimeError(f"Invalid index recovery snapshot: {path}: {exc}") from exc


def _validate_staged_index_files(transaction: dict) -> dict[str, dict]:
    source_id = transaction["source_id"]
    chunks_path = Path(transaction["staged_chunks"])
    manifest_path = Path(transaction["staged_manifest"])
    chunks: dict[str, dict] = {}
    try:
        for row in _iter_jsonl(chunks_path):
            chunk = Chunk.model_validate(row)
            if chunk.source_id != source_id:
                raise ValueError(
                    f"staged chunk {chunk.chunk_id!r} belongs to {chunk.source_id!r}"
                )
            if chunk.chunk_id in chunks:
                raise ValueError(f"duplicate staged chunk id: {chunk.chunk_id}")
            chunks[chunk.chunk_id] = _recovery_chunk_content(chunk)
        if not chunks:
            raise ValueError("staged chunk file is empty")
        manifest = SourceManifest.model_validate_json(manifest_path.read_text())
        if manifest.source_id != source_id:
            raise ValueError(
                f"staged manifest belongs to {manifest.source_id!r}, expected {source_id!r}"
            )
    except Exception as exc:
        raise RuntimeError(f"Invalid staged index recovery files for {source_id}: {exc}") from exc
    return chunks


def _recovery_chunk_content(chunk: Chunk) -> dict:
    # Weaviate assigns created_at while writing and does not persist
    # chunker_version in the current schema. Every other Chunk field is
    # round-tripped and must match before local metadata can be published.
    return chunk.model_dump(mode="json", exclude={"created_at", "chunker_version"})


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open() as rows:
        for line in rows:
            if line.strip():
                yield json.loads(line)


def _jsonl_row_count(path: Path) -> int:
    with path.open() as rows:
        return sum(1 for line in rows if line.strip())


def _install_staged_files(files: list[tuple[Path, Path]]) -> None:
    for staged, live in files:
        durable_copy(staged, live)


def recover_index_transactions(config: AppConfig, logger) -> None:  # noqa: ANN001
    with _index_lock(config.data_dir):
        _recover_index_transactions(config, logger)


def index_recovery_pending(data_dir: Path) -> bool:
    return _incremental_journal_path(data_dir).exists() or _rebuild_journal_path(data_dir).exists()


def _recover_index_transactions(config: AppConfig, logger) -> None:  # noqa: ANN001
    incremental = _incremental_journal_path(config.data_dir)
    if incremental.exists():
        transaction = _read_journal(incremental, "incremental")
        state = transaction["state"]
        if state == "prepared":
            _finish_incremental_transaction(incremental, transaction, logger)
        elif state == "mutating":
            store = WeaviateStore(config.weaviate)
            try:
                _restore_source(
                    store,
                    transaction["source_id"],
                    Path(transaction["snapshot"]),
                    config.embedding.batch_size,
                )
            finally:
                store.close()
            _finish_incremental_transaction(incremental, transaction, logger)
            logger.warning("Restored interrupted index update for %s", transaction["source_id"])
        elif state == "remote_committed":
            expected_chunks = _validate_staged_index_files(transaction)
            store = WeaviateStore(config.weaviate)
            try:
                remote_chunks = {
                    chunk.chunk_id: _recovery_chunk_content(chunk)
                    for chunk, _ in store.iter_source_chunks(transaction["source_id"])
                }
                if remote_chunks != expected_chunks:
                    _restore_source(
                        store,
                        transaction["source_id"],
                        Path(transaction["snapshot"]),
                        config.embedding.batch_size,
                    )
                    logger.warning(
                        "Restored interrupted index update for %s because the remote "
                        "generation did not match the staged chunks",
                        transaction["source_id"],
                    )
                else:
                    _install_staged_files(
                        [
                            (
                                Path(transaction["staged_chunks"]),
                                Path(transaction["live_chunks"]),
                            ),
                            (
                                Path(transaction["staged_manifest"]),
                                Path(transaction["live_manifest"]),
                            ),
                        ]
                    )
                    logger.warning(
                        "Finished interrupted index update for %s", transaction["source_id"]
                    )
            finally:
                store.close()
            _finish_incremental_transaction(incremental, transaction, logger)
        else:
            raise RuntimeError(f"Unknown incremental index transaction state: {state!r}")

    rebuild = _rebuild_journal_path(config.data_dir)
    if rebuild.exists():
        transaction = _read_journal(rebuild, "rebuild")
        store = WeaviateStore(config.weaviate)
        try:
            staging = transaction["staging_collection"]
            expected_count = transaction["expected_count"]
            actual_count = store.object_count(staging)
            if actual_count != expected_count:
                raise RuntimeError(
                    f"Interrupted rebuild collection is incomplete: expected {expected_count} "
                    f"chunks in {staging}, found {actual_count}"
                )
            alias_target = store.alias_target()
            if alias_target != staging:
                if transaction["state"] != "prepared":
                    raise RuntimeError(
                        f"Rebuild journal expects alias {config.weaviate.collection!r} to target "
                        f"{staging!r}, found {alias_target!r}"
                    )
                previous = store.promote_collection(staging)
                transaction["state"] = "promoted"
                transaction["previous_collection"] = previous
                _write_journal(rebuild, transaction)
            files = [
                (Path(entry["staged"]), Path(entry["live"]))
                for entry in transaction["files"]
            ]
            _install_staged_files(files)
            previous = transaction.get("previous_collection")
            if previous and previous != staging:
                try:
                    store.delete_collection(previous)
                except Exception as exc:
                    logger.warning("Could not remove previous collection %s: %s", previous, exc)
            _finish_rebuild_transaction(rebuild, transaction, logger)
            logger.warning("Finished interrupted full index rebuild")
        finally:
            store.close()


def _incremental_journal_path(data_dir: Path) -> Path:
    return data_dir / ".index-transaction.json"


def _rebuild_journal_path(data_dir: Path) -> Path:
    return data_dir / ".index-rebuild-transaction.json"


def _write_journal(path: Path, transaction: dict) -> None:
    atomic_write_text(path, json.dumps(transaction, indent=2))


def _read_journal(path: Path, kind: str) -> dict:
    try:
        transaction = json.loads(path.read_text())
    except Exception as exc:
        raise RuntimeError(f"Unreadable index recovery journal: {path}") from exc
    if transaction.get("version") != 1 or transaction.get("kind") != kind:
        raise RuntimeError(f"Unsupported index recovery journal: {path}")
    return transaction


def _finish_incremental_transaction(path: Path, transaction: dict, logger) -> None:  # noqa: ANN001
    durable_unlink(path, missing_ok=True)
    _remove_committed_artifacts(
        [Path(transaction[key]) for key in ("snapshot", "staged_chunks", "staged_manifest")],
        logger,
    )


def _finish_rebuild_transaction(path: Path, transaction: dict, logger) -> None:  # noqa: ANN001
    durable_unlink(path, missing_ok=True)
    _remove_committed_artifacts(
        [Path(entry["staged"]) for entry in transaction["files"]], logger
    )


def _remove_committed_artifacts(paths: list[Path], logger) -> None:  # noqa: ANN001
    for path in paths:
        try:
            durable_unlink(path, missing_ok=True)
        except OSError as exc:
            logger.warning("Committed index transaction retained cleanup artifact %s: %s", path, exc)


@contextmanager
def _index_lock(data_dir: Path):  # noqa: ANN202
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "index.lock"
    with path.open("a+") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another index operation is already running: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _set_embedding_summary(
    manifest: SourceManifest,
    config: AppConfig,
    dimension: int | None,
    operation_id: str | None,
) -> None:
    manifest.embedding_summary.embedding_model = config.embedding.model
    manifest.embedding_summary.embedding_dimension = dimension
    manifest.embedding_summary.embedding_base_url = config.embedding.base_url
    manifest.embedding_summary.indexed_in_weaviate = True
    manifest.embedding_summary.weaviate_collection = config.weaviate.collection
    manifest.embedding_summary.index_operation_id = operation_id
    manifest.updated_at = datetime.now(timezone.utc)
