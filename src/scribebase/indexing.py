from __future__ import annotations

import fcntl
import json
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from scribebase.chunking.chunker import chunk_markdown
from scribebase.config import AppConfig
from scribebase.embeddings.llamacpp_client import LlamaCppEmbeddingClient
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
    snapshot_path = Path(manifest.data_dir) / "metadata" / f"index_snapshot_{uuid4().hex}.jsonl"
    embedder = LlamaCppEmbeddingClient(config.embedding)
    total_batches = (len(chunks) + config.embedding.batch_size - 1) // config.embedding.batch_size
    store = WeaviateStore(config.weaviate)
    dimension: int | None = None
    inserted = 0
    old_chunk_ids: set[str] = set()
    mutation_started = False
    preserve_snapshot = False
    try:
        if no_create_collection:
            client = store.connect()
            target = collection_name or config.weaviate.collection
            if not client.collections.exists(target):
                raise RuntimeError(f"Weaviate collection missing: {target}")
        else:
            store.ensure_collection()
            old_chunk_ids = _snapshot_source(store, source_id, snapshot_path)
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
    except Exception as exc:
        if collection_name is None and mutation_started:
            try:
                _restore_source(store, source_id, snapshot_path, config.embedding.batch_size)
            except Exception as rollback_exc:
                preserve_snapshot = True
                raise RuntimeError(
                    f"Index update failed for {source_id}; restoring the previous vectors also failed: "
                    f"{rollback_exc}. Recovery snapshot preserved at {snapshot_path}"
                ) from exc
        raise
    finally:
        store.close()
        if not preserve_snapshot:
            snapshot_path.unlink(missing_ok=True)

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
    try:
        logger.info("Building staged Weaviate collection %s", staging)
        store.create_collection(staging)
        expected = 0
        pending_files: list[tuple[Path, Path]] = []
        promoted = False
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
                    staged_manifest.write_text(manifest.model_dump_json(indent=2))
                    pending_files.append((staged_manifest, live_manifest))
                    expected += _jsonl_row_count(staged_chunks)
            actual = store.object_count(staging)
            if actual != expected:
                raise RuntimeError(
                    f"Staged index verification failed: expected {expected} chunks, found {actual}"
                )
            previous = store.promote_collection(staging)
            promoted = True
            logger.info("Promoted %s as alias %s", staging, config.weaviate.collection)
            _install_staged_files(pending_files)
            if previous and previous != staging:
                try:
                    store.delete_collection(previous)
                except Exception as exc:
                    logger.warning("Could not remove previous collection %s: %s", previous, exc)
        except Exception as exc:
            preserve_staging = isinstance(exc, CollectionAliasMigrationError)
            if not promoted and not preserve_staging:
                try:
                    store.delete_collection(staging)
                except Exception as exc:
                    logger.warning(
                        "Could not remove failed staging collection %s: %s", staging, exc
                    )
            if not promoted:
                for staged_path, _ in pending_files:
                    staged_path.unlink(missing_ok=True)
            else:
                logger.error(
                    "Index was promoted but local metadata finalization failed; staged files "
                    "were preserved for recovery: %s",
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
    temporary = path.with_suffix(f"{path.suffix}.{uuid4().hex}.tmp")
    try:
        write_jsonl(temporary, [chunk.model_dump(mode="json") for chunk in chunks])
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _snapshot_source(store: WeaviateStore, source_id: str, path: Path) -> set[str]:
    chunk_ids: set[str] = set()
    with path.open("w") as output:
        for chunk, vector in store.iter_source_chunks(source_id, include_vectors=True):
            chunk_ids.add(chunk.chunk_id)
            output.write(
                json.dumps(
                    {"chunk": chunk.model_dump(mode="json"), "vector": vector},
                    ensure_ascii=False,
                )
                + "\n"
            )
    return chunk_ids


def _restore_source(
    store: WeaviateStore,
    source_id: str,
    snapshot_path: Path,
    batch_size: int,
) -> None:
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


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open() as rows:
        for line in rows:
            if line.strip():
                yield json.loads(line)


def _jsonl_row_count(path: Path) -> int:
    with path.open() as rows:
        return sum(1 for line in rows if line.strip())


def _install_staged_files(files: list[tuple[Path, Path]]) -> None:
    backups: list[tuple[Path | None, Path]] = []
    preserve_backups = False
    try:
        for _, live in files:
            if live.exists():
                backup = live.with_suffix(f"{live.suffix}.{uuid4().hex}.backup")
                shutil.copy2(live, backup)
            else:
                backup = None
            backups.append((backup, live))
        for staged, live in files:
            temporary = live.with_suffix(f"{live.suffix}.{uuid4().hex}.tmp")
            try:
                shutil.copy2(staged, temporary)
                temporary.replace(live)
            finally:
                temporary.unlink(missing_ok=True)
    except Exception as install_exc:
        restore_errors = []
        for backup, live in backups:
            try:
                if backup is None:
                    live.unlink(missing_ok=True)
                else:
                    temporary = live.with_suffix(f"{live.suffix}.{uuid4().hex}.restore")
                    try:
                        shutil.copy2(backup, temporary)
                        temporary.replace(live)
                    finally:
                        temporary.unlink(missing_ok=True)
            except Exception as exc:
                restore_errors.append(f"{live}: {exc}")
        if restore_errors:
            preserve_backups = True
            locations = ", ".join(str(backup) for backup, _ in backups if backup is not None)
            raise RuntimeError(
                "Local index metadata installation and rollback failed; backups preserved at "
                f"{locations}: {'; '.join(restore_errors)}"
            ) from install_exc
        raise
    else:
        for staged, _ in files:
            staged.unlink()
    finally:
        if not preserve_backups:
            for backup, _ in backups:
                if backup is not None:
                    backup.unlink(missing_ok=True)


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
