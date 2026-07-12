from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from scribebase.chunking.chunker import chunk_markdown
from scribebase.config import AppConfig
from scribebase.embeddings.llamacpp_client import LlamaCppEmbeddingClient
from scribebase.extraction import read_page_metadata
from scribebase.models import Chunk, SourceManifest
from scribebase.source_registry import find_source, read_jsonl, write_jsonl, write_manifest
from scribebase.vectorstores.weaviate_store import WeaviateStore


def chunk_source(manifest: SourceManifest, config: AppConfig) -> list[Chunk]:
    root = Path(manifest.data_dir)
    markdown_path = root / "markdown" / "document.md"
    if manifest.chapter:
        chapter_path = root / "markdown" / "chapters" / _chapter_file_name(manifest.chapter)
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
    collection_name: str | None = None,
    write_manifest_summary: bool = True,
) -> SourceManifest:
    manifest = find_source(config.data_dir, source_id)
    chunks = chunk_source(manifest, config)
    if not chunks:
        raise RuntimeError(f"No chunks created for source: {source_id}")
    logger.info("Created %s chunks", len(chunks))

    chunks_path = Path(manifest.data_dir) / "metadata" / "chunks.jsonl"
    old_chunk_ids = {chunk.chunk_id for chunk in load_chunks(config.data_dir, source_id)}
    embedder = LlamaCppEmbeddingClient(config.embedding)
    total_batches = (len(chunks) + config.embedding.batch_size - 1) // config.embedding.batch_size
    store = WeaviateStore(config.weaviate)
    dimension: int | None = None
    inserted = 0
    try:
        if no_create_collection:
            client = store.connect()
            target = collection_name or config.weaviate.collection
            if not client.collections.exists(target):
                raise RuntimeError(f"Weaviate collection missing: {target}")
        else:
            store.ensure_collection()
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
            store.upsert_chunks(batch_chunks, batch_vectors, collection_name=collection_name)
            inserted += len(batch_chunks)
        if inserted != len(chunks):
            raise RuntimeError(f"Embedded {inserted} of {len(chunks)} chunks")
        if collection_name is None:
            store.delete_chunks(old_chunk_ids - {chunk.chunk_id for chunk in chunks})
    finally:
        store.close()

    _write_chunks_atomic(chunks_path, chunks)

    _set_embedding_summary(manifest, config, dimension)
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
    from scribebase.source_registry import list_manifests

    if not all_sources and not source_id:
        raise ValueError("Provide --source-id or --all")
    ids = (
        [source_id]
        if source_id
        else [manifest.source_id for manifest in list_manifests(config.data_dir)]
    )
    if not all_sources:
        index_source(source_id or "", config, logger)
        return

    staging = f"{config.weaviate.collection}Build{datetime.now(timezone.utc):%Y%m%d%H%M%S}{uuid4().hex[:6]}"
    store = WeaviateStore(config.weaviate)
    try:
        logger.info("Building staged Weaviate collection %s", staging)
        store.create_collection(staging)
        expected = 0
        rebuilt: list[SourceManifest] = []
        promoted = False
        try:
            for sid in ids:
                if sid:
                    manifest = index_source(
                        sid,
                        config,
                        logger,
                        no_create_collection=True,
                        allow_existing_model_mismatch=True,
                        collection_name=staging,
                        write_manifest_summary=False,
                    )
                    rebuilt.append(manifest)
                    expected += len(load_chunks(config.data_dir, manifest.source_id))
            actual = store.object_count(staging)
            if actual != expected:
                raise RuntimeError(
                    f"Staged index verification failed: expected {expected} chunks, found {actual}"
                )
            previous = store.promote_collection(staging)
            promoted = True
            logger.info("Promoted %s as alias %s", staging, config.weaviate.collection)
            for manifest in rebuilt:
                write_manifest(manifest)
            if previous and previous != staging:
                try:
                    store.delete_collection(previous)
                except Exception as exc:
                    logger.warning("Could not remove previous collection %s: %s", previous, exc)
        except Exception:
            if not promoted:
                try:
                    store.delete_collection(staging)
                except Exception as exc:
                    logger.warning(
                        "Could not remove failed staging collection %s: %s", staging, exc
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


def _chapter_file_name(chapter: str) -> str:
    from scribebase.paths import chapter_file_name

    return chapter_file_name(chapter)


def _write_chunks_atomic(path: Path, chunks: list[Chunk]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.{uuid4().hex}.tmp")
    try:
        write_jsonl(temporary, [chunk.model_dump(mode="json") for chunk in chunks])
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _set_embedding_summary(
    manifest: SourceManifest,
    config: AppConfig,
    dimension: int | None,
) -> None:
    manifest.embedding_summary.embedding_model = config.embedding.model
    manifest.embedding_summary.embedding_dimension = dimension
    manifest.embedding_summary.embedding_base_url = config.embedding.base_url
    manifest.embedding_summary.indexed_in_weaviate = True
    manifest.embedding_summary.weaviate_collection = config.weaviate.collection
    manifest.updated_at = datetime.now(timezone.utc)
