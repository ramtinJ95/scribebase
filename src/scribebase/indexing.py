from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

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
) -> SourceManifest:
    manifest = find_source(config.data_dir, source_id)
    chunks = chunk_source(manifest, config)
    if not chunks:
        raise RuntimeError(f"No chunks created for source: {source_id}")
    logger.info("Created %s chunks", len(chunks))

    embedder = LlamaCppEmbeddingClient(config.embedding)
    vectors: list[list[float]] = []
    total_batches = (len(chunks) + config.embedding.batch_size - 1) // config.embedding.batch_size
    for batch_index, batch_vectors in enumerate(embedder.embed_batches([c.text for c in chunks]), start=1):
        logger.info("Embedding batch %s/%s", batch_index, total_batches)
        vectors.extend(batch_vectors)
    dimension = len(vectors[0]) if vectors else None
    _validate_embedding_dimension(config, dimension)
    for chunk in chunks:
        chunk.embedding_model = config.embedding.model
        chunk.embedding_dimension = dimension

    chunks_path = Path(manifest.data_dir) / "metadata" / "chunks.jsonl"
    write_jsonl(chunks_path, [chunk.model_dump(mode="json") for chunk in chunks])

    store = WeaviateStore(config.weaviate)
    try:
        if no_create_collection:
            client = store.connect()
            if not client.collections.exists(config.weaviate.collection):
                raise RuntimeError(f"Weaviate collection missing: {config.weaviate.collection}")
        else:
            store.ensure_collection()
        store.delete_source(source_id)
        store.upsert_chunks(chunks, vectors)
    finally:
        store.close()

    manifest.embedding_summary.embedding_model = config.embedding.model
    manifest.embedding_summary.embedding_dimension = dimension
    manifest.embedding_summary.embedding_base_url = config.embedding.base_url
    manifest.embedding_summary.indexed_in_weaviate = True
    manifest.embedding_summary.weaviate_collection = config.weaviate.collection
    manifest.updated_at = datetime.now(timezone.utc)
    write_manifest(manifest)
    logger.info("Indexed %s chunks into Weaviate collection %s", len(chunks), config.weaviate.collection)
    return manifest


def load_chunks(data_dir: Path, source_id: str | None = None) -> list[Chunk]:
    roots = [Path(find_source(data_dir, source_id).data_dir)] if source_id else sorted((data_dir / "sources").glob("*"))
    chunks: list[Chunk] = []
    for root in roots:
        path = root / "metadata" / "chunks.jsonl"
        chunks.extend(Chunk.model_validate(row) for row in read_jsonl(path))
    return chunks


def rebuild_index(source_id: str | None, all_sources: bool, config: AppConfig, logger) -> None:
    from scribebase.source_registry import list_manifests

    if not all_sources and not source_id:
        raise ValueError("Provide --source-id or --all")
    ids = [source_id] if source_id else [manifest.source_id for manifest in list_manifests(config.data_dir)]
    for sid in ids:
        if sid:
            index_source(sid, config, logger)


def _validate_embedding_dimension(config: AppConfig, dimension: int | None) -> None:
    from scribebase.source_registry import list_manifests

    if dimension is None:
        return
    for manifest in list_manifests(config.data_dir):
        summary = manifest.embedding_summary
        if not summary.indexed_in_weaviate:
            continue
        if summary.weaviate_collection != config.weaviate.collection:
            continue
        if summary.embedding_model == config.embedding.model and summary.embedding_dimension != dimension:
            raise RuntimeError(
                "Embedding dimension mismatch for existing index: "
                f"configured model produced {dimension}, but {manifest.source_id} stores "
                f"{summary.embedding_dimension}. Rebuild the index."
            )


def _chapter_file_name(chapter: str) -> str:
    from scribebase.paths import chapter_file_name

    return chapter_file_name(chapter)
