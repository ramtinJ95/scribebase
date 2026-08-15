from __future__ import annotations

from scribebase.config import AppConfig
from scribebase.embeddings.llamacpp_client import LlamaCppEmbeddingClient
from scribebase.embeddings.profile import embedding_profile_fingerprint
from scribebase.models import SearchFilters, SearchResult
from scribebase.source_registry import list_manifests
from scribebase.vectorstores.weaviate_store import WeaviateStore


def search_chunks(
    query: str,
    filters: SearchFilters,
    config: AppConfig,
    top_k: int | None = None,
    alpha: float | None = None,
) -> list[SearchResult]:
    embedder = LlamaCppEmbeddingClient(config.embedding)
    vector = embedder.embed_query(query)
    expected_profile = embedding_profile_fingerprint(config.embedding, len(vector))
    _validate_manifest_profiles(config, expected_profile, len(vector))
    store = WeaviateStore(config.weaviate)
    try:
        results = store.hybrid_search(
            query=query,
            vector=vector,
            filters=filters,
            top_k=top_k or config.retrieval.top_k,
            alpha=alpha if alpha is not None else config.retrieval.alpha,
        )
    finally:
        store.close()
    mismatches = sorted(
        result.chunk.chunk_id
        for result in results
        if result.chunk.embedding_profile_fingerprint != expected_profile
    )
    if mismatches:
        raise RuntimeError(
            "Embedding profile mismatch in search results for chunks "
            f"{mismatches[:3]}. Run `scribebase rebuild-index --all`."
        )
    return results


def _validate_manifest_profiles(
    config: AppConfig, expected_profile: str, dimension: int
) -> None:
    for manifest in list_manifests(config.data_dir):
        summary = manifest.embedding_summary
        if not summary.indexed_in_weaviate:
            continue
        if summary.weaviate_collection != config.weaviate.collection:
            continue
        if summary.embedding_profile_fingerprint is None:
            raise RuntimeError(
                "Embedding profile metadata is missing for indexed source "
                f"{manifest.source_id!r}. Run `scribebase rebuild-index --all`."
            )
        if summary.embedding_dimension != dimension:
            raise RuntimeError(
                "Embedding dimension mismatch for indexed source "
                f"{manifest.source_id!r}. Run `scribebase rebuild-index --all`."
            )
        if summary.embedding_profile_fingerprint != expected_profile:
            raise RuntimeError(
                "Embedding profile mismatch for indexed source "
                f"{manifest.source_id!r}. Run `scribebase rebuild-index --all`."
            )


def format_search_results(results: list[SearchResult]) -> str:
    lines: list[str] = []
    for i, result in enumerate(results, start=1):
        c = result.chunk
        pages = _pages(c.page_start, c.page_end)
        snippet = c.text.replace("\n", " ")[:300]
        lines.extend(
            [
                f"{i}. {c.title}, chapter {c.chapter or '-'}, section {c.section or '-'}, pages {pages}",
                f"   score: {result.score if result.score is not None else '-'}",
                f"   chunk_id: {c.chunk_id}",
                f"   snippet: {snippet}",
            ]
        )
    return "\n".join(lines)


def _pages(start: int | None, end: int | None) -> str:
    if start is None:
        return "unknown"
    if not end or end == start:
        return str(start)
    return f"{start}–{end}"
