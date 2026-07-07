from __future__ import annotations

from scribebase.config import AppConfig
from scribebase.embeddings.llamacpp_client import LlamaCppEmbeddingClient
from scribebase.models import SearchFilters, SearchResult
from scribebase.vectorstores.weaviate_store import WeaviateStore


def search_chunks(
    query: str,
    filters: SearchFilters,
    config: AppConfig,
    top_k: int | None = None,
    alpha: float | None = None,
    allow_model_mismatch: bool = False,
) -> list[SearchResult]:
    embedder = LlamaCppEmbeddingClient(config.embedding)
    vector = embedder.embed_query(query)
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
        {
            result.chunk.embedding_model
            for result in results
            if result.chunk.embedding_model and result.chunk.embedding_model != config.embedding.model
        }
    )
    if mismatches and not allow_model_mismatch:
        raise RuntimeError(
            "Embedding model mismatch. Current config uses "
            f"{config.embedding.model!r}, but results use {mismatches}. "
            "Rebuild the index or pass --allow-model-mismatch."
        )
    return results


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
