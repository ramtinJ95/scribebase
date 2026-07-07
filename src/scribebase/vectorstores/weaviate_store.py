from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from scribebase.config import WeaviateConfig
from scribebase.models import Chunk, SearchFilters, SearchResult


class WeaviateStore:
    def __init__(self, config: WeaviateConfig):
        self.config = config
        self.client = None

    def connect(self):
        import weaviate

        self.client = weaviate.connect_to_local(
            host=self.config.url.replace("http://", "").replace("https://", "").split(":")[0],
            port=int(self.config.url.rsplit(":", 1)[1]) if ":" in self.config.url.rsplit("//", 1)[-1] else 8080,
        )
        return self.client

    def close(self) -> None:
        if self.client is not None:
            self.client.close()

    def is_ready(self) -> bool:
        client = self.client or self.connect()
        return bool(client.is_ready())

    def ensure_collection(self) -> None:
        from weaviate.classes.config import Configure, DataType, Property

        client = self.client or self.connect()
        if client.collections.exists(self.config.collection):
            return
        client.collections.create(
            self.config.collection,
            vectorizer_config=[
                Configure.NamedVectors.none(
                    name=self.config.vector_name,
                    vector_index_config=Configure.VectorIndex.hnsw(),
                )
            ],
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="chunk_id", data_type=DataType.TEXT),
                Property(name="source_id", data_type=DataType.TEXT),
                Property(name="source_type", data_type=DataType.TEXT),
                Property(name="title", data_type=DataType.TEXT),
                Property(name="course", data_type=DataType.TEXT),
                Property(name="chapter", data_type=DataType.TEXT),
                Property(name="section", data_type=DataType.TEXT),
                Property(name="page_start", data_type=DataType.INT),
                Property(name="page_end", data_type=DataType.INT),
                Property(name="chunk_index", data_type=DataType.INT),
                Property(name="file_path", data_type=DataType.TEXT),
                Property(name="extraction_method", data_type=DataType.TEXT),
                Property(name="ocr_model", data_type=DataType.TEXT),
                Property(name="language", data_type=DataType.TEXT),
                Property(name="embedding_model", data_type=DataType.TEXT),
                Property(name="embedding_dimension", data_type=DataType.INT),
                Property(name="created_at", data_type=DataType.DATE),
            ],
        )

    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        from weaviate.util import generate_uuid5

        if len(chunks) != len(vectors):
            raise ValueError("chunk/vector length mismatch")
        self.ensure_collection()
        collection = (self.client or self.connect()).collections.get(self.config.collection)
        with collection.batch.dynamic() as batch:
            for chunk, vector in zip(chunks, vectors):
                props = _chunk_properties(chunk)
                batch.add_object(
                    properties=props,
                    vector={self.config.vector_name: vector},
                    uuid=generate_uuid5(chunk.chunk_id),
                )

    def delete_source(self, source_id: str) -> None:
        from weaviate.classes.query import Filter

        self.ensure_collection()
        collection = (self.client or self.connect()).collections.get(self.config.collection)
        collection.data.delete_many(where=Filter.by_property("source_id").equal(source_id))

    def hybrid_search(
        self,
        query: str,
        vector: list[float],
        filters: SearchFilters,
        top_k: int,
        alpha: float,
    ) -> list[SearchResult]:
        from weaviate.classes.query import MetadataQuery

        self.ensure_collection()
        collection = (self.client or self.connect()).collections.get(self.config.collection)
        kwargs: dict[str, Any] = {
            "query": query,
            "vector": vector,
            "alpha": alpha,
            "limit": top_k,
            "return_metadata": MetadataQuery(score=True, explain_score=True),
            "target_vector": self.config.vector_name,
        }
        where = build_filter(filters)
        if where is not None:
            kwargs["filters"] = where
        result = collection.query.hybrid(**kwargs)
        out: list[SearchResult] = []
        for obj in result.objects:
            props = dict(obj.properties)
            chunk = Chunk(**_props_to_chunk(props))
            meta = obj.metadata
            out.append(
                SearchResult(
                    chunk=chunk,
                    score=getattr(meta, "score", None),
                    explain_score=getattr(meta, "explain_score", None),
                )
            )
        return out


def build_filter(filters: SearchFilters):
    from weaviate.classes.query import Filter

    clauses = []
    for field in ["source_id", "title", "source_type", "course", "chapter", "section", "language"]:
        value = getattr(filters, field)
        if value is not None:
            clauses.append(Filter.by_property(field).equal(value))
    if filters.page_start is not None:
        clauses.append(Filter.by_property("page_end").greater_or_equal(filters.page_start))
    if filters.page_end is not None:
        clauses.append(Filter.by_property("page_start").less_or_equal(filters.page_end))
    if not clauses:
        return None
    current = clauses[0]
    for clause in clauses[1:]:
        current = current & clause
    return current


def _chunk_properties(chunk: Chunk) -> dict[str, Any]:
    data = chunk.model_dump(mode="json")
    data["created_at"] = data.get("created_at") or datetime.now(timezone.utc).isoformat()
    return {key: value for key, value in data.items() if value is not None}


def _props_to_chunk(props: dict[str, Any]) -> dict[str, Any]:
    if props.get("created_at") and not isinstance(props["created_at"], str):
        props["created_at"] = props["created_at"].isoformat()
    return props
