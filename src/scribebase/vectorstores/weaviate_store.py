from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from scribebase.config import WeaviateConfig
from scribebase.models import Chunk, SearchFilters, SearchResult


WEAVIATE_CHUNK_PROPERTIES = {
    "text",
    "chunk_id",
    "source_id",
    "source_type",
    "title",
    "course",
    "chapter",
    "tags",
    "origin",
    "publisher",
    "author",
    "created_at_source",
    "updated_at_source",
    "retrieved_at",
    "url",
    "canonical_url",
    "external_id",
    "collection",
    "summary",
    "section",
    "page_start",
    "page_end",
    "chunk_index",
    "file_path",
    "extraction_method",
    "ocr_model",
    "language",
    "embedding_model",
    "embedding_dimension",
    "created_at",
}


class CollectionAliasMigrationError(RuntimeError):
    def __init__(self, alias_name: str, target: str):
        self.alias_name = alias_name
        self.target = target
        super().__init__(
            f"Could not create alias {alias_name!r} after removing its legacy physical collection. "
            f"Verified data remains in {target!r}; retry alias creation before deleting it."
        )


class WeaviateStore:
    def __init__(self, config: WeaviateConfig):
        self.config = config
        self.client = None

    def connect(self):
        import weaviate

        self.client = weaviate.connect_to_local(
            host=self.config.url.replace("http://", "").replace("https://", "").split(":")[0],
            port=int(self.config.url.rsplit(":", 1)[1])
            if ":" in self.config.url.rsplit("//", 1)[-1]
            else 8080,
        )
        return self.client

    def close(self) -> None:
        if self.client is not None:
            self.client.close()

    def is_ready(self) -> bool:
        client = self.client or self.connect()
        return bool(client.is_ready())

    def ensure_collection(self) -> None:
        client = self.client or self.connect()
        if client.collections.exists(self.config.collection):
            return
        if client.alias.get(alias_name=self.config.collection) is not None:
            return
        physical_name = f"{self.config.collection}Index{uuid4().hex[:10]}"
        self.create_collection(physical_name)
        client.alias.create(
            alias_name=self.config.collection,
            target_collection=physical_name,
        )

    def create_collection(self, name: str) -> None:
        from weaviate.classes.config import Configure, DataType, Property

        client = self.client or self.connect()
        if client.collections.exists(name):
            return
        client.collections.create(
            name,
            vector_config=[
                Configure.Vectors.self_provided(
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
                Property(name="tags", data_type=DataType.TEXT_ARRAY),
                Property(name="origin", data_type=DataType.TEXT),
                Property(name="publisher", data_type=DataType.TEXT),
                Property(name="author", data_type=DataType.TEXT),
                Property(name="created_at_source", data_type=DataType.DATE),
                Property(name="updated_at_source", data_type=DataType.DATE),
                Property(name="retrieved_at", data_type=DataType.DATE),
                Property(name="url", data_type=DataType.TEXT),
                Property(name="canonical_url", data_type=DataType.TEXT),
                Property(name="external_id", data_type=DataType.TEXT),
                Property(name="collection", data_type=DataType.TEXT),
                Property(name="summary", data_type=DataType.TEXT),
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

    def reset_collection(self) -> None:
        client = self.client or self.connect()
        if client.collections.exists(self.config.collection):
            client.collections.delete(self.config.collection)
        self.ensure_collection()

    def promote_collection(self, target: str) -> str | None:
        client = self.client or self.connect()
        alias = client.alias.get(alias_name=self.config.collection)
        previous = alias.collection if alias is not None else None
        if alias is not None:
            if not client.alias.update(
                alias_name=self.config.collection,
                new_target_collection=target,
            ):
                raise RuntimeError(f"Failed to promote Weaviate collection: {target}")
            return previous
        if client.collections.exists(self.config.collection):
            client.collections.delete(self.config.collection)
        try:
            client.alias.create(alias_name=self.config.collection, target_collection=target)
        except Exception as exc:
            alias = client.alias.get(alias_name=self.config.collection)
            if alias is not None and alias.collection == target:
                return None
            raise CollectionAliasMigrationError(self.config.collection, target) from exc
        return None

    def delete_collection(self, name: str) -> None:
        client = self.client or self.connect()
        if client.collections.exists(name):
            client.collections.delete(name)

    def object_count(self, collection_name: str | None = None) -> int:
        name = collection_name or self.config.collection
        collection = (self.client or self.connect()).collections.use(name)
        result = collection.aggregate.over_all(total_count=True)
        return int(result.total_count or 0)

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        collection_name: str | None = None,
    ) -> None:
        from weaviate.util import generate_uuid5

        if len(chunks) != len(vectors):
            raise ValueError("chunk/vector length mismatch")
        if collection_name is None:
            self.ensure_collection()
        name = collection_name or self.config.collection
        collection = (self.client or self.connect()).collections.use(name)
        with collection.batch.dynamic() as batch:
            for chunk, vector in zip(chunks, vectors):
                props = _chunk_properties(chunk)
                batch.add_object(
                    properties=props,
                    vector={self.config.vector_name: vector},
                    uuid=generate_uuid5(chunk.chunk_id),
                )
        failed_objects = getattr(collection.batch, "failed_objects", None)
        if failed_objects:
            raise RuntimeError(f"Failed to insert {len(failed_objects)} chunks into Weaviate")

    def delete_chunks(self, chunk_ids: set[str]) -> None:
        from weaviate.util import generate_uuid5

        if not chunk_ids:
            return
        self.ensure_collection()
        collection = (self.client or self.connect()).collections.use(self.config.collection)
        for chunk_id in chunk_ids:
            collection.data.delete_by_id(generate_uuid5(chunk_id))

    def iter_source_chunks(
        self,
        source_id: str,
        include_vectors: bool = False,
        page_size: int = 100,
    ):
        from weaviate.classes.query import Filter

        self.ensure_collection()
        collection = (self.client or self.connect()).collections.use(self.config.collection)
        offset = 0
        while True:
            result = collection.query.fetch_objects(
                filters=Filter.by_property("source_id").equal(source_id),
                include_vector=self.config.vector_name if include_vectors else False,
                limit=page_size,
                offset=offset,
            )
            if not result.objects:
                return
            for obj in result.objects:
                vector = None
                if include_vectors:
                    vectors = obj.vector or {}
                    vector = vectors.get(self.config.vector_name)
                    if vector is None:
                        raise RuntimeError(
                            f"Missing vector {self.config.vector_name!r} for source {source_id}"
                        )
                yield Chunk(**_props_to_chunk(dict(obj.properties))), vector
            if len(result.objects) < page_size:
                return
            offset += len(result.objects)

    def delete_source(self, source_id: str) -> None:
        from weaviate.classes.query import Filter

        self.ensure_collection()
        collection = (self.client or self.connect()).collections.use(self.config.collection)
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
        collection = (self.client or self.connect()).collections.use(self.config.collection)
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
    for field in [
        "source_id",
        "title",
        "source_type",
        "course",
        "chapter",
        "section",
        "language",
        "origin",
        "publisher",
        "author",
        "url",
        "canonical_url",
        "external_id",
        "collection",
    ]:
        value = getattr(filters, field)
        if value is not None:
            clauses.append(Filter.by_property(field).equal(value))
    if filters.tags:
        clauses.append(Filter.by_property("tags").contains_all(filters.tags))
    for field in ["created_at_source", "updated_at_source", "retrieved_at"]:
        after = getattr(filters, f"{field}_after")
        before = getattr(filters, f"{field}_before")
        if after is not None:
            clauses.append(Filter.by_property(field).greater_or_equal(after))
        if before is not None:
            clauses.append(Filter.by_property(field).less_or_equal(before))
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
    return {
        key: value
        for key, value in data.items()
        if key in WEAVIATE_CHUNK_PROPERTIES and value is not None and value != []
    }


def _props_to_chunk(props: dict[str, Any]) -> dict[str, Any]:
    for field in ["created_at", "created_at_source", "updated_at_source", "retrieved_at"]:
        if props.get(field) and not isinstance(props[field], str):
            props[field] = props[field].isoformat()
    return props
