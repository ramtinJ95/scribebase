from __future__ import annotations

from datetime import datetime, timezone

from scribebase.models import Chunk, SearchFilters
from scribebase.vectorstores import weaviate_store


class FakeExpr:
    def __init__(self, op: str, prop: str | None = None, value=None, children=None):  # noqa: ANN001
        self.op = op
        self.prop = prop
        self.value = value
        self.children = children or []

    def __and__(self, other):  # noqa: ANN001
        return FakeExpr("and", children=[self, other])


class FakeProp:
    def __init__(self, prop: str):
        self.prop = prop

    def equal(self, value):  # noqa: ANN001
        return FakeExpr("equal", self.prop, value)

    def contains_all(self, value):  # noqa: ANN001
        return FakeExpr("contains_all", self.prop, value)

    def greater_or_equal(self, value):  # noqa: ANN001
        return FakeExpr("greater_or_equal", self.prop, value)

    def less_or_equal(self, value):  # noqa: ANN001
        return FakeExpr("less_or_equal", self.prop, value)


class FakeFilter:
    @staticmethod
    def by_property(prop: str) -> FakeProp:
        return FakeProp(prop)


def _flatten(expr: FakeExpr) -> list[FakeExpr]:
    if expr.op != "and":
        return [expr]
    out: list[FakeExpr] = []
    for child in expr.children:
        out.extend(_flatten(child))
    return out


def test_chunk_properties_include_generic_metadata() -> None:
    created = datetime(2026, 7, 8, tzinfo=timezone.utc)
    chunk = Chunk(
        chunk_id="chunk-1",
        source_id="source-1",
        source_type="article",
        title="GitOps Article",
        tags=["kubernetes", "gitops"],
        origin="company_blog",
        publisher="Example Blog",
        author="Author",
        created_at_source=created,
        updated_at_source=created,
        retrieved_at=created,
        url="https://example.com/gitops",
        canonical_url="https://example.com/gitops",
        external_id="article-1",
        collection="infra-reading",
        summary="Article summary.",
        chunk_index=0,
        text="Argo CD reconciles declared state.",
        file_path="document.md",
        extraction_method="markdown",
        language="en",
    )

    props = weaviate_store._chunk_properties(chunk)

    assert props["tags"] == ["kubernetes", "gitops"]
    assert props["origin"] == "company_blog"
    assert props["publisher"] == "Example Blog"
    assert props["author"] == "Author"
    assert props["created_at_source"] == "2026-07-08T00:00:00Z"
    assert props["retrieved_at"] == "2026-07-08T00:00:00Z"
    assert props["url"] == "https://example.com/gitops"
    assert props["external_id"] == "article-1"
    assert props["collection"] == "infra-reading"
    assert props["summary"] == "Article summary."


def test_props_to_chunk_converts_weaviate_datetimes() -> None:
    created = datetime(2026, 7, 8, tzinfo=timezone.utc)
    props = {
        "chunk_id": "chunk-1",
        "source_id": "source-1",
        "source_type": "article",
        "title": "GitOps Article",
        "chunk_index": 0,
        "text": "Body",
        "file_path": "document.md",
        "extraction_method": "markdown",
        "created_at_source": created,
        "updated_at_source": created,
        "retrieved_at": created,
        "created_at": created,
    }

    converted = weaviate_store._props_to_chunk(props)

    assert converted["created_at_source"] == "2026-07-08T00:00:00+00:00"
    assert converted["updated_at_source"] == "2026-07-08T00:00:00+00:00"
    assert converted["retrieved_at"] == "2026-07-08T00:00:00+00:00"
    assert converted["created_at"] == "2026-07-08T00:00:00+00:00"


def test_build_filter_includes_generic_metadata(monkeypatch) -> None:
    monkeypatch.setattr("weaviate.classes.query.Filter", FakeFilter)
    filters = SearchFilters(
        source_type="article",
        tags=["kubernetes", "gitops"],
        origin="company_blog",
        publisher="Example Blog",
        author="Author",
        url="https://example.com/gitops",
        canonical_url="https://example.com/gitops",
        external_id="article-1",
        collection="infra-reading",
        created_at_source_after="2026-07-01T00:00:00Z",
        created_at_source_before="2026-07-31T00:00:00Z",
        retrieved_at_after="2026-07-08T00:00:00Z",
        retrieved_at_before="2026-07-09T00:00:00Z",
    )

    expr = weaviate_store.build_filter(filters)
    clauses = {(clause.op, clause.prop): clause.value for clause in _flatten(expr)}

    assert clauses[("equal", "source_type")] == "article"
    assert clauses[("contains_all", "tags")] == ["kubernetes", "gitops"]
    assert clauses[("equal", "origin")] == "company_blog"
    assert clauses[("equal", "publisher")] == "Example Blog"
    assert clauses[("equal", "author")] == "Author"
    assert clauses[("equal", "url")] == "https://example.com/gitops"
    assert clauses[("equal", "canonical_url")] == "https://example.com/gitops"
    assert clauses[("equal", "external_id")] == "article-1"
    assert clauses[("equal", "collection")] == "infra-reading"
    assert clauses[("greater_or_equal", "created_at_source")].isoformat().startswith("2026-07-01")
    assert clauses[("less_or_equal", "created_at_source")].isoformat().startswith("2026-07-31")
    assert clauses[("greater_or_equal", "retrieved_at")].isoformat().startswith("2026-07-08")
    assert clauses[("less_or_equal", "retrieved_at")].isoformat().startswith("2026-07-09")
