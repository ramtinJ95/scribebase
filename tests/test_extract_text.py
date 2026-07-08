import logging
from datetime import datetime, timezone

import pytest

from scribebase.config import AppConfig
from scribebase.extraction import extract_source, read_page_metadata
from scribebase.indexing import chunk_source


def test_extract_plain_text_document(tmp_path) -> None:
    text = tmp_path / "notes.txt"
    text.write_text("Line one.\n\nLine two about Kubernetes scheduling.", encoding="utf-8")
    config = AppConfig(data_dir=tmp_path / ".study_local")

    manifest = extract_source(
        text,
        title="Scheduling Notes",
        source_type="notes",
        course=None,
        chapter=None,
        language="en",
        ocr="auto",
        config=config,
        logger=logging.getLogger("test"),
    )

    root = config.data_dir / "sources" / manifest.source_id
    document = root / "markdown" / "document.md"
    assert document.exists()
    assert "Line two about Kubernetes scheduling." in document.read_text()
    pages = read_page_metadata(root)
    assert pages[0].input_type == "text"
    assert pages[0].extraction_method == "text"
    assert pages[0].page_number == 1
    assert manifest.extraction_summary.pages_total == 1
    assert manifest.extraction_summary.pages_ocr == 0

    chunks = chunk_source(manifest, config)
    assert chunks
    assert chunks[0].source_type == "notes"
    assert chunks[0].extraction_method == "text"


def test_extract_markdown_document_preserves_markdown(tmp_path) -> None:
    markdown = tmp_path / "article.md"
    markdown.write_text(
        "# Argo CD Notes\n\nGitOps reconciles declared state.\n\n## Sync waves\n\nSync waves order resources.",
        encoding="utf-8",
    )
    config = AppConfig(data_dir=tmp_path / ".study_local")

    manifest = extract_source(
        markdown,
        title="Argo CD Notes",
        source_type="article",
        course=None,
        chapter=None,
        language="en",
        ocr="auto",
        config=config,
        logger=logging.getLogger("test"),
    )

    root = config.data_dir / "sources" / manifest.source_id
    document_text = (root / "markdown" / "document.md").read_text()
    assert "# Argo CD Notes" in document_text
    assert "## Sync waves" in document_text
    pages = read_page_metadata(root)
    assert pages[0].input_type == "markdown"
    assert pages[0].extraction_method == "markdown"

    chunks = chunk_source(manifest, config)
    assert chunks[0].section == "Sync waves"
    assert "# Argo CD Notes" in chunks[0].text
    assert chunks[0].extraction_method == "markdown"


def test_extract_empty_text_document_fails_before_page_marker(tmp_path) -> None:
    text = tmp_path / "empty.txt"
    text.write_text("\n\t  ", encoding="utf-8")
    config = AppConfig(data_dir=tmp_path / ".study_local")

    with pytest.raises(RuntimeError, match="Empty text document"):
        extract_source(
            text,
            title="Empty Notes",
            source_type="notes",
            course=None,
            chapter=None,
            language="en",
            ocr="auto",
            config=config,
            logger=logging.getLogger("test"),
        )

    sources = list((config.data_dir / "sources").glob("*/markdown/document.md"))
    assert sources == []


def test_extract_text_document_persists_generic_metadata(tmp_path) -> None:
    text = tmp_path / "snippet.txt"
    text.write_text("A note about service meshes.", encoding="utf-8")
    config = AppConfig(data_dir=tmp_path / ".study_local")

    manifest = extract_source(
        text,
        title="Service Mesh Note",
        source_type="snippet",
        course=None,
        chapter=None,
        language="en",
        ocr="auto",
        config=config,
        logger=logging.getLogger("test"),
        tags="kubernetes, networking",
        origin="manual",
        publisher="Personal",
        author="Ramtin",
        created_at_source="2026-07-08",
        updated_at_source=datetime(2026, 7, 9, tzinfo=timezone.utc),
        retrieved_at="2026-07-10T12:00:00Z",
        url="https://example.com/source",
        canonical_url="https://example.com/source",
        external_id="note-1",
        collection="infra-reading",
        summary="Service mesh note.",
    )

    assert manifest.tags == ["kubernetes", "networking"]
    assert manifest.origin == "manual"
    assert manifest.collection == "infra-reading"
    assert manifest.created_at_source == datetime(2026, 7, 8)
    chunks = chunk_source(manifest, config)
    assert chunks[0].tags == ["kubernetes", "networking"]
    assert chunks[0].origin == "manual"
    assert chunks[0].publisher == "Personal"
    assert chunks[0].external_id == "note-1"
    assert chunks[0].collection == "infra-reading"


def test_extract_markdown_frontmatter_supplies_metadata_and_is_removed(tmp_path) -> None:
    markdown = tmp_path / "article.md"
    markdown.write_text(
        "---\n"
        "title: Frontmatter Article\n"
        "source_type: article\n"
        "language: en\n"
        "tags: [kubernetes, gitops]\n"
        "origin: company_blog\n"
        "publisher: Example Blog\n"
        "created_at_source: '2026-07-08'\n"
        "url: https://example.com/gitops\n"
        "collection: infra-reading\n"
        "---\n\n"
        "# GitOps\n\nArgo CD reconciles declared state.",
        encoding="utf-8",
    )
    config = AppConfig(data_dir=tmp_path / ".study_local")

    manifest = extract_source(
        markdown,
        title=None,
        source_type=None,
        course=None,
        chapter=None,
        language=None,
        ocr="auto",
        config=config,
        logger=logging.getLogger("test"),
    )

    root = config.data_dir / "sources" / manifest.source_id
    document_text = (root / "markdown" / "document.md").read_text()
    assert "title: Frontmatter Article" not in document_text
    assert "# GitOps" in document_text
    assert manifest.title == "Frontmatter Article"
    assert manifest.source_type == "article"
    assert manifest.language == "en"
    assert manifest.tags == ["kubernetes", "gitops"]
    assert manifest.origin == "company_blog"
    assert manifest.publisher == "Example Blog"
    assert manifest.url == "https://example.com/gitops"
    assert manifest.collection == "infra-reading"

    chunks = chunk_source(manifest, config)
    assert chunks[0].title == "Frontmatter Article"
    assert chunks[0].tags == ["kubernetes", "gitops"]
    assert chunks[0].origin == "company_blog"


def test_extract_markdown_explicit_metadata_overrides_frontmatter(tmp_path) -> None:
    markdown = tmp_path / "article.md"
    markdown.write_text(
        "---\n"
        "title: Frontmatter Title\n"
        "source_type: article\n"
        "language: en\n"
        "tags: [frontmatter]\n"
        "origin: company_blog\n"
        "collection: old\n"
        "---\n\n"
        "Body text.",
        encoding="utf-8",
    )
    config = AppConfig(data_dir=tmp_path / ".study_local")

    manifest = extract_source(
        markdown,
        title="Explicit Title",
        source_type="notes",
        course=None,
        chapter=None,
        language="sv",
        ocr="auto",
        config=config,
        logger=logging.getLogger("test"),
        tags="explicit, tags",
        origin="manual",
        collection="new",
    )

    assert manifest.title == "Explicit Title"
    assert manifest.source_type == "notes"
    assert manifest.language == "sv"
    assert manifest.tags == ["explicit", "tags"]
    assert manifest.origin == "manual"
    assert manifest.collection == "new"


def test_extract_markdown_invalid_frontmatter_fails(tmp_path) -> None:
    markdown = tmp_path / "bad.md"
    markdown.write_text(
        "---\n"
        "tags: 123\n"
        "created_at_source: not-a-date\n"
        "---\n\n"
        "Body text.",
        encoding="utf-8",
    )
    config = AppConfig(data_dir=tmp_path / ".study_local")

    with pytest.raises(ValueError, match="Invalid Markdown frontmatter"):
        extract_source(
            markdown,
            title="Bad Frontmatter",
            source_type="article",
            course=None,
            chapter=None,
            language="en",
            ocr="auto",
            config=config,
            logger=logging.getLogger("test"),
        )
