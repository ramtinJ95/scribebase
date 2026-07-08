import logging

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
