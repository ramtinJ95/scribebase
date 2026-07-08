from datetime import datetime, timezone

from scribebase.config import ChunkingConfig
from scribebase.chunking.chunker import chunk_markdown
from scribebase.models import PageMetadata, SourceManifest


def test_chunker_preserves_page_metadata(tmp_path) -> None:
    md = tmp_path / "document.md"
    md.write_text(
        "<!-- page: 1 -->\n\n## Page 1\n\n# Chapter 4\n\nWorking memory text. "
        * 30
        + "\n\n<!-- page: 2 -->\n\n## Page 2\n\nMore detail. "
        * 40
    )
    manifest = SourceManifest(
        source_id="src",
        title="Cognitive Psychology",
        source_type="book",
        chapter="4",
        language="en",
        original_path="source.pdf",
        data_dir=str(tmp_path),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    pages = [
        PageMetadata(
            source_id="src",
            page_number=1,
            page_index=0,
            input_type="pdf_page",
            text_layer_detected=True,
            extraction_method="pymupdf4llm",
            markdown_path="p1.md",
        ),
        PageMetadata(
            source_id="src",
            page_number=2,
            page_index=1,
            input_type="pdf_page",
            text_layer_detected=True,
            extraction_method="pymupdf4llm",
            markdown_path="p2.md",
        ),
    ]
    chunks = chunk_markdown(md, manifest, pages, ChunkingConfig(target_chars=500, min_chars=100))
    assert chunks
    assert chunks[0].source_id == "src"
    assert chunks[0].page_start == 1
    assert any(chunk.page_end == 2 or chunk.page_start == 2 for chunk in chunks)


def test_chunker_infers_explicit_chapters(tmp_path) -> None:
    md = tmp_path / "document.md"
    md.write_text(
        "<!-- page: 1 -->\n\n## Page 1\n\n# **CHAPTER 1 Introduction**\n\n"
        + "Alpha text. " * 30
        + "\n\n<!-- page: 2 -->\n\n## Page 2\n\n# **CHAPTER 2 Scheduling**\n\n"
        + "Beta text. " * 30
    )
    manifest = _manifest(tmp_path, chapter=None)
    chunks = chunk_markdown(md, manifest, _pages(), ChunkingConfig(target_chars=500, min_chars=100))

    chapters = {chunk.chapter for chunk in chunks}
    assert "CHAPTER 1 Introduction" in chapters
    assert "CHAPTER 2 Scheduling" in chapters


def test_chunker_infers_untitled_chapter_from_this_chapter_covers(tmp_path) -> None:
    md = tmp_path / "document.md"
    md.write_text(
        "<!-- page: 1 -->\n\n## Page 1\n\n# _Introducing Kubernetes_\n\n"
        "## _This chapter covers_\n\n- Clusters\n\nBody text. "
        + "More body text. " * 30
    )
    manifest = _manifest(tmp_path, chapter=None)
    chunks = chunk_markdown(md, manifest, _pages(), ChunkingConfig(target_chars=500, min_chars=100))

    assert {chunk.chapter for chunk in chunks} == {"Chapter 1: Introducing Kubernetes"}


def test_chunker_manifest_chapter_overrides_inferred_chapter(tmp_path) -> None:
    md = tmp_path / "document.md"
    md.write_text("<!-- page: 1 -->\n\n## Page 1\n\n# 1: Inferred\n\n" + "Body text. " * 30)
    manifest = _manifest(tmp_path, chapter="Manual Chapter")
    chunks = chunk_markdown(md, manifest, _pages(), ChunkingConfig(target_chars=500, min_chars=100))

    assert {chunk.chapter for chunk in chunks} == {"Manual Chapter"}


def _manifest(tmp_path, chapter: str | None) -> SourceManifest:
    return SourceManifest(
        source_id="src",
        title="Kubernetes",
        source_type="book",
        chapter=chapter,
        language="en",
        original_path="source.pdf",
        data_dir=str(tmp_path),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def _pages() -> list[PageMetadata]:
    return [
        PageMetadata(
            source_id="src",
            page_number=1,
            page_index=0,
            input_type="pdf_page",
            text_layer_detected=True,
            extraction_method="pymupdf4llm",
            markdown_path="p1.md",
        ),
        PageMetadata(
            source_id="src",
            page_number=2,
            page_index=1,
            input_type="pdf_page",
            text_layer_detected=True,
            extraction_method="pymupdf4llm",
            markdown_path="p2.md",
        ),
    ]
