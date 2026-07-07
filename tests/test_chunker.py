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
