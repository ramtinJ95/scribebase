from scribebase.models import Chunk, SearchResult
from scribebase.retrieval.context_pack import build_context_pack


def test_context_pack_includes_citations_and_metadata() -> None:
    chunk = Chunk(
        chunk_id="src_ch4_p87_0001",
        source_id="src",
        source_type="book",
        title="Cognitive Psychology",
        chapter="4",
        section="4.2 Working Memory",
        page_start=87,
        page_end=88,
        chunk_index=1,
        text="Working memory has limited capacity.",
        file_path="document.md",
        extraction_method="pymupdf4llm",
        embedding_model="model",
        embedding_dimension=2,
    )
    pack = build_context_pack("Explain working memory.", [SearchResult(chunk=chunk)])
    assert "# Context Pack" in pack
    assert "Pages: 87–88" in pack
    assert "Chunk ID: src_ch4_p87_0001" in pack
    assert "Cite sources as [Title, p. 87]" in pack
