from scribebase.extraction import read_page_metadata
from scribebase.models import PageMetadata


def test_page_metadata_round_trip(tmp_path) -> None:
    meta_dir = tmp_path / "metadata"
    meta_dir.mkdir()
    page = PageMetadata(
        source_id="src",
        page_number=1,
        page_index=0,
        input_type="image",
        text_layer_detected=False,
        extraction_method="ocr",
        ocr_provider="shell",
        markdown_path="page_0001.md",
    )
    (meta_dir / "page_0001.json").write_text(page.model_dump_json())
    loaded = read_page_metadata(tmp_path)
    assert loaded[0].ocr_provider == "shell"
