import logging

import fitz

from scribebase.config import AppConfig, PDFDetectionConfig
from scribebase.extraction import extract_source


def test_extract_true_text_pdf_without_ocr(tmp_path) -> None:
    pdf = tmp_path / "book.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Working memory has limited capacity. " * 40)
    doc.save(pdf)
    doc.close()

    config = AppConfig(
        data_dir=tmp_path / ".study_local",
        pdf_detection=PDFDetectionConfig(min_chars_per_page=50),
    )
    manifest = extract_source(
        pdf,
        title="Tiny Book",
        source_type="book",
        course=None,
        chapter="1",
        language="en",
        ocr="never",
        config=config,
        logger=logging.getLogger("test"),
    )

    root = config.data_dir / "sources" / manifest.source_id
    assert (root / "markdown" / "page_0001.md").exists()
    assert (root / "markdown" / "document.md").read_text().startswith("<!-- page: 1 -->")
    assert manifest.extraction_summary.pages_total == 1
    assert manifest.extraction_summary.pages_ocr == 0
