import logging
from types import SimpleNamespace

import fitz

from scribebase.config import AppConfig, PDFDetectionConfig
from scribebase.extraction import extract_source
from scribebase.models import OCRResult


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


def test_auto_true_text_pdf_does_not_ocr_blank_pages(tmp_path, monkeypatch) -> None:
    pdf = tmp_path / "book-with-blank.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Working memory has limited capacity. " * 40)
    doc.new_page()
    doc.save(pdf)
    doc.close()

    def fail_ocr_provider(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("auto should not OCR blank pages in true-text PDFs")

    monkeypatch.setattr("scribebase.extraction._ocr_provider", fail_ocr_provider)
    config = AppConfig(
        data_dir=tmp_path / ".study_local",
        pdf_detection=PDFDetectionConfig(min_chars_per_page=50),
    )

    manifest = extract_source(
        pdf,
        title="Book With Blank",
        source_type="book",
        course=None,
        chapter=None,
        language="en",
        ocr="auto",
        config=config,
        logger=logging.getLogger("test"),
    )

    assert manifest.extraction_summary.pages_total == 2
    assert manifest.extraction_summary.pages_ocr == 0


def test_auto_scanned_pdf_ocr_image_backed_pages(tmp_path, monkeypatch) -> None:
    image = tmp_path / "scan.png"
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 20, 20), 0)
    pix.clear_with(255)
    pix.save(image)

    pdf = tmp_path / "scan.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_image(fitz.Rect(72, 72, 200, 200), filename=image)
    doc.save(pdf)
    doc.close()

    class FakeOCRProvider:
        name = "fake"
        config = SimpleNamespace(render_dpi=None)

        def ocr_image(self, image_path, output_md_path, metadata):  # noqa: ANN001
            return OCRResult(
                markdown_path=output_md_path,
                text="OCR text from scanned page",
                provider=self.name,
                model="fake-ocr",
            )

    monkeypatch.setattr("scribebase.extraction._ocr_provider", lambda *_: FakeOCRProvider())
    config = AppConfig(data_dir=tmp_path / ".study_local")

    manifest = extract_source(
        pdf,
        title="Scanned PDF",
        source_type="book",
        course=None,
        chapter=None,
        language="en",
        ocr="auto",
        config=config,
        logger=logging.getLogger("test"),
    )

    assert manifest.extraction_summary.pages_total == 1
    assert manifest.extraction_summary.pages_ocr == 1
