import logging
from pathlib import Path
from types import SimpleNamespace

import fitz
import pytest

from scribebase.config import AppConfig, PDFDetectionConfig
from scribebase.extraction import extract_source, read_page_metadata
from scribebase.models import OCRResult


def test_extract_true_text_pdf_without_ocr(tmp_path) -> None:
    pdf = tmp_path / "book.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Working memory has limited capacity. " * 40)
    doc.save(pdf)
    doc.close()

    config = AppConfig(
        data_dir=tmp_path / ".scribebase",
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
        data_dir=tmp_path / ".scribebase",
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
    config = AppConfig(data_dir=tmp_path / ".scribebase")

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


def test_auto_mixed_pdf_ocr_scanned_pages(tmp_path, monkeypatch) -> None:
    image = tmp_path / "scan.png"
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 20, 20), 0)
    pix.clear_with(255)
    pix.save(image)

    pdf = tmp_path / "mixed.pdf"
    doc = fitz.open()
    for _ in range(3):
        page = doc.new_page()
        page.insert_text((72, 72), "This is a true text page. " * 40)
    scanned = doc.new_page()
    scanned.insert_image(fitz.Rect(72, 72, 200, 200), filename=image)
    doc.save(pdf)
    doc.close()

    class FakeOCRProvider:
        name = "fake"
        config = SimpleNamespace(render_dpi=None)

        def ocr_image(self, image_path, output_md_path, metadata):  # noqa: ANN001
            return OCRResult(
                markdown_path=output_md_path,
                text=f"OCR text from scanned page {metadata['page_number']}",
                provider=self.name,
                model="fake-ocr",
            )

    monkeypatch.setattr("scribebase.extraction._ocr_provider", lambda *_: FakeOCRProvider())
    config = AppConfig(data_dir=tmp_path / ".scribebase")

    manifest = extract_source(
        pdf,
        title="Mixed PDF",
        source_type="book",
        course=None,
        chapter=None,
        language="en",
        ocr="auto",
        config=config,
        logger=logging.getLogger("test"),
    )

    root = config.data_dir / "sources" / manifest.source_id
    assert manifest.extraction_summary.pages_total == 4
    assert manifest.extraction_summary.pages_ocr == 1
    assert "OCR text from scanned page 4" in (root / "markdown" / "page_0004.md").read_text()


def test_pdf_extraction_opens_document_once(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    pdf_path = tmp_path / "book.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "True text page " * 80)
    doc.save(pdf_path)
    doc.close()
    real_open = fitz.open
    opened = 0

    def counted_open(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal opened
        opened += 1
        return real_open(*args, **kwargs)

    monkeypatch.setattr(fitz, "open", counted_open)
    config = AppConfig(data_dir=tmp_path / "data")

    extract_source(
        pdf_path,
        "Book",
        "book",
        None,
        None,
        "en",
        "never",
        config,
        logging.getLogger("test"),
    )

    assert opened == 1


def test_pymupdf4llm_failure_is_visible_in_page_metadata(tmp_path, monkeypatch, caplog) -> None:  # noqa: ANN001
    pdf_path = tmp_path / "book.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Fallback text " * 80)
    doc.save(pdf_path)
    doc.close()
    monkeypatch.setattr(
        "pymupdf4llm.to_markdown",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("layout failed")),
    )
    config = AppConfig(data_dir=tmp_path / "data")

    with caplog.at_level(logging.WARNING):
        manifest = extract_source(
            pdf_path,
            "Book",
            "book",
            None,
            None,
            "en",
            "never",
            config,
            logging.getLogger("test"),
        )

    pages = read_page_metadata(Path(manifest.data_dir))
    assert pages[0].extraction_method == "pymupdf"
    assert "pymupdf4llm_failed:RuntimeError" in pages[0].quality_flags
    assert "using PyMuPDF text fallback" in caplog.text
    assert manifest.extraction_summary.pages_extracted_with_pymupdf4llm == 0


def test_auto_ocrs_vector_backed_page_without_embedded_images(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    pdf_path = tmp_path / "vector-scan.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.draw_rect(fitz.Rect(60, 60, 500, 700), color=(0, 0, 0), fill=(0.8, 0.8, 0.8))
    doc.save(pdf_path)
    doc.close()

    class FakeOCRProvider:
        name = "fake"
        config = SimpleNamespace(render_dpi=72)

        def ocr_image(self, image_path, output_md_path, metadata):  # noqa: ANN001
            return OCRResult(
                markdown_path=output_md_path,
                text="OCR text from vector page",
                provider=self.name,
                model="fake-ocr",
            )

    monkeypatch.setattr("scribebase.extraction._ocr_provider", lambda *_: FakeOCRProvider())
    config = AppConfig(data_dir=tmp_path / "data")

    manifest = extract_source(
        pdf_path,
        "Vector Scan",
        "book",
        None,
        None,
        "en",
        "auto",
        config,
        logging.getLogger("test"),
    )

    pages = read_page_metadata(Path(manifest.data_dir))
    assert pages[0].extraction_method == "ocr"
    assert manifest.extraction_summary.pages_ocr == 1


def test_layout_api_incompatibility_fails_fast(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    pdf_path = tmp_path / "book.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Text " * 100)
    doc.save(pdf_path)
    doc.close()
    monkeypatch.setattr(
        "pymupdf4llm.to_markdown",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TypeError("unexpected keyword")),
    )
    config = AppConfig(data_dir=tmp_path / "data")

    with pytest.raises(RuntimeError, match="Incompatible PyMuPDF4LLM API"):
        extract_source(
            pdf_path,
            "Book",
            "book",
            None,
            None,
            "en",
            "never",
            config,
            logging.getLogger("test"),
        )

    assert not list((config.data_dir / "sources").glob("*/metadata/manifest.json"))


def test_multi_page_layout_failure_does_not_close_document(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    pdf_path = tmp_path / "book.pdf"
    doc = fitz.open()
    for _ in range(2):
        page = doc.new_page()
        page.insert_text((72, 72), "Text page " * 100)
    doc.save(pdf_path)
    doc.close()
    calls = 0

    def convert(document, **_kwargs):  # noqa: ANN001, ANN202
        nonlocal calls
        calls += 1
        assert not document.is_closed
        if calls == 1:
            raise RuntimeError("page layout failed")
        return "Second page layout Markdown"

    monkeypatch.setattr("pymupdf4llm.to_markdown", convert)
    config = AppConfig(data_dir=tmp_path / "data")

    manifest = extract_source(
        pdf_path,
        "Book",
        "book",
        None,
        None,
        "en",
        "never",
        config,
        logging.getLogger("test"),
    )

    pages = read_page_metadata(Path(manifest.data_dir))
    assert pages[0].extraction_method == "pymupdf"
    assert pages[1].extraction_method == "pymupdf4llm"
    assert calls == 2


def test_pdf_document_closes_once_when_extraction_raises(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    from scribebase.extractors.pymupdf_extractor import PDFDocument

    pdf_path = tmp_path / "book.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Text " * 100)
    doc.save(pdf_path)
    doc.close()
    closes = 0
    real_close = PDFDocument.close

    def counted_close(document):  # noqa: ANN001
        nonlocal closes
        closes += 1
        real_close(document)

    monkeypatch.setattr(PDFDocument, "close", counted_close)
    monkeypatch.setattr(
        "pymupdf4llm.to_markdown",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TypeError("bad API")),
    )
    config = AppConfig(data_dir=tmp_path / "data")

    with pytest.raises(RuntimeError):
        extract_source(
            pdf_path,
            "Book",
            "book",
            None,
            None,
            "en",
            "never",
            config,
            logging.getLogger("test"),
        )

    assert closes == 1
