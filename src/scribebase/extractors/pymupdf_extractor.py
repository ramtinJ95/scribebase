from __future__ import annotations

from pathlib import Path


class PDFDocument:
    def __init__(self, pdf_path: Path):
        import fitz

        self.pdf_path = pdf_path
        self.document = fitz.open(pdf_path)

    def __enter__(self) -> "PDFDocument":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    @property
    def page_count(self) -> int:
        return self.document.page_count

    def close(self) -> None:
        self.document.close()

    def extract_page_text(self, page_index: int) -> str:
        return self.document[page_index].get_text("text") or ""

    def page_has_images(self, page_index: int) -> bool:
        return bool(self.document[page_index].get_images(full=True))

    def page_has_visual_content(self, page_index: int) -> bool:
        import fitz

        page = self.document[page_index]
        if page.get_drawings():
            return True
        pixmap = page.get_pixmap(matrix=fitz.Matrix(0.25, 0.25), colorspace=fitz.csGRAY)
        samples = pixmap.samples
        if not samples:
            return False
        dark_pixels = sum(value < 245 for value in samples)
        return dark_pixels / len(samples) >= 0.002

    def extract_page_markdown(
        self,
        page_index: int,
        fallback_text: str,
    ) -> tuple[str, str, str | None]:
        """Return Markdown, method, and an optional visible fallback warning."""
        try:
            import pymupdf4llm
        except ImportError as exc:
            raise RuntimeError("PyMuPDF4LLM is unavailable") from exc

        if self.document.is_closed:
            raise RuntimeError("PDF document was unexpectedly closed before layout extraction")
        try:
            markdown = pymupdf4llm.to_markdown(
                self.document,
                pages=[page_index],
                use_ocr=False,
                force_text=True,
            )
            if markdown and markdown.strip():
                return markdown, "pymupdf4llm", None
            return fallback_text, "pymupdf", "pymupdf4llm_empty"
        except (TypeError, AttributeError) as exc:
            raise RuntimeError("Incompatible PyMuPDF4LLM API") from exc
        except Exception as exc:
            warning = f"pymupdf4llm_failed:{exc.__class__.__name__}"
            return fallback_text, "pymupdf", warning
        finally:
            if self.document.is_closed:
                raise RuntimeError("PyMuPDF4LLM closed the caller-owned PDF document")

    def render_page(self, page_index: int, output_path: Path, dpi: int = 300) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pixmap = self.document[page_index].get_pixmap(dpi=dpi)
        pixmap.save(output_path)
        return output_path
