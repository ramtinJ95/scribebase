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

    def extract_page_markdown(
        self,
        page_index: int,
        fallback_text: str,
    ) -> tuple[str, str, str | None]:
        """Return Markdown, method, and an optional visible fallback warning."""
        try:
            import pymupdf4llm

            markdown = pymupdf4llm.to_markdown(
                self.document,
                pages=[page_index],
                use_ocr=False,
                force_text=True,
            )
            if markdown and markdown.strip():
                return markdown, "pymupdf4llm", None
            return fallback_text, "pymupdf", "pymupdf4llm_empty"
        except Exception as exc:
            warning = f"pymupdf4llm_failed:{exc.__class__.__name__}"
            return fallback_text, "pymupdf", warning

    def render_page(self, page_index: int, output_path: Path, dpi: int = 300) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pixmap = self.document[page_index].get_pixmap(dpi=dpi)
        pixmap.save(output_path)
        return output_path
