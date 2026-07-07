from __future__ import annotations

from pathlib import Path


def render_pdf_page(pdf_path: Path, page_index: int, output_path: Path, dpi: int = 300) -> Path:
    import fitz

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_index]
        pix = page.get_pixmap(dpi=dpi)
        pix.save(output_path)
        return output_path
    finally:
        doc.close()
