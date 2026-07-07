from __future__ import annotations

from pathlib import Path


def extract_page_text(pdf_path: Path, page_index: int) -> str:
    import fitz

    doc = fitz.open(pdf_path)
    try:
        return doc[page_index].get_text("text") or ""
    finally:
        doc.close()


def extract_page_markdown(pdf_path: Path, page_index: int) -> tuple[str, str]:
    """Return (markdown, method). Prefer PyMuPDF4LLM, fallback to PyMuPDF text."""
    try:
        import pymupdf4llm

        md = pymupdf4llm.to_markdown(str(pdf_path), pages=[page_index])
        if md and md.strip():
            return md, "pymupdf4llm"
    except Exception:
        pass
    return extract_page_text(pdf_path, page_index), "pymupdf"


def pdf_page_count(pdf_path: Path) -> int:
    import fitz

    doc = fitz.open(pdf_path)
    try:
        return doc.page_count
    finally:
        doc.close()


def page_has_images(pdf_path: Path, page_index: int) -> bool:
    import fitz

    doc = fitz.open(pdf_path)
    try:
        return bool(doc[page_index].get_images(full=True))
    finally:
        doc.close()
