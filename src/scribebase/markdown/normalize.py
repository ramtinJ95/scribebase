from __future__ import annotations

import re


PAGE_MARKER_RE = re.compile(r"<!--\s*page:\s*(\d+)\s*-->")


def normalize_page_markdown(text: str, page_number: int) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    marker = f"<!-- page: {page_number} -->"
    visible = f"\n\n## Page {page_number}\n\n"
    if not text.startswith(marker):
        text = f"{marker}{visible}{text}" if text else f"{marker}{visible}"
    return text.rstrip() + "\n"


def combine_pages(page_markdowns: list[str]) -> str:
    return "\n".join(md.strip() for md in page_markdowns if md is not None).strip() + "\n"


def page_numbers_in_text(text: str) -> list[int]:
    return [int(match.group(1)) for match in PAGE_MARKER_RE.finditer(text)]


def current_page_for_offset(text: str, offset: int) -> int | None:
    current: int | None = None
    for match in PAGE_MARKER_RE.finditer(text):
        if match.start() > offset:
            break
        current = int(match.group(1))
    return current
