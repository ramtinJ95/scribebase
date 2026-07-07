from __future__ import annotations

from datetime import datetime
from pathlib import Path

from scribebase.models import SearchResult
from scribebase.source_registry import slugify


def build_context_pack(question: str, results: list[SearchResult], task: str = "answer") -> str:
    lines = [
        "# Context Pack",
        "",
        "User question:",
        question,
        "",
        "Instructions:",
        "Use only the provided context. Cite sources as [Title, p. 87]. "
        "If the answer is not in the context, say so.",
        "",
        f"Task: {task}",
        "",
    ]
    for i, result in enumerate(results, start=1):
        chunk = result.chunk
        pages = _pages(chunk.page_start, chunk.page_end)
        lines.extend(
            [
                f"## Source {i}",
                f"Title: {chunk.title}",
                f"Chapter: {chunk.chapter or ''}",
                f"Section: {chunk.section or ''}",
                f"Pages: {pages}",
                f"Chunk ID: {chunk.chunk_id}",
                "",
                chunk.text.strip(),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def save_context_pack(output_dir: Path, question: str, content: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{stamp}_{slugify(question)[:60]}.md"
    path.write_text(content)
    return path


def _pages(start: int | None, end: int | None) -> str:
    if start is None:
        return "unknown"
    if end is None or end == start:
        return str(start)
    return f"{start}–{end}"
