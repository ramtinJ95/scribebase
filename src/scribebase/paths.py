from __future__ import annotations

from pathlib import Path
import re


SOURCE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,199}")


def ensure_data_layout(data_dir: Path) -> None:
    for rel in [
        "sources",
        "logs",
        "uploads",
        "jobs",
    ]:
        (data_dir / rel).mkdir(parents=True, exist_ok=True)


def source_dir(data_dir: Path, source_id: str) -> Path:
    validate_source_id(source_id)
    return data_dir / "sources" / source_id


def source_subdirs(data_dir: Path, source_id: str) -> dict[str, Path]:
    return source_root_subdirs(source_dir(data_dir, source_id))


def source_root_subdirs(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "original": root / "original",
        "pages": root / "pages",
        "markdown": root / "markdown",
        "chapters": root / "markdown" / "chapters",
        "metadata": root / "metadata",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def validate_source_id(source_id: str) -> str:
    if not SOURCE_ID_RE.fullmatch(source_id):
        raise ValueError(f"Invalid source id: {source_id!r}")
    return source_id


def chapter_file_name(chapter: str) -> str:
    cleaned = chapter.strip().replace(" ", "_").replace("/", "_")
    if cleaned.isdigit():
        cleaned = f"{int(cleaned):02d}"
    if not cleaned.lower().startswith("chapter_"):
        cleaned = f"chapter_{cleaned}"
    return f"{cleaned}.md"
