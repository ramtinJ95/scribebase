from __future__ import annotations

from pathlib import Path


def ensure_data_layout(data_dir: Path) -> None:
    for rel in [
        "sources",
        "outputs/context_packs",
        "outputs/answers",
        "outputs/quizzes",
        "logs",
    ]:
        (data_dir / rel).mkdir(parents=True, exist_ok=True)


def source_dir(data_dir: Path, source_id: str) -> Path:
    return data_dir / "sources" / source_id


def source_subdirs(data_dir: Path, source_id: str) -> dict[str, Path]:
    root = source_dir(data_dir, source_id)
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


def chapter_file_name(chapter: str) -> str:
    cleaned = chapter.strip().replace(" ", "_").replace("/", "_")
    if cleaned.isdigit():
        cleaned = f"{int(cleaned):02d}"
    if not cleaned.lower().startswith("chapter_"):
        cleaned = f"chapter_{cleaned}"
    return f"{cleaned}.md"
