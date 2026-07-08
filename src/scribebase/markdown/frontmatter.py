from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from scribebase.models import SourceMetadataInput

_FRONTMATTER_MARKERS = {"---", "..."}


def read_markdown_with_frontmatter(path: Path) -> tuple[SourceMetadataInput, str]:
    return split_markdown_frontmatter(path.read_text(encoding="utf-8-sig"))


def split_markdown_frontmatter(text: str) -> tuple[SourceMetadataInput, str]:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return SourceMetadataInput(), text

    closing_index: int | None = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() in _FRONTMATTER_MARKERS:
            closing_index = index
            break
    if closing_index is None:
        raise ValueError("Markdown frontmatter is missing a closing --- marker")

    raw_frontmatter = "".join(lines[1:closing_index])
    body = "".join(lines[closing_index + 1 :])
    loaded = yaml.safe_load(raw_frontmatter) if raw_frontmatter.strip() else {}
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ValueError("Markdown frontmatter must be a YAML mapping")
    try:
        metadata = SourceMetadataInput.model_validate(loaded)
    except ValidationError as exc:
        raise ValueError(f"Invalid Markdown frontmatter: {exc}") from exc
    return metadata, body
