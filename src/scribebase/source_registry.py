from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .models import SourceManifest
from .paths import source_dir, source_subdirs


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_") or "source"


def generate_source_id(title: str, created_at: datetime | None = None) -> str:
    import hashlib

    created_at = created_at or datetime.now(timezone.utc)
    digest = hashlib.sha1(f"{title}:{created_at.isoformat()}".encode()).hexdigest()[:6]
    year = created_at.year
    return f"{slugify(title)}_{year}_{digest}"


def manifest_path(root: Path) -> Path:
    return root / "metadata" / "manifest.json"


def write_manifest(manifest: SourceManifest) -> Path:
    path = manifest_path(Path(manifest.data_dir))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2))
    return path


def read_manifest(path_or_source_dir: Path) -> SourceManifest:
    path = path_or_source_dir
    if path.is_dir():
        path = manifest_path(path)
    return SourceManifest.model_validate_json(path.read_text())


def list_manifests(data_dir: Path) -> list[SourceManifest]:
    sources = data_dir / "sources"
    if not sources.exists():
        return []
    manifests: list[SourceManifest] = []
    for path in sorted(sources.glob("*/metadata/manifest.json")):
        manifests.append(read_manifest(path))
    return manifests


def find_source(data_dir: Path, source_id: str) -> SourceManifest:
    root = source_dir(data_dir, source_id)
    if not root.exists():
        raise FileNotFoundError(f"Source not found: {source_id}")
    return read_manifest(root)


def create_manifest(
    data_dir: Path,
    input_path: Path,
    title: str,
    source_type: str,
    course: str | None,
    chapter: str | None,
    language: str,
) -> SourceManifest:
    now = datetime.now(timezone.utc)
    source_id = generate_source_id(title, now)
    paths = source_subdirs(data_dir, source_id)
    original_path = copy_original(input_path, paths["original"])
    manifest = SourceManifest(
        source_id=source_id,
        title=title,
        source_type=source_type,  # type: ignore[arg-type]
        course=course,
        chapter=chapter,
        language=language,  # type: ignore[arg-type]
        original_path=str(original_path),
        data_dir=str(paths["root"]),
        created_at=now,
        updated_at=now,
    )
    write_manifest(manifest)
    return manifest


def copy_original(input_path: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    if input_path.is_dir():
        dest = dest_dir / input_path.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(input_path, dest)
        return dest.resolve()
    dest = dest_dir / input_path.name
    shutil.copy2(input_path, dest)
    return dest.resolve()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
