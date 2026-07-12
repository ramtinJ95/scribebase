from __future__ import annotations

import json
import hashlib
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from .models import SourceManifest, normalize_tags
from .paths import source_dir, source_subdirs


class DuplicateSourceError(ValueError):
    def __init__(self, source_id: str, identity_key: str):
        self.source_id = source_id
        self.identity_key = identity_key
        super().__init__(f"Source already exists: {source_id} ({identity_key})")


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
    tags: list[str] | str | None = None,
    origin: str | None = None,
    publisher: str | None = None,
    author: str | None = None,
    created_at_source: datetime | str | None = None,
    updated_at_source: datetime | str | None = None,
    retrieved_at: datetime | str | None = None,
    url: str | None = None,
    canonical_url: str | None = None,
    external_id: str | None = None,
    collection: str | None = None,
    summary: str | None = None,
    source_id: str | None = None,
    identity_key: str | None = None,
    content_sha256: str | None = None,
) -> SourceManifest:
    now = datetime.now(timezone.utc)
    source_id = source_id or generate_source_id(title, now)
    paths = source_subdirs(data_dir, source_id)
    original_path = copy_original(input_path, paths["original"])
    manifest = SourceManifest(
        source_id=source_id,
        title=title,
        source_type=source_type,  # type: ignore[arg-type]
        course=course,
        chapter=chapter,
        language=language,  # type: ignore[arg-type]
        identity_key=identity_key,
        content_sha256=content_sha256,
        tags=normalize_tags(tags),
        origin=origin,
        publisher=publisher,
        author=author,
        created_at_source=created_at_source,
        updated_at_source=updated_at_source,
        retrieved_at=retrieved_at,
        url=url,
        canonical_url=canonical_url,
        external_id=external_id,
        collection=collection,
        summary=summary,
        original_path=str(original_path),
        data_dir=str(paths["root"]),
        created_at=now,
        updated_at=now,
    )
    write_manifest(manifest)
    return manifest


def prepare_source_identity(
    data_dir: Path,
    input_path: Path,
    *,
    origin: str | None,
    canonical_url: str | None,
    url: str | None,
    external_id: str | None,
    source_id: str | None = None,
    duplicate_policy: str = "reject",
) -> tuple[str, str]:
    if duplicate_policy not in {"reject", "create"}:
        raise ValueError("duplicate_policy must be reject or create")
    content_sha256 = hash_source(input_path)
    identity_key = source_identity_key(
        content_sha256=content_sha256,
        origin=origin,
        canonical_url=canonical_url,
        url=url,
        external_id=external_id,
    )
    duplicate = find_source_by_identity(data_dir, identity_key)
    if duplicate is not None and duplicate.source_id != source_id and duplicate_policy == "reject":
        raise DuplicateSourceError(duplicate.source_id, identity_key)
    return identity_key, content_sha256


def source_identity_key(
    *,
    content_sha256: str,
    origin: str | None,
    canonical_url: str | None,
    url: str | None,
    external_id: str | None,
) -> str:
    if external_id:
        return f"external:{(origin or '').strip().lower()}:{external_id.strip()}"
    if source_url := canonical_url or url:
        return f"url:{normalize_url(source_url)}"
    return f"sha256:{content_sha256}"


def find_source_by_identity(data_dir: Path, identity_key: str) -> SourceManifest | None:
    for manifest in list_manifests(data_dir):
        existing_key = manifest.identity_key
        if existing_key is None:
            existing_key = _backfill_identity(manifest)
        if existing_key == identity_key:
            return manifest
    return None


def hash_source(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        _hash_file(path, digest)
        return digest.hexdigest()
    if not path.is_dir():
        raise FileNotFoundError(f"Cannot hash missing source: {path}")
    for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        _hash_file(child, digest)
        digest.update(b"\0")
    return digest.hexdigest()


def normalize_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()
    port = f":{parsed.port}" if parsed.port else ""
    netloc = f"{hostname}{port}"
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((scheme, netloc, path, parsed.query, ""))


def _backfill_identity(manifest: SourceManifest) -> str | None:
    original = Path(manifest.original_path)
    if not original.exists():
        return None
    content_sha256 = manifest.content_sha256 or hash_source(original)
    identity_key = source_identity_key(
        content_sha256=content_sha256,
        origin=manifest.origin,
        canonical_url=manifest.canonical_url,
        url=manifest.url,
        external_id=manifest.external_id,
    )
    manifest.content_sha256 = content_sha256
    manifest.identity_key = identity_key
    manifest.updated_at = datetime.now(timezone.utc)
    write_manifest(manifest)
    return identity_key


def _hash_file(path: Path, digest) -> None:  # noqa: ANN001
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)


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
