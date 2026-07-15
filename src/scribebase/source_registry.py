from __future__ import annotations

import fcntl
import hashlib
import json
import re
import shutil
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from .durable_fs import (
    atomic_write_jsonl,
    atomic_write_text,
    durable_copy,
    durable_replace,
    durable_unlink,
)
from .models import SourceManifest, normalize_tags
from .paths import source_dir, source_subdirs, validate_source_id


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
    atomic_write_text(path, manifest.model_dump_json(indent=2))
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
    validate_source_id(source_id)
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
        if not origin or not origin.strip():
            raise ValueError("origin is required when external_id is provided")
        return f"external:{origin.strip().lower()}:{external_id.strip()}"
    if source_url := canonical_url or url:
        return f"url:{normalize_url(source_url)}"
    return f"sha256:{content_sha256}"


def find_source_by_identity(data_dir: Path, identity_key: str) -> SourceManifest | None:
    for manifest in list_manifests(data_dir):
        existing_key = manifest.identity_key or _existing_identity_key(manifest)
        if existing_key == identity_key:
            return manifest
    return None


def reserve_source_identity(
    data_dir: Path,
    identity_key: str,
    *,
    owner_id: str,
    source_id: str,
    duplicate_policy: str,
    owner_type: str = "direct",
) -> bool:
    if duplicate_policy == "create":
        return False
    with source_registry_lock(data_dir):
        duplicate = find_source_by_identity(data_dir, identity_key)
        if duplicate is not None and duplicate.source_id != source_id:
            raise DuplicateSourceError(duplicate.source_id, identity_key)
        path = _identity_reservation_path(data_dir, identity_key)
        if path.exists():
            reservation = _read_identity_reservation(path)
            if reservation is None:
                path = _identity_reservation_path(data_dir, identity_key)
            else:
                if reservation["owner_id"] != owner_id:
                    raise DuplicateSourceError(reservation["source_id"], identity_key)
                return False
        atomic_write_text(
            path,
            json.dumps(
                {
                    "identity_key": identity_key,
                    "owner_id": owner_id,
                    "owner_type": owner_type,
                    "source_id": source_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
        )
        return True


def release_source_identity(data_dir: Path, identity_key: str, owner_id: str) -> None:
    with source_registry_lock(data_dir):
        path = _identity_reservation_path(data_dir, identity_key)
        if not path.exists():
            return
        reservation = _read_identity_reservation(path)
        if reservation is not None and reservation["owner_id"] == owner_id:
            durable_unlink(path)


def refresh_source_identity_reservation(data_dir: Path, identity_key: str, owner_id: str) -> bool:
    with source_registry_lock(data_dir):
        path = _identity_reservation_path(data_dir, identity_key)
        if not path.exists():
            return False
        reservation = _read_identity_reservation(path)
        if reservation is None or reservation["owner_id"] != owner_id:
            return False
        reservation["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_write_text(path, json.dumps(reservation, indent=2))
        return True


def identity_reservation_owned(data_dir: Path, identity_key: str, owner_id: str) -> bool:
    path = _identity_reservation_path(data_dir, identity_key)
    if not path.exists():
        return False
    reservation = _read_identity_reservation(path)
    return reservation is not None and reservation["owner_id"] == owner_id


def _identity_reservation_path(data_dir: Path, identity_key: str) -> Path:
    digest = hashlib.sha256(identity_key.encode()).hexdigest()
    return data_dir / "sources" / ".identity-reservations" / f"{digest}.json"


def reconcile_source_identity_reservations(
    data_dir: Path,
    active_job_ids: set[str],
    *,
    orphan_job_seconds: int,
    direct_seconds: int,
) -> int:
    removed = 0
    now = datetime.now(timezone.utc)
    with source_registry_lock(data_dir):
        directory = data_dir / "sources" / ".identity-reservations"
        for path in directory.glob("*.json"):
            reservation = _read_identity_reservation(path)
            if reservation is None:
                continue
            updated = datetime.fromisoformat(reservation["updated_at"])
            age = (now - updated).total_seconds()
            owner_type = reservation["owner_type"]
            expired_job = (
                owner_type == "job"
                and reservation["owner_id"] not in active_job_ids
                and age >= orphan_job_seconds
            )
            expired_direct = owner_type == "direct" and age >= direct_seconds
            published = find_source_by_identity(data_dir, reservation["identity_key"])
            if (
                expired_job
                or expired_direct
                or (published is not None and published.source_id == reservation["source_id"])
            ):
                path.unlink(missing_ok=True)
                removed += 1
    return removed


def _read_identity_reservation(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text())
        required = {
            "identity_key",
            "owner_id",
            "owner_type",
            "source_id",
            "created_at",
            "updated_at",
        }
        if not required.issubset(payload):
            raise ValueError(f"missing fields: {sorted(required - set(payload))}")
        return payload
    except Exception:
        quarantine = path.with_suffix(f".corrupt.{int(datetime.now().timestamp())}")
        path.replace(quarantine)
        warnings.warn(f"Quarantined corrupt identity reservation: {path}", stacklevel=2)
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
    if parsed.username or parsed.password:
        raise ValueError("Source identity URLs must not contain userinfo")
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()
    if scheme not in {"http", "https"} or not hostname:
        raise ValueError(f"Invalid HTTP(S) source URL: {value!r}")
    port_number = parsed.port
    port = (
        f":{port_number}"
        if port_number is not None
        and not (scheme == "https" and port_number == 443)
        and not (scheme == "http" and port_number == 80)
        else ""
    )
    host_for_netloc = f"[{hostname}]" if ":" in hostname else hostname
    netloc = f"{host_for_netloc}{port}"
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((scheme, netloc, path, parsed.query, ""))


def backfill_source_identities(data_dir: Path) -> int:
    snapshot = list_manifests(data_dir)
    computed: dict[str, tuple[datetime, str, str, str]] = {}
    for manifest in snapshot:
        if manifest.identity_key and manifest.content_sha256:
            continue
        original = Path(manifest.original_path)
        if not original.exists():
            continue
        content_sha256 = manifest.content_sha256 or hash_source(original)
        identity_key = source_identity_key(
            content_sha256=content_sha256,
            origin=manifest.origin,
            canonical_url=manifest.canonical_url,
            url=manifest.url,
            external_id=manifest.external_id if manifest.origin else None,
        )
        computed[manifest.source_id] = (
            manifest.updated_at,
            manifest.original_path,
            identity_key,
            content_sha256,
        )

    with source_registry_lock(data_dir):
        pending: list[tuple[SourceManifest, str, str]] = []
        identities: dict[str, list[str]] = {}
        for manifest in list_manifests(data_dir):
            if manifest.identity_key and manifest.content_sha256:
                identity_key = manifest.identity_key
                content_sha256 = manifest.content_sha256
            else:
                prepared = computed.get(manifest.source_id)
                if prepared is None:
                    if Path(manifest.original_path).exists():
                        raise RuntimeError("Source registry changed during backfill; retry")
                    continue
                prior_updated, prior_original, identity_key, content_sha256 = prepared
                if manifest.updated_at != prior_updated or manifest.original_path != prior_original:
                    raise RuntimeError("Source registry changed during backfill; retry")
                pending.append((manifest, identity_key, content_sha256))
            identities.setdefault(identity_key, []).append(manifest.source_id)
        collisions = {key: ids for key, ids in identities.items() if len(ids) > 1}
        if collisions:
            details = "; ".join(f"{key}: {', '.join(ids)}" for key, ids in collisions.items())
            raise RuntimeError(f"Identity collisions found; no manifests changed: {details}")
        for manifest, identity_key, content_sha256 in pending:
            manifest.content_sha256 = content_sha256
            manifest.identity_key = identity_key
            manifest.updated_at = datetime.now(timezone.utc)
        _install_manifest_batch(data_dir, [manifest for manifest, _, _ in pending])
    return len(pending)


def _existing_identity_key(manifest: SourceManifest) -> str | None:
    if manifest.external_id and manifest.origin:
        return source_identity_key(
            content_sha256=manifest.content_sha256 or "",
            origin=manifest.origin,
            canonical_url=manifest.canonical_url,
            url=manifest.url,
            external_id=manifest.external_id,
        )
    if manifest.canonical_url or manifest.url:
        return source_identity_key(
            content_sha256=manifest.content_sha256 or "",
            origin=manifest.origin,
            canonical_url=manifest.canonical_url,
            url=manifest.url,
            external_id=None,
        )
    if manifest.content_sha256:
        return f"sha256:{manifest.content_sha256}"
    return None


@contextmanager
def source_registry_lock(data_dir: Path):  # noqa: ANN202
    path = data_dir / "sources" / ".registry.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            _recover_manifest_transaction(data_dir)
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _install_manifest_batch(data_dir: Path, manifests: list[SourceManifest]) -> None:
    if not manifests:
        return
    entries = []
    transaction_id = uuid4().hex
    marker = data_dir / "sources" / ".manifest-transaction.json"
    for manifest in manifests:
        live = manifest_path(Path(manifest.data_dir))
        staged = live.with_suffix(f"{live.suffix}.{transaction_id}.stage")
        backup = live.with_suffix(f"{live.suffix}.{transaction_id}.backup")
        entries.append({"live": str(live), "staged": str(staged), "backup": str(backup)})
    atomic_write_text(marker, json.dumps({"entries": entries}, indent=2))
    try:
        for manifest, entry in zip(manifests, entries):
            staged = Path(entry["staged"])
            backup = Path(entry["backup"])
            live = Path(entry["live"])
            atomic_write_text(staged, manifest.model_dump_json(indent=2))
            durable_copy(live, backup)
        for entry in entries:
            durable_replace(Path(entry["staged"]), Path(entry["live"]))
    except Exception:
        _restore_manifest_entries(entries)
        _cleanup_manifest_transaction(marker, entries)
        raise
    else:
        _cleanup_manifest_transaction(marker, entries)


def _recover_manifest_transaction(data_dir: Path) -> None:
    marker = data_dir / "sources" / ".manifest-transaction.json"
    if not marker.exists():
        return
    payload = json.loads(marker.read_text())
    entries = payload["entries"]
    _restore_manifest_entries(entries)
    _cleanup_manifest_transaction(marker, entries)


def _restore_manifest_entries(entries: list[dict]) -> None:
    for entry in entries:
        backup = Path(entry["backup"])
        if backup.exists():
            temporary = Path(entry["live"]).with_suffix(f".restore.{uuid4().hex}.tmp")
            try:
                durable_copy(backup, temporary)
                durable_replace(temporary, Path(entry["live"]))
            finally:
                temporary.unlink(missing_ok=True)


def _cleanup_manifest_transaction(marker: Path, entries: list[dict]) -> None:
    durable_unlink(marker, missing_ok=True)
    for entry in entries:
        durable_unlink(Path(entry["staged"]), missing_ok=True)
        durable_unlink(Path(entry["backup"]), missing_ok=True)


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
    atomic_write_jsonl(path, rows)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
