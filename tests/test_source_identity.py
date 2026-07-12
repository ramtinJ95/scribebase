import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scribebase.config import default_config
from scribebase.extraction import extract_source
from scribebase.models import SourceManifest
from scribebase.source_registry import (
    DuplicateSourceError,
    backfill_source_identities,
    hash_source,
    normalize_url,
    prepare_source_identity,
    read_manifest,
    write_manifest,
    list_manifests,
)


def _extract(path, title, config, **metadata):  # noqa: ANN001, ANN202
    return extract_source(
        path,
        title,
        "notes",
        None,
        None,
        "en",
        "auto",
        config,
        logging.getLogger("test"),
        **metadata,
    )


def test_duplicate_content_is_rejected_by_default(tmp_path) -> None:
    config = default_config()
    config.data_dir = tmp_path / "data"
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("same content")
    second.write_text("same content")

    manifest = _extract(first, "First", config)

    with pytest.raises(DuplicateSourceError) as error:
        _extract(second, "Second", config)

    assert error.value.source_id == manifest.source_id
    assert manifest.identity_key == f"sha256:{manifest.content_sha256}"


def test_create_policy_allows_explicit_duplicate_copy(tmp_path) -> None:
    config = default_config()
    config.data_dir = tmp_path / "data"
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("same content")
    second.write_text("same content")

    original = _extract(first, "First", config)
    copy = _extract(second, "Second", config, duplicate_policy="create")

    assert copy.source_id != original.source_id
    assert copy.identity_key == original.identity_key
    assert copy.content_sha256 == original.content_sha256


def test_canonical_url_identity_precedes_changed_content(tmp_path) -> None:
    config = default_config()
    config.data_dir = tmp_path / "data"
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("version one")
    second.write_text("version two")

    manifest = _extract(first, "First", config, canonical_url="HTTPS://Example.COM/article/#old")

    with pytest.raises(DuplicateSourceError) as error:
        _extract(second, "Second", config, canonical_url="https://example.com/article")

    assert error.value.source_id == manifest.source_id
    assert manifest.identity_key == "url:https://example.com/article"


def test_external_identity_precedes_url_and_content(tmp_path) -> None:
    config = default_config()
    config.data_dir = tmp_path / "data"
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("one")
    second.write_text("two")

    manifest = _extract(
        first,
        "First",
        config,
        origin="Example",
        external_id="item-1",
        canonical_url="https://example.com/old",
    )

    with pytest.raises(DuplicateSourceError) as error:
        _extract(
            second,
            "Second",
            config,
            origin="example",
            external_id="item-1",
            canonical_url="https://example.com/new",
        )

    assert error.value.source_id == manifest.source_id
    assert manifest.identity_key == "external:example:item-1"


def test_legacy_manifest_identity_is_backfilled(tmp_path) -> None:
    data_dir = tmp_path / "data"
    original = tmp_path / "legacy.txt"
    candidate = tmp_path / "candidate.txt"
    original.write_text("legacy content")
    candidate.write_text("legacy content")
    root = data_dir / "sources" / "legacy"
    now = datetime.now(timezone.utc)
    write_manifest(
        SourceManifest(
            source_id="legacy",
            title="Legacy",
            source_type="notes",
            original_path=str(original),
            data_dir=str(root),
            created_at=now,
            updated_at=now,
        )
    )

    assert backfill_source_identities(data_dir) == 1

    with pytest.raises(DuplicateSourceError):
        prepare_source_identity(
            data_dir,
            candidate,
            origin=None,
            canonical_url=None,
            url=None,
            external_id=None,
        )

    updated = read_manifest(root)
    assert updated.content_sha256 == hash_source(original)
    assert updated.identity_key == f"sha256:{updated.content_sha256}"


def test_directory_hash_includes_relative_paths(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "a.txt").write_text("content")
    (second / "b.txt").write_text("content")

    assert hash_source(first) != hash_source(second)


def test_url_normalization_removes_fragment_and_trailing_slash() -> None:
    assert normalize_url("HTTPS://Example.COM/path/#fragment") == "https://example.com/path"


def test_url_normalization_removes_default_ports() -> None:
    assert normalize_url("https://example.com:443/path") == "https://example.com/path"
    assert normalize_url("http://example.com:80/path") == "http://example.com/path"


def test_external_id_requires_origin(tmp_path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("content")

    with pytest.raises(ValueError, match="origin is required"):
        prepare_source_identity(
            tmp_path / "data",
            source,
            origin=None,
            canonical_url=None,
            url=None,
            external_id="common-id",
        )


def test_unsafe_source_id_is_rejected_before_publication(tmp_path) -> None:
    config = default_config()
    config.data_dir = tmp_path / "data"
    source = tmp_path / "source.txt"
    source.write_text("content")

    with pytest.raises(ValueError, match="Invalid source id"):
        _extract(source, "Source", config, source_id="../outside")

    assert not (tmp_path / "outside").exists()


def test_failed_extraction_does_not_publish_manifest(tmp_path) -> None:
    config = default_config()
    config.data_dir = tmp_path / "data"
    source = tmp_path / "source.bin"
    source.write_bytes(b"unsupported")

    with pytest.raises(ValueError, match="Unsupported input type"):
        _extract(source, "Source", config)

    assert list_manifests(config.data_dir) == []


def test_same_id_recovery_replaces_stale_generated_files(tmp_path) -> None:
    config = default_config()
    config.data_dir = tmp_path / "data"
    source = tmp_path / "source.txt"
    source.write_text("content")
    manifest = _extract(source, "Source", config, source_id="stable-source")
    root = Path(manifest.data_dir)
    stale = root / "markdown" / "stale.md"
    stale.write_text("stale")
    (root / "markdown" / "document.md").unlink()

    recovered = _extract(source, "Source", config, source_id="stable-source")

    assert recovered.source_id == "stable-source"
    assert not stale.exists()
    assert (root / "markdown" / "document.md").exists()


def test_concurrent_default_ingestion_publishes_one_identity(tmp_path) -> None:
    config = default_config()
    config.data_dir = tmp_path / "data"
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("same")
    second.write_text("same")

    def ingest(args):  # noqa: ANN001, ANN202
        path, title = args
        try:
            return _extract(path, title, config).source_id
        except DuplicateSourceError:
            return "duplicate"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(ingest, [(first, "First"), (second, "Second")]))

    assert results.count("duplicate") == 1
    assert len(list_manifests(config.data_dir)) == 1
