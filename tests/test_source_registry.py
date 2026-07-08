from datetime import datetime, timezone

from scribebase.models import SourceManifest
from scribebase.source_registry import generate_source_id, read_manifest, write_manifest


def test_source_id_generation_is_stable_shape() -> None:
    source_id = generate_source_id(
        "Cognitive Psychology", datetime(2026, 7, 7, tzinfo=timezone.utc)
    )
    assert source_id.startswith("cognitive_psychology_2026_")
    assert len(source_id.rsplit("_", 1)[-1]) == 6


def test_manifest_round_trip(tmp_path) -> None:
    root = tmp_path / "source"
    (root / "metadata").mkdir(parents=True)
    manifest = SourceManifest(
        source_id="src1",
        title="Title",
        original_path="/tmp/source.pdf",
        data_dir=str(root),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    path = write_manifest(manifest)
    loaded = read_manifest(path)
    assert loaded.source_id == "src1"
    assert loaded.title == "Title"
    assert loaded.tags == []
    assert loaded.origin is None
