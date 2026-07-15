import pytest

from scribebase.durable_fs import atomic_write, atomic_write_text


def test_atomic_write_syncs_file_and_parent_directory(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    calls = []
    real_fsync = __import__("os").fsync

    def record_fsync(descriptor: int) -> None:
        calls.append(descriptor)
        real_fsync(descriptor)

    monkeypatch.setattr("scribebase.durable_fs.os.fsync", record_fsync)

    path = tmp_path / "state.json"
    atomic_write_text(path, '{"durable": true}')

    assert path.read_text() == '{"durable": true}'
    assert len(calls) >= 2


def test_failed_atomic_write_keeps_previous_complete_file(tmp_path) -> None:
    path = tmp_path / "state.json"
    path.write_text("old")

    def fail_after_partial_write(output) -> None:  # noqa: ANN001
        output.write(b"partial")
        raise OSError("write interrupted")

    with pytest.raises(OSError, match="write interrupted"):
        atomic_write(path, fail_after_partial_write)

    assert path.read_text() == "old"
    assert not list(tmp_path.glob("*.tmp"))
