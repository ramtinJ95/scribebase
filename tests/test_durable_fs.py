import pytest

from scribebase.durable_fs import atomic_write, atomic_write_text, sync_file_descriptor


def test_atomic_write_syncs_file_and_parent_directory(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    fsync_calls = []
    full_sync_calls = []
    real_fsync = __import__("os").fsync
    real_fcntl = __import__("fcntl").fcntl

    def record_fsync(descriptor: int) -> None:
        fsync_calls.append(descriptor)
        real_fsync(descriptor)

    def record_full_sync(descriptor: int, operation: int):
        full_sync_calls.append((descriptor, operation))
        return real_fcntl(descriptor, operation)

    monkeypatch.setattr("scribebase.durable_fs.os.fsync", record_fsync)
    monkeypatch.setattr("scribebase.durable_fs.fcntl.fcntl", record_full_sync)

    path = tmp_path / "state.json"
    atomic_write_text(path, '{"durable": true}')

    assert path.read_text() == '{"durable": true}'
    assert len(fsync_calls) + len(full_sync_calls) >= 2


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


def test_fullfsync_replaces_redundant_fsync_on_macos(monkeypatch) -> None:  # noqa: ANN001
    full_sync_calls = []
    fsync_calls = []
    monkeypatch.setattr("scribebase.durable_fs.fcntl.F_FULLFSYNC", 51, raising=False)
    monkeypatch.setattr(
        "scribebase.durable_fs.fcntl.fcntl",
        lambda descriptor, operation: full_sync_calls.append((descriptor, operation)),
    )
    monkeypatch.setattr(
        "scribebase.durable_fs.os.fsync", lambda descriptor: fsync_calls.append(descriptor)
    )

    sync_file_descriptor(7)

    assert full_sync_calls == [(7, 51)]
    assert fsync_calls == []


def test_fsync_is_used_when_fullfsync_is_unavailable(monkeypatch) -> None:  # noqa: ANN001
    calls = []
    monkeypatch.delattr("scribebase.durable_fs.fcntl.F_FULLFSYNC", raising=False)
    monkeypatch.setattr("scribebase.durable_fs.os.fsync", lambda descriptor: calls.append(descriptor))

    sync_file_descriptor(7)

    assert calls == [7]
