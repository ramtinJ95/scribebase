from __future__ import annotations

import fcntl
import json
import os
import shutil
from pathlib import Path
from typing import BinaryIO, Callable, Iterable
from uuid import uuid4


def sync_directory(path: Path) -> None:
    """Persist directory-entry changes made before this call."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sync_file(path: Path) -> None:
    with path.open("rb") as source:
        sync_file_descriptor(source.fileno())


def sync_file_descriptor(descriptor: int) -> None:
    full_sync = getattr(fcntl, "F_FULLFSYNC", None)
    if full_sync is not None:
        fcntl.fcntl(descriptor, full_sync)
    else:
        os.fsync(descriptor)


def sync_tree(root: Path) -> None:
    """Persist every file and directory in a completed staging tree."""
    for path in sorted(root.rglob("*")):
        if path.is_file():
            sync_file(path)
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        sync_directory(path)
    sync_directory(root)


def durable_replace(source: Path, destination: Path) -> None:
    """Atomically replace a same-filesystem path and persist the rename."""
    source_parent = source.parent
    destination_parent = destination.parent
    os.replace(source, destination)
    sync_directory(destination_parent)
    if source_parent != destination_parent:
        sync_directory(source_parent)


def durable_unlink(path: Path, *, missing_ok: bool = False) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        if missing_ok:
            return
        raise
    sync_directory(path.parent)


def durable_rmtree(path: Path) -> None:
    if not path.exists():
        return
    parent = path.parent
    shutil.rmtree(path)
    sync_directory(parent)


def atomic_write(path: Path, writer: Callable[[BinaryIO], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as output:
            writer(output)
            output.flush()
            sync_file_descriptor(output.fileno())
        durable_replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_bytes(path: Path, content: bytes) -> None:
    atomic_write(path, lambda output: output.write(content))


def atomic_write_text(path: Path, content: str) -> None:
    atomic_write_bytes(path, content.encode())


def atomic_write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    def write_rows(output: BinaryIO) -> None:
        for row in rows:
            output.write((json.dumps(row, default=str, ensure_ascii=False) + "\n").encode())

    atomic_write(path, write_rows)


def durable_copy(source: Path, destination: Path) -> None:
    def copy(output: BinaryIO) -> None:
        with source.open("rb") as input_file:
            shutil.copyfileobj(input_file, output)

    atomic_write(destination, copy)
    shutil.copystat(source, destination)
    sync_file(destination)
    sync_directory(destination.parent)
