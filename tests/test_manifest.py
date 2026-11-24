"""Tests for the MANIFEST: round trips, validation and atomic replacement."""

from __future__ import annotations

import errno
import json
import os
import tempfile
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from lsmkv.errors import CorruptionError
from lsmkv.manifest import Manifest, fsync_dir, table_path


def write_raw(directory: Path, content: bytes | str) -> None:
    data = content.encode() if isinstance(content, str) else content
    (directory / "MANIFEST").write_bytes(data)


def doc(**overrides: object) -> str:
    fields: dict[str, object] = {"version": 1, "next_id": 5, "tables": [3, 1]}
    fields.update(overrides)
    return json.dumps(fields)


def test_missing_file_is_a_fresh_database(tmp_path: Path) -> None:
    assert Manifest.load(tmp_path) == Manifest(1, ())


def test_roundtrip(tmp_path: Path) -> None:
    manifest = Manifest(9, (8, 3, 1))
    Manifest.commit(tmp_path, manifest)
    assert Manifest.load(tmp_path) == manifest
    assert not (tmp_path / "MANIFEST.tmp").exists()


@given(
    st.lists(st.integers(1, 10**9), unique=True, max_size=30),
    st.integers(0, 100),
)
def test_roundtrip_property(tables: list[int], slack: int) -> None:
    with tempfile.TemporaryDirectory() as raw:
        next_id = max(tables, default=0) + 1 + slack
        manifest = Manifest(next_id, tuple(tables))
        Manifest.commit(raw, manifest)
        assert Manifest.load(raw) == manifest


def test_commit_replaces_previous(tmp_path: Path) -> None:
    Manifest.commit(tmp_path, Manifest(2, (1,)))
    Manifest.commit(tmp_path, Manifest(3, (2, 1)))
    assert Manifest.load(tmp_path) == Manifest(3, (2, 1))


def test_stale_tmp_file_is_ignored(tmp_path: Path) -> None:
    Manifest.commit(tmp_path, Manifest(2, (1,)))
    (tmp_path / "MANIFEST.tmp").write_bytes(b"\x00 half written")
    assert Manifest.load(tmp_path) == Manifest(2, (1,))
    Manifest.commit(tmp_path, Manifest(3, (2, 1)))
    assert Manifest.load(tmp_path) == Manifest(3, (2, 1))


def test_stale_tmp_without_manifest_is_ignored(tmp_path: Path) -> None:
    (tmp_path / "MANIFEST.tmp").write_text(doc())
    assert Manifest.load(tmp_path) == Manifest(1, ())


def test_valid_document_loads(tmp_path: Path) -> None:
    write_raw(tmp_path, doc())
    assert Manifest.load(tmp_path) == Manifest(5, (3, 1))


def test_exactly_one_mib_is_accepted(tmp_path: Path) -> None:
    base = doc(tables=[])
    write_raw(tmp_path, base + " " * (1024 * 1024 - len(base)))
    assert Manifest.load(tmp_path) == Manifest(5, ())


def test_oversized_file_is_rejected(tmp_path: Path) -> None:
    base = doc(tables=[])
    write_raw(tmp_path, base + " " * (1024 * 1024 + 1 - len(base)))
    with pytest.raises(CorruptionError, match="larger"):
        Manifest.load(tmp_path)


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"not json",
        b"\xff\xfe\x00",
        b"[" * 100_000,
        b"NaN",
        b"[]",
        b"null",
        b'"text"',
        b"{}",
        doc(version=2),
        doc(version=0),
        doc(version="1"),
        doc(version=True),
        doc(version=1.0),
        doc(next_id="5"),
        doc(next_id=True),
        doc(next_id=5.0),
        doc(next_id=None),
        doc(next_id=0),
        doc(next_id=-1),
        doc(next_id=3),
        doc(next_id=2),
        doc(tables="31"),
        doc(tables=None),
        doc(tables={"a": 1}),
        doc(tables=[3, True]),
        doc(tables=[3.0]),
        doc(tables=["3"]),
        doc(tables=[None]),
        doc(tables=[[3]]),
        doc(tables=[0]),
        doc(tables=[-1]),
        doc(tables=[3, 3]),
        doc(tables=[1, 2, 1]),
        '{"version": 1, "next_id": 5}',
        '{"version": 1, "tables": []}',
        '{"next_id": 5, "tables": []}',
        '{"version": 1, "next_id": 5, "tables": [], "extra": 1}',
    ],
)
def test_malformed_manifest_is_corruption(tmp_path: Path, content: bytes | str) -> None:
    write_raw(tmp_path, content)
    with pytest.raises(CorruptionError):
        Manifest.load(tmp_path)


@given(st.binary(max_size=200))
def test_arbitrary_bytes_never_raise_anything_but_corruption(content: bytes) -> None:
    with tempfile.TemporaryDirectory() as raw:
        directory = Path(raw)
        write_raw(directory, content)
        try:
            manifest = Manifest.load(directory)
        except CorruptionError:
            return
        assert manifest.next_id >= 1


@pytest.mark.parametrize(
    "manifest",
    [
        Manifest(0, ()),
        Manifest(2, (2,)),
        Manifest(5, (3, 3)),
        Manifest(5, (0,)),
        Manifest(5, (-1,)),
    ],
)
def test_commit_rejects_invalid_manifest(tmp_path: Path, manifest: Manifest) -> None:
    with pytest.raises(ValueError):
        Manifest.commit(tmp_path, manifest)
    assert not (tmp_path / "MANIFEST").exists()


def test_table_count_cap(tmp_path: Path) -> None:
    at_cap = list(range(10_000, 0, -1))
    write_raw(tmp_path, doc(next_id=10_001, tables=at_cap))
    assert Manifest.load(tmp_path).tables == tuple(at_cap)
    write_raw(tmp_path, doc(next_id=10_002, tables=[10_001, *at_cap]))
    with pytest.raises(CorruptionError, match="too many tables"):
        Manifest.load(tmp_path)


def test_commit_rejects_too_many_tables(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="too many tables"):
        Manifest.commit(tmp_path, Manifest(10_002, tuple(range(10_001, 0, -1))))
    assert not (tmp_path / "MANIFEST").exists()


def test_fsync_dir_propagates_real_io_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_fsync(fd: int) -> None:
        raise OSError(errno.EIO, "input/output error")

    monkeypatch.setattr(os, "fsync", failing_fsync)
    with pytest.raises(OSError) as info:
        fsync_dir(tmp_path)
    assert info.value.errno == errno.EIO


@pytest.mark.parametrize(
    "code", [errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EBADF], ids=str
)
def test_fsync_dir_swallows_unsupported(
    code: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unsupported(fd: int) -> None:
        raise OSError(code, "unsupported")

    monkeypatch.setattr(os, "fsync", unsupported)
    fsync_dir(tmp_path)


def test_fsync_dir_open_failure_propagates_off_windows(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        fsync_dir(tmp_path / "missing")


def test_fsync_dir_tolerates_windows_style_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: object, **kwargs: object) -> int:
        raise PermissionError(errno.EACCES, "access denied")

    with monkeypatch.context() as patch:
        patch.setattr(os, "name", "nt")
        patch.setattr(os, "open", refuse)
        fsync_dir(tmp_path)

    def eio(fd: int) -> None:
        raise OSError(errno.EIO, "input/output error")

    with monkeypatch.context() as patch:
        patch.setattr(os, "name", "nt")
        patch.setattr(os, "fsync", eio)
        fsync_dir(tmp_path)


def test_table_path_is_derived_from_the_integer_id(tmp_path: Path) -> None:
    assert table_path(tmp_path, 42) == tmp_path / "000042.sst"
    assert table_path(tmp_path, 1234567) == tmp_path / "1234567.sst"
    assert table_path(tmp_path, 7).parent == tmp_path
