"""Unit tests for the write-ahead log."""

from pathlib import Path
from typing import BinaryIO

import pytest

from lsmkv.errors import ClosedError
from lsmkv.types import MAX_KEY_SIZE, MAX_VALUE_SIZE
from lsmkv.wal import WAL, recover_wal, scan_wal


def test_roundtrip_put_delete_and_empty_value(tmp_path: Path) -> None:
    path = tmp_path / "wal"
    with WAL.open(path) as wal:
        wal.append_put(b"a", b"1")
        wal.append_put(b"b", b"")
        wal.append_delete(b"a")
    scan = scan_wal(path)
    assert scan.entries == [(b"a", b"1"), (b"b", b""), (b"a", None)]
    assert scan.valid_length == path.stat().st_size
    assert not scan.torn


def test_missing_file_is_empty(tmp_path: Path) -> None:
    scan = scan_wal(tmp_path / "nope")
    assert (scan.entries, scan.valid_length, scan.torn) == ([], 0, False)
    assert recover_wal(tmp_path / "nope") == []


def test_reopen_preserves_earlier_records(tmp_path: Path) -> None:
    path = tmp_path / "wal"
    with WAL.open(path) as wal:
        wal.append_put(b"a", b"1")
    with WAL.open(path, sync=True) as wal:
        wal.append_put(b"b", b"2")
        wal.flush()
    assert scan_wal(path).entries == [(b"a", b"1"), (b"b", b"2")]


def test_max_size_key_and_value(tmp_path: Path) -> None:
    path = tmp_path / "wal"
    key, value = b"k" * MAX_KEY_SIZE, b"v" * MAX_VALUE_SIZE
    with WAL.open(path) as wal:
        wal.append_put(key, value)
    assert scan_wal(path).entries == [(key, value)]


def test_reset_truncates_and_allows_more_appends(tmp_path: Path) -> None:
    path = tmp_path / "wal"
    with WAL.open(path) as wal:
        wal.append_put(b"a", b"1")
        wal.reset()
        assert path.stat().st_size == 0
        wal.append_put(b"b", b"2")
    assert scan_wal(path).entries == [(b"b", b"2")]


def test_closed_wal_raises(tmp_path: Path) -> None:
    wal = WAL.open(tmp_path / "wal")
    wal.close()
    wal.close()
    with pytest.raises(ClosedError):
        wal.append_put(b"a", b"1")
    with pytest.raises(ClosedError):
        wal.append_delete(b"a")
    with pytest.raises(ClosedError):
        wal.flush()
    with pytest.raises(ClosedError):
        wal.reset()


def test_invalid_input_rejected_and_not_logged(tmp_path: Path) -> None:
    path = tmp_path / "wal"
    with WAL.open(path) as wal:
        with pytest.raises(ValueError):
            wal.append_put(b"k" * (MAX_KEY_SIZE + 1), b"v")
        with pytest.raises(ValueError):
            wal.append_put(b"", b"v")
        with pytest.raises(ValueError):
            wal.append_put(b"k", b"v" * (MAX_VALUE_SIZE + 1))
        with pytest.raises(ValueError):
            wal.append_delete(b"")
        with pytest.raises(TypeError):
            wal.append_put("k", b"v")  # type: ignore[arg-type]
    assert path.stat().st_size == 0


def test_recover_truncates_torn_tail_and_appends_are_readable(tmp_path: Path) -> None:
    path = tmp_path / "wal"
    with WAL.open(path) as wal:
        wal.append_put(b"a", b"1")
        wal.append_put(b"b", b"2")
    good = path.stat().st_size
    with path.open("ab") as f:
        f.write(b"\x01\x02\x03")
    assert scan_wal(path).torn
    assert recover_wal(path) == [(b"a", b"1"), (b"b", b"2")]
    assert path.stat().st_size == good
    with WAL.open(path) as wal:
        wal.append_delete(b"a")
    scan = scan_wal(path)
    assert scan.entries == [(b"a", b"1"), (b"b", b"2"), (b"a", None)]
    assert not scan.torn


def test_open_repairs_torn_tail_before_appending(tmp_path: Path) -> None:
    path = tmp_path / "wal"
    with WAL.open(path) as wal:
        wal.append_put(b"a", b"1")
    good = path.stat().st_size
    with path.open("ab") as f:
        f.write(b"garbage")
    with WAL.open(path) as wal:
        assert path.stat().st_size == good
        wal.append_put(b"b", b"2")
    scan = scan_wal(path)
    assert scan.entries == [(b"a", b"1"), (b"b", b"2")]
    assert not scan.torn


class _FailingFile:
    """File stub that writes part of a record, then raises ``OSError``."""

    def __init__(self, real: BinaryIO, *, fail_on: str) -> None:
        self._real = real
        self._fail_on = fail_on

    def write(self, data: bytes) -> int:
        if self._fail_on == "write":
            self._real.write(data[:5])
            self._real.flush()
            raise OSError("disk full")
        return self._real.write(data)

    def flush(self) -> None:
        if self._fail_on == "flush":
            raise OSError("flush failed")
        self._real.flush()

    def fileno(self) -> int:
        return self._real.fileno()

    def close(self) -> None:
        self._real.close()


@pytest.mark.parametrize("fail_on", ["write", "flush"])
def test_io_error_fails_closed_and_reopen_repairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_on: str
) -> None:
    path = tmp_path / "wal"
    wal = WAL.open(path)
    wal.append_put(b"a", b"1")
    real = wal._file
    assert real is not None
    monkeypatch.setattr(wal, "_file", _FailingFile(real, fail_on=fail_on))
    with pytest.raises(OSError, match=r"disk full|flush failed"):
        wal.append_put(b"b", b"2")
    with pytest.raises(ClosedError, match="I/O error"):
        wal.append_put(b"c", b"3")
    with pytest.raises(ClosedError, match="I/O error"):
        wal.append_delete(b"c")
    with pytest.raises(ClosedError, match="I/O error"):
        wal.flush()
    with pytest.raises(ClosedError, match="I/O error"):
        wal.reset()
    wal.close()
    assert real.closed
    with WAL.open(path) as reopened:
        reopened.append_put(b"d", b"4")
    scan = scan_wal(path)
    assert scan.entries[0] == (b"a", b"1")
    assert scan.entries[-1] == (b"d", b"4")
    assert not scan.torn


def test_scan_propagates_non_missing_os_errors(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        scan_wal(tmp_path)


def test_huge_declared_length_is_rejected_without_reading(tmp_path: Path) -> None:
    path = tmp_path / "wal"
    path.write_bytes(b"\x00\x00\x00\x00" + b"\xff\xff\xff\xff")
    scan = scan_wal(path)
    assert scan.entries == []
    assert scan.valid_length == 0
    assert scan.torn
