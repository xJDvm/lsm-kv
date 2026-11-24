"""Tests for DB: basic operations, flushing, recovery and crash simulation."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from lsmkv import DB, ClosedError, CorruptionError
from lsmkv.manifest import Manifest


def kv(i: int) -> tuple[bytes, bytes]:
    return f"key{i:05d}".encode(), f"value-{i}".encode() * 4


def names(path: Path) -> set[str]:
    return {p.name for p in path.iterdir()}


def test_put_get_overwrite_delete(tmp_path: Path) -> None:
    with DB.open(tmp_path / "db") as db:
        assert db.get(b"a") is None
        db.put(b"a", b"1")
        assert db.get(b"a") == b"1"
        db.put(b"a", b"2")
        assert db.get(b"a") == b"2"
        db.delete(b"a")
        assert db.get(b"a") is None
        db.delete(b"never-existed")
        assert db.get(b"never-existed") is None


def test_empty_value_is_distinct_from_deleted(tmp_path: Path) -> None:
    with DB.open(tmp_path) as db:
        db.put(b"k", b"")
        assert db.get(b"k") == b""
        db.delete(b"k")
        assert db.get(b"k") is None
        db.put(b"k", b"")
        db.flush()
        assert db.get(b"k") == b""


def test_creates_nested_directory(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b"
    with DB.open(target) as db:
        db.put(b"k", b"v")
    assert target.is_dir()


def test_invalid_arguments_are_rejected(tmp_path: Path) -> None:
    with DB.open(tmp_path) as db:
        with pytest.raises(TypeError):
            db.put("str", b"v")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            db.put(b"k", "str")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            db.put(b"", b"v")
        with pytest.raises(ValueError):
            db.delete(b"")
        with pytest.raises(TypeError):
            db.get("k")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            db.get(b"")
        assert db.memtable_entries == 0


def test_reopen_after_close_recovers_from_wal_only(tmp_path: Path) -> None:
    with DB.open(tmp_path) as db:
        db.put(b"a", b"1")
        db.put(b"b", b"2")
        db.delete(b"a")
        assert db.table_ids == ()
    with DB.open(tmp_path) as db:
        assert db.table_ids == ()
        assert db.get(b"a") is None
        assert db.get(b"b") == b"2"
        assert db.memtable_entries == 2


def test_reopen_after_flush(tmp_path: Path) -> None:
    with DB.open(tmp_path) as db:
        db.put(b"a", b"1")
        db.put(b"b", b"2")
        db.flush()
        assert db.table_ids == (1,)
        assert db.memtable_entries == 0
        assert (tmp_path / "wal.log").stat().st_size == 0
    with DB.open(tmp_path) as db:
        assert db.table_ids == (1,)
        assert db.get(b"a") == b"1"
        assert db.get(b"b") == b"2"
        assert db.memtable_entries == 0


def test_flush_of_empty_memtable_is_a_noop(tmp_path: Path) -> None:
    with DB.open(tmp_path) as db:
        db.flush()
        assert db.table_ids == ()
        db.put(b"a", b"1")
        db.flush()
        db.flush()
        assert list(db.table_ids) == [1]
    assert names(tmp_path) == {"wal.log", "MANIFEST", "000001.sst"}


def test_small_memtable_forces_several_flushes(tmp_path: Path) -> None:
    expected = dict(kv(i) for i in range(300))
    with DB.open(tmp_path, memtable_bytes=1024, block_size=256, compaction_trigger=10**6) as db:
        for key, value in expected.items():
            db.put(key, value)
        assert len(db.table_ids) > 3
        assert db.table_ids == tuple(sorted(db.table_ids, reverse=True))
        for key, value in expected.items():
            assert db.get(key) == value
        table_count = len(db.table_ids)
    with DB.open(tmp_path, memtable_bytes=1024) as db:
        assert len(db.table_ids) == table_count
        for key, value in expected.items():
            assert db.get(key) == value
        assert db.get(b"absent") is None


def test_newest_version_wins_across_tables(tmp_path: Path) -> None:
    with DB.open(tmp_path) as db:
        db.put(b"k", b"old")
        db.flush()
        db.put(b"k", b"mid")
        db.flush()
        db.put(b"k", b"new")
        db.flush()
        assert db.table_ids == (3, 2, 1)
        assert db.get(b"k") == b"new"
        db.put(b"k", b"memtable")
        assert db.get(b"k") == b"memtable"
    with DB.open(tmp_path) as db:
        assert db.get(b"k") == b"memtable"


def test_tombstone_in_newer_table_hides_older_value(tmp_path: Path) -> None:
    with DB.open(tmp_path) as db:
        db.put(b"k", b"v")
        db.put(b"other", b"x")
        db.flush()
        db.delete(b"k")
        db.flush()
        assert db.get(b"k") is None
        assert db.get(b"other") == b"x"
        db.put(b"k", b"again")
        assert db.get(b"k") == b"again"
    with DB.open(tmp_path) as db:
        assert db.get(b"k") == b"again"
        db.delete(b"k")
    with DB.open(tmp_path) as db:
        assert db.get(b"k") is None


def test_tombstone_in_memtable_hides_table_value(tmp_path: Path) -> None:
    with DB.open(tmp_path) as db:
        db.put(b"k", b"v")
        db.flush()
        db.delete(b"k")
        assert db.get(b"k") is None


def test_orphan_files_are_removed_on_open(tmp_path: Path) -> None:
    with DB.open(tmp_path) as db:
        db.put(b"a", b"1")
        db.flush()
    (tmp_path / "000002.sst").write_bytes(b"orphan from a crashed flush")
    (tmp_path / "000009.sst").write_bytes(b"")
    (tmp_path / "000003.sst.tmp").write_bytes(b"partial")
    (tmp_path / "MANIFEST.tmp").write_bytes(b"partial")
    (tmp_path / "notes.txt").write_bytes(b"not ours")
    with DB.open(tmp_path) as db:
        assert db.get(b"a") == b"1"
    assert names(tmp_path) == {"wal.log", "MANIFEST", "000001.sst", "notes.txt"}


def test_orphan_does_not_clash_with_next_flush(tmp_path: Path) -> None:
    (tmp_path / "000001.sst").write_bytes(b"garbage")
    with DB.open(tmp_path) as db:
        db.put(b"a", b"1")
        db.flush()
        assert db.get(b"a") == b"1"


def test_corrupt_manifest_is_reported(tmp_path: Path) -> None:
    with DB.open(tmp_path) as db:
        db.put(b"a", b"1")
        db.flush()
    (tmp_path / "MANIFEST").write_bytes(b"{broken")
    with pytest.raises(CorruptionError):
        DB.open(tmp_path)


def test_missing_table_is_reported_and_nothing_leaks(tmp_path: Path) -> None:
    with DB.open(tmp_path) as db:
        db.put(b"a", b"1")
        db.flush()
        db.put(b"b", b"2")
        db.flush()
    (tmp_path / "000001.sst").unlink()
    with pytest.raises(CorruptionError, match="missing"):
        DB.open(tmp_path)


def test_corrupt_table_is_reported(tmp_path: Path) -> None:
    with DB.open(tmp_path) as db:
        db.put(b"a", b"1")
        db.flush()
    (tmp_path / "000001.sst").write_bytes(b"short")
    with pytest.raises(CorruptionError):
        DB.open(tmp_path)


def test_torn_wal_tail_is_recovered_and_repaired(tmp_path: Path) -> None:
    with DB.open(tmp_path) as db:
        db.put(b"a", b"1")
        db.put(b"b", b"2")
    wal = tmp_path / "wal.log"
    intact = wal.stat().st_size
    with open(wal, "ab") as f:
        f.write(b"\x99\x01\x02torn")
    with DB.open(tmp_path) as db:
        assert db.get(b"a") == b"1"
        assert db.get(b"b") == b"2"
        assert wal.stat().st_size == intact
        db.put(b"c", b"3")
    with DB.open(tmp_path) as db:
        assert db.get(b"c") == b"3"
        assert db.get(b"a") == b"1"


def test_every_wal_truncation_recovers_a_prefix(tmp_path: Path) -> None:
    source = tmp_path / "source"
    with DB.open(source) as db:
        for i in range(5):
            db.put(*kv(i))
    wal = (source / "wal.log").read_bytes()
    for cut in range(len(wal) + 1):
        target = tmp_path / f"cut{cut}"
        target.mkdir()
        (target / "wal.log").write_bytes(wal[:cut])
        with DB.open(target) as db:
            present = [db.get(kv(i)[0]) is not None for i in range(5)]
        assert present == sorted(present, reverse=True)


def test_crash_without_close_loses_nothing(tmp_path: Path) -> None:
    live = tmp_path / "live"
    db = DB.open(live, memtable_bytes=1024)
    expected = dict(kv(i) for i in range(100))
    for key, value in expected.items():
        db.put(key, value)
    db.delete(kv(7)[0])
    del expected[kv(7)[0]]
    crashed = tmp_path / "crashed"
    shutil.copytree(live, crashed)
    db.close()
    with DB.open(crashed, memtable_bytes=1024) as copy:
        for key, value in expected.items():
            assert copy.get(key) == value
        assert copy.get(kv(7)[0]) is None


def test_crash_at_every_put_boundary(tmp_path: Path) -> None:
    live = tmp_path / "live"
    db = DB.open(live, memtable_bytes=600)
    model: dict[bytes, bytes] = {}
    for i in range(40):
        key, value = kv(i % 13)
        db.put(key, value + bytes([i]))
        model[key] = value + bytes([i])
        snapshot = tmp_path / f"snap{i}"
        shutil.copytree(live, snapshot)
        with DB.open(snapshot) as copy:
            for k, v in model.items():
                assert copy.get(k) == v
    db.close()


def test_crash_during_flush_at_manifest_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = DB.open(tmp_path, memtable_bytes=400)
    expected: dict[bytes, bytes] = {}
    for i in range(3):
        key, value = kv(i)
        db.put(key, value)
        expected[key] = value
    assert db.table_ids == ()

    def boom(directory: object, manifest: object) -> None:
        raise OSError("simulated crash")

    monkeypatch.setattr(Manifest, "commit", staticmethod(boom))
    with pytest.raises(OSError, match="simulated crash"):
        for i in range(3, 20):
            key, value = kv(i)
            db.put(key, value)
            expected[key] = value
    # the write that triggered the flush was logged first, so it is durable
    expected[kv(i)[0]] = kv(i)[1]
    with pytest.raises(ClosedError):
        db.get(b"x")
    db.close()
    monkeypatch.undo()

    with DB.open(tmp_path, memtable_bytes=400) as reopened:
        assert reopened.table_ids == ()
        for key, value in expected.items():
            assert reopened.get(key) == value
    assert names(tmp_path) == {"wal.log"}


def test_crash_after_manifest_commit_before_wal_reset(tmp_path: Path) -> None:
    """Replaying a log whose records are already in a table changes nothing."""
    with DB.open(tmp_path) as db:
        db.put(b"a", b"1")
        db.put(b"b", b"2")
        db.delete(b"c")
        wal_before = (tmp_path / "wal.log").read_bytes()
        db.flush()
    (tmp_path / "wal.log").write_bytes(wal_before)
    with DB.open(tmp_path) as db:
        assert db.table_ids == (1,)
        assert db.get(b"a") == b"1"
        assert db.get(b"b") == b"2"
        assert db.get(b"c") is None
        db.put(b"a", b"newer")
        db.flush()
        assert db.get(b"a") == b"newer"


def test_failed_sstable_write_leaves_database_usable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> int:
        raise OSError("disk full")

    with DB.open(tmp_path) as db:
        db.put(b"a", b"1")
        with monkeypatch.context() as patch:
            patch.setattr("lsmkv.db.write_sstable", boom)
            with pytest.raises(OSError, match="disk full"):
                db.flush()
        assert db.get(b"a") == b"1"
        db.flush()
        assert db.table_ids == (1,)


def test_closed_database_raises(tmp_path: Path) -> None:
    db = DB.open(tmp_path)
    db.put(b"a", b"1")
    db.close()
    db.close()
    with pytest.raises(ClosedError):
        db.put(b"a", b"1")
    with pytest.raises(ClosedError):
        db.get(b"a")
    with pytest.raises(ClosedError):
        db.delete(b"a")
    with pytest.raises(ClosedError):
        db.flush()
    with pytest.raises(ClosedError):
        _ = db.table_ids
    with pytest.raises(ClosedError):
        _ = db.memtable_entries


def test_context_manager_closes(tmp_path: Path) -> None:
    with DB.open(tmp_path) as db:
        db.put(b"a", b"1")
    with pytest.raises(ClosedError):
        db.get(b"a")


@pytest.mark.parametrize(
    ("memtable_bytes", "block_size"),
    [(0, 4096), (-5, 4096), (1024, 0), (1024, -1)],
)
def test_invalid_options(tmp_path: Path, memtable_bytes: int, block_size: int) -> None:
    with pytest.raises(ValueError):
        DB.open(tmp_path / "db", memtable_bytes=memtable_bytes, block_size=block_size)
    assert not (tmp_path / "db").exists()


def test_sync_mode_roundtrip(tmp_path: Path) -> None:
    with DB.open(tmp_path, sync=True) as db:
        db.put(b"a", b"1")
    with DB.open(tmp_path, sync=True) as db:
        assert db.get(b"a") == b"1"


def test_large_value_roundtrip_through_flush(tmp_path: Path) -> None:
    value = bytes(range(256)) * 4096
    with DB.open(tmp_path) as db:
        db.put(b"big", value)
        db.flush()
        assert db.get(b"big") == value
