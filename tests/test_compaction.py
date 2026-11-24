"""Tests for size-tiered compaction: the planner and its use inside DB."""

from __future__ import annotations

import itertools
import random
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from lsmkv import DB, ClosedError
from lsmkv.compaction import pick_tiered_run
from lsmkv.manifest import MANIFEST_NAME, Manifest, table_path
from lsmkv.sstable import SSTableReader

# --- pick_tiered_run -------------------------------------------------------


def is_similar(sizes: list[int], ratio: float) -> bool:
    clamped = [max(s, 1) for s in sizes]
    return max(clamped) <= ratio * min(clamped)


def brute_force_best_length(sizes: list[int], min_tables: int, ratio: float) -> int:
    best = 0
    for start, stop in itertools.combinations(range(len(sizes) + 1), 2):
        if stop - start >= min_tables and is_similar(sizes[start:stop], ratio):
            best = max(best, stop - start)
    return best


def test_planner_basic_cases() -> None:
    assert pick_tiered_run([], min_tables=2) is None
    assert pick_tiered_run([10], min_tables=2) is None
    assert pick_tiered_run([10, 10], min_tables=2) == range(0, 2)
    assert pick_tiered_run([10, 10, 10], min_tables=4) is None
    assert pick_tiered_run([10, 11, 12, 13], min_tables=4) == range(0, 4)
    # Boundary: exactly the ratio is similar, anything above is not.
    assert pick_tiered_run([10, 20], min_tables=2) == range(0, 2)
    assert pick_tiered_run([10, 21], min_tables=2) is None


def test_planner_requires_contiguity() -> None:
    # The two small tables are separated by a big one: they cannot be merged.
    assert pick_tiered_run([10, 1000, 10], min_tables=2) is None
    assert pick_tiered_run([10, 1000, 10, 11], min_tables=2) == range(2, 4)


def test_planner_prefers_longest_then_newest() -> None:
    assert pick_tiered_run([10, 10, 1000, 1000, 1000], min_tables=2) == range(2, 5)
    assert pick_tiered_run([10, 10, 1000, 1000], min_tables=2) == range(0, 2)


def test_planner_treats_zero_as_one() -> None:
    assert pick_tiered_run([0, 0, 2], min_tables=3) == range(0, 3)
    assert pick_tiered_run([0, 3], min_tables=2) is None


def test_planner_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        pick_tiered_run([1, 2], min_tables=1)
    with pytest.raises(ValueError):
        pick_tiered_run([1, 2], min_tables=2, bucket_ratio=0.5)


@given(
    sizes=st.lists(st.integers(min_value=0, max_value=10_000), max_size=12),
    min_tables=st.integers(min_value=2, max_value=5),
    ratio=st.floats(min_value=1.0, max_value=8.0, allow_nan=False),
)
def test_planner_matches_brute_force(sizes: list[int], min_tables: int, ratio: float) -> None:
    run = pick_tiered_run(sizes, min_tables=min_tables, bucket_ratio=ratio)
    expected = brute_force_best_length(sizes, min_tables, ratio)
    if run is None:
        assert expected == 0
        return
    assert run.step == 1
    assert 0 <= run.start < run.stop <= len(sizes)
    assert len(run) >= min_tables
    assert is_similar(sizes[run.start : run.stop], ratio)
    assert len(run) == expected


# --- DB integration --------------------------------------------------------


def key(i: int) -> bytes:
    return f"key{i:05d}".encode()


def live_tombstones(path: Path, db: DB) -> int:
    total = 0
    for table_id in db.table_ids:
        with SSTableReader(table_path(path, table_id)) as reader:
            total += sum(1 for _, value in reader.items() if value is None)
    return total


def table_files(path: Path) -> set[str]:
    return {p.name for p in path.iterdir() if p.suffix == ".sst"}


def assert_clean_directory(path: Path, db: DB) -> None:
    expected = {table_path(path, i).name for i in db.table_ids}
    assert table_files(path) == expected
    assert not [p for p in path.iterdir() if p.suffix == ".tmp"]


def run_random_workload(db: DB, seed: int, ops: int) -> dict[bytes, bytes]:
    rng = random.Random(seed)
    model: dict[bytes, bytes] = {}
    for _ in range(ops):
        k = key(rng.randrange(60))
        if rng.random() < 0.3:
            db.delete(k)
            model.pop(k, None)
        else:
            v = bytes([rng.randrange(256)]) * rng.randrange(0, 40)
            db.put(k, v)
            model[k] = v
    return model


def test_option_validation(tmp_path: Path) -> None:
    for bad in (1, 0, -3):
        with pytest.raises(ValueError):
            DB.open(tmp_path, compaction_trigger=bad)
    DB.open(tmp_path, compaction_trigger=2).close()


def test_automatic_compaction_bounds_table_count(tmp_path: Path) -> None:
    with DB.open(tmp_path / "off", memtable_bytes=200, compaction_trigger=10**6) as db:
        for i in range(400):
            db.put(key(i), b"v" * 20)
        uncompacted = len(db.table_ids)
    with DB.open(tmp_path / "on", memtable_bytes=200, compaction_trigger=4) as db:
        for i in range(400):
            db.put(key(i), b"v" * 20)
        assert len(db.table_ids) < uncompacted // 4
        assert len(db.table_ids) <= 16
        assert_clean_directory(tmp_path / "on", db)
        assert list(db.scan()) == [(key(i), b"v" * 20) for i in range(400)]


@pytest.mark.parametrize("seed", range(4))
def test_data_matches_model_across_compactions_and_reopen(tmp_path: Path, seed: int) -> None:
    model: dict[bytes, bytes] = {}
    with DB.open(tmp_path, memtable_bytes=300, compaction_trigger=3) as db:
        model = run_random_workload(db, seed, 800)
        assert list(db.scan()) == sorted(model.items())
    with DB.open(tmp_path, memtable_bytes=300, compaction_trigger=3) as db:
        assert list(db.scan()) == sorted(model.items())
        for i in range(60):
            assert db.get(key(i)) == model.get(key(i))
        model.update(run_random_workload(db, seed + 100, 400))
        db.compact()
        assert_clean_directory(tmp_path, db)


def test_compact_merges_everything_and_drops_tombstones(tmp_path: Path) -> None:
    with DB.open(tmp_path, compaction_trigger=10**6) as db:
        for i in range(30):
            db.put(key(i), b"old")
        db.flush()
        for i in range(0, 30, 2):
            db.delete(key(i))
        db.flush()
        for i in range(1, 30, 4):
            db.put(key(i), b"new")
        db.flush()
        before = list(db.scan())
        assert len(db.table_ids) == 3
        assert live_tombstones(tmp_path, db) == 15
        db.compact()
        assert len(db.table_ids) == 1
        assert live_tombstones(tmp_path, db) == 0
        assert list(db.scan()) == before
        for i in range(30):
            assert db.get(key(i)) == dict(before).get(key(i))
        assert_clean_directory(tmp_path, db)


def test_compact_single_table_reclaims_tombstones(tmp_path: Path) -> None:
    with DB.open(tmp_path) as db:
        db.put(b"a", b"1")
        db.put(b"b", b"2")
        db.delete(b"a")
        db.flush()
        assert live_tombstones(tmp_path, db) == 1
        db.compact()
        assert len(db.table_ids) == 1
        assert live_tombstones(tmp_path, db) == 0
        assert list(db.scan()) == [(b"b", b"2")]


def test_compact_on_empty_db_is_a_noop(tmp_path: Path) -> None:
    with DB.open(tmp_path) as db:
        db.compact()
        assert db.table_ids == ()
        db.put(b"k", b"v")
        db.compact()  # memtable only, no tables
        assert db.table_ids == ()
        assert db.get(b"k") == b"v"


def test_compact_when_everything_is_deleted_leaves_no_tables(tmp_path: Path) -> None:
    with DB.open(tmp_path, compaction_trigger=10**6) as db:
        for i in range(20):
            db.put(key(i), b"v")
        db.flush()
        for i in range(20):
            db.delete(key(i))
        db.flush()
        db.compact()
        assert db.table_ids == ()
        assert table_files(tmp_path) == set()
        assert list(db.scan()) == []
        db.put(b"x", b"y")
        db.flush()
        assert db.get(b"x") == b"y"
    with DB.open(tmp_path) as db:
        assert list(db.scan()) == [(b"x", b"y")]


def test_tombstones_survive_compaction_that_excludes_the_bottom_table(tmp_path: Path) -> None:
    with DB.open(tmp_path, compaction_trigger=2) as db:
        for i in range(200):
            db.put(key(i), b"x" * 50)
        db.flush()  # one large bottom table
        db.delete(key(0))
        db.flush()  # tiny table holding a tombstone
        db.put(key(1000), b"z")
        db.flush()  # second tiny table: compacts with the tombstone table only
        assert len(db.table_ids) == 2
        assert live_tombstones(tmp_path, db) == 1
        assert db.get(key(0)) is None
        assert key(0) not in dict(db.scan())
        db.compact()
        assert len(db.table_ids) == 1
        assert live_tombstones(tmp_path, db) == 0
        assert db.get(key(0)) is None
        assert db.get(key(1)) == b"x" * 50


def test_tombstones_are_dropped_when_run_includes_the_bottom_table(tmp_path: Path) -> None:
    with DB.open(tmp_path, compaction_trigger=2) as db:
        db.put(b"a", b"1")
        db.put(b"b", b"2")
        db.flush()
        db.delete(b"a")
        db.put(b"c", b"3")
        db.flush()  # similar sizes: compacts into the bottom table
        assert len(db.table_ids) == 1
        assert live_tombstones(tmp_path, db) == 0
        assert list(db.scan()) == [(b"b", b"2"), (b"c", b"3")]


def test_old_table_files_are_removed(tmp_path: Path) -> None:
    with DB.open(tmp_path, compaction_trigger=10**6) as db:
        for batch in range(3):
            db.put(key(batch), b"v")
            db.flush()
        old = table_files(tmp_path)
        assert len(old) == 3
        db.compact()
        new = table_files(tmp_path)
        assert len(new) == 1
        assert not new & old


class _CommitFailure(OSError):
    pass


def _prepare_crash_db(path: Path) -> tuple[DB, dict[bytes, bytes]]:
    db = DB.open(path, compaction_trigger=10**6)
    model: dict[bytes, bytes] = {}
    for batch in range(3):
        for i in range(10):
            db.put(key(batch * 5 + i), f"v{batch}".encode())
            model[key(batch * 5 + i)] = f"v{batch}".encode()
        db.delete(key(batch))
        model.pop(key(batch), None)
        db.flush()
    return db, model


def test_crash_before_manifest_commit_keeps_old_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db, model = _prepare_crash_db(tmp_path)
    old_tables = db.table_ids

    def failing_commit(directory: object, manifest: Manifest) -> None:
        raise _CommitFailure("injected")

    with monkeypatch.context() as m:
        m.setattr(Manifest, "commit", staticmethod(failing_commit))
        with pytest.raises(_CommitFailure):
            db.compact()
    with pytest.raises(ClosedError):
        db.get(key(0))
    assert Manifest.load(tmp_path).tables == old_tables
    with DB.open(tmp_path) as reopened:
        assert reopened.table_ids == old_tables
        assert list(reopened.scan()) == sorted(model.items())
        assert_clean_directory(tmp_path, reopened)


def test_crash_after_manifest_commit_uses_new_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db, model = _prepare_crash_db(tmp_path)
    old_tables = db.table_ids
    real_commit = Manifest.commit

    def commit_then_fail(directory: Path, manifest: Manifest) -> None:
        real_commit(directory, manifest)
        raise _CommitFailure("injected")

    with monkeypatch.context() as m:
        m.setattr(Manifest, "commit", staticmethod(commit_then_fail))
        with pytest.raises(_CommitFailure):
            db.compact()
    with pytest.raises(ClosedError):
        db.scan()
    assert len(Manifest.load(tmp_path).tables) == 1
    assert not set(old_tables) & set(Manifest.load(tmp_path).tables)
    with DB.open(tmp_path) as reopened:
        assert len(reopened.table_ids) == 1
        assert list(reopened.scan()) == sorted(model.items())
        assert_clean_directory(tmp_path, reopened)
    assert (tmp_path / MANIFEST_NAME).is_file()


def test_failed_merge_leaves_database_usable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db, model = _prepare_crash_db(tmp_path)

    def failing_write(*args: object, **kwargs: object) -> int:
        raise _CommitFailure("disk full")

    with monkeypatch.context() as m:
        m.setattr("lsmkv.db.write_sstable", failing_write)
        with pytest.raises(_CommitFailure):
            db.compact()
    with db:
        assert list(db.scan()) == sorted(model.items())
        db.compact()
        assert len(db.table_ids) == 1
        assert list(db.scan()) == sorted(model.items())
