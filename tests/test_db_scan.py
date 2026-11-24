"""Tests for DB.scan: ordering, shadowing, bounds, snapshots and a model check."""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from lsmkv import DB, ClosedError


def test_scan_empty_db(tmp_path: Path) -> None:
    with DB.open(tmp_path) as db:
        assert list(db.scan()) == []
        assert list(db.scan(b"a", b"z")) == []


def test_scan_across_memtable_and_tables(tmp_path: Path) -> None:
    with DB.open(tmp_path) as db:
        db.put(b"a", b"1")
        db.put(b"b", b"1")
        db.put(b"c", b"1")
        db.flush()
        db.put(b"b", b"2")
        db.delete(b"c")
        db.put(b"d", b"1")
        db.flush()
        db.put(b"a", b"3")
        db.delete(b"d")
        db.put(b"e", b"")
        assert len(db.table_ids) == 2
        assert list(db.scan()) == [(b"a", b"3"), (b"b", b"2"), (b"e", b"")]


def test_scan_with_small_memtable_many_tables(tmp_path: Path) -> None:
    expected: dict[bytes, bytes] = {}
    with DB.open(tmp_path, memtable_bytes=64) as db:
        for i in range(60):
            key = f"k{i % 20:03d}".encode()
            if i % 7 == 0:
                db.delete(key)
                expected.pop(key, None)
            else:
                value = f"v{i}".encode()
                db.put(key, value)
                expected[key] = value
        assert len(db.table_ids) > 1
        assert list(db.scan()) == sorted(expected.items())


def test_scan_bounds(tmp_path: Path) -> None:
    with DB.open(tmp_path) as db:
        for key in (b"a", b"b", b"c", b"d"):
            db.put(key, key)
        db.flush()
        db.put(b"e", b"e")
        keys = [b"a", b"b", b"c", b"d", b"e"]
        assert [k for k, _ in db.scan(b"b", b"d")] == [b"b", b"c"]
        assert [k for k, _ in db.scan(b"b")] == keys[1:]
        assert [k for k, _ in db.scan(end=b"c")] == [b"a", b"b"]
        assert [k for k, _ in db.scan(None, None)] == keys
        assert [k for k, _ in db.scan(b"bb", b"cc")] == [b"c"]
        assert list(db.scan(b"d", b"b")) == []
        assert list(db.scan(b"c", b"c")) == []
        assert list(db.scan(b"x")) == []


def test_scan_rejects_invalid_bounds(tmp_path: Path) -> None:
    with DB.open(tmp_path) as db:
        with pytest.raises(ValueError):
            db.scan(b"")
        with pytest.raises(ValueError):
            db.scan(None, b"")
        with pytest.raises(TypeError):
            db.scan("a")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            db.scan(None, "z")  # type: ignore[arg-type]


def test_scan_on_closed_db(tmp_path: Path) -> None:
    db = DB.open(tmp_path)
    db.close()
    with pytest.raises(ClosedError):
        db.scan()


def test_scan_after_reopen(tmp_path: Path) -> None:
    with DB.open(tmp_path) as db:
        db.put(b"a", b"1")
        db.put(b"b", b"1")
        db.flush()
        db.put(b"b", b"2")
        db.delete(b"a")
        db.put(b"c", b"3")
    with DB.open(tmp_path) as db:
        assert list(db.scan()) == [(b"b", b"2"), (b"c", b"3")]


def test_scan_is_a_snapshot(tmp_path: Path) -> None:
    with DB.open(tmp_path, memtable_bytes=64) as db:
        for i in range(20):
            db.put(f"k{i:03d}".encode(), b"old")
        snapshot = db.scan()
        first = next(snapshot)
        for i in range(20):
            db.put(f"k{i:03d}".encode(), b"new")
            db.put(f"z{i:03d}".encode(), b"extra")
        db.delete(b"k005")
        db.flush()
        rest = list(snapshot)
        assert [first, *rest] == [(f"k{i:03d}".encode(), b"old") for i in range(20)]
        assert db.get(b"k005") is None


def test_scan_survives_close(tmp_path: Path) -> None:
    db = DB.open(tmp_path)
    db.put(b"a", b"1")
    snapshot = db.scan()
    db.close()
    assert list(snapshot) == [(b"a", b"1")]


_KEYS = st.binary(min_size=1, max_size=3)
_OPS = st.one_of(
    st.tuples(st.just("put"), _KEYS, st.binary(max_size=4)),
    st.tuples(st.just("delete"), _KEYS, st.just(b"")),
    st.tuples(st.just("flush"), st.just(b"x"), st.just(b"")),
)
_BOUND = st.one_of(st.none(), _KEYS)


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    ops=st.lists(_OPS, max_size=60),
    ranges=st.lists(st.tuples(_BOUND, _BOUND), min_size=1, max_size=8),
)
def test_scan_matches_dict_model(
    tmp_path_factory: pytest.TempPathFactory,
    ops: list[tuple[str, bytes, bytes]],
    ranges: list[tuple[bytes | None, bytes | None]],
) -> None:
    model: dict[bytes, bytes] = {}
    with DB.open(tmp_path_factory.mktemp("db"), memtable_bytes=48) as db:
        for op, key, value in ops:
            if op == "put":
                db.put(key, value)
                model[key] = value
            elif op == "delete":
                db.delete(key)
                model.pop(key, None)
            else:
                db.flush()
        for start, end in ranges:
            expected = sorted(
                (k, v)
                for k, v in model.items()
                if (start is None or k >= start) and (end is None or k < end)
            )
            assert list(db.scan(start, end)) == expected
