"""Tests for the skip-list memtable."""

import itertools

import pytest
from hypothesis import given
from hypothesis import strategies as st

from lsmkv.memtable import ENTRY_OVERHEAD, Memtable
from lsmkv.types import MAX_KEY_SIZE, MISSING

_keys = st.binary(min_size=1, max_size=4)
_ops = st.lists(
    st.one_of(
        st.tuples(st.just("put"), _keys, st.binary(max_size=8)),
        st.tuples(st.just("del"), _keys, st.none()),
    ),
    max_size=80,
)


def _apply(
    table: Memtable, model: dict[bytes, bytes | None], ops: list[tuple[str, bytes, bytes | None]]
) -> None:
    for op, key, value in ops:
        if op == "put":
            assert value is not None
            table.put(key, value)
            model[key] = value
        else:
            table.delete(key)
            model[key] = None


def test_empty() -> None:
    table = Memtable()
    assert len(table) == 0
    assert table.approx_bytes == 0
    assert table.get(b"a") is MISSING
    assert list(table.items()) == []


def test_put_get_and_overwrite() -> None:
    table = Memtable()
    table.put(b"a", b"1")
    table.put(b"a", b"22")
    assert table.get(b"a") == b"22"
    assert len(table) == 1


def test_delete_then_put() -> None:
    table = Memtable()
    table.put(b"a", b"1")
    table.delete(b"a")
    assert table.get(b"a") is None
    table.put(b"a", b"2")
    assert table.get(b"a") == b"2"
    assert len(table) == 1


def test_delete_absent_key_creates_tombstone() -> None:
    table = Memtable()
    table.delete(b"x")
    assert table.get(b"x") is None
    assert len(table) == 1
    assert list(table.items()) == [(b"x", None)]


def test_empty_value_is_not_tombstone() -> None:
    table = Memtable()
    table.put(b"a", b"")
    assert table.get(b"a") == b""
    assert table.get(b"a") is not None
    assert list(table.items()) == [(b"a", b"")]


def test_range_bounds_are_half_open() -> None:
    table = Memtable()
    for key in (b"a", b"b", b"c", b"d"):
        table.put(key, key)
    assert [k for k, _ in table.items(b"b", b"d")] == [b"b", b"c"]
    assert [k for k, _ in table.items(start=b"c")] == [b"c", b"d"]
    assert [k for k, _ in table.items(end=b"b")] == [b"a"]
    assert [k for k, _ in table.items(b"bb", b"cc")] == [b"c"]


def test_range_start_not_before_end_is_empty() -> None:
    table = Memtable()
    table.put(b"a", b"1")
    table.put(b"b", b"2")
    assert list(table.items(b"b", b"b")) == []
    assert list(table.items(b"b", b"a")) == []


def test_approx_bytes_tracks_overwrite_and_tombstone() -> None:
    table = Memtable()
    table.put(b"key", b"vvvv")
    assert table.approx_bytes == 3 + 4 + ENTRY_OVERHEAD
    table.put(b"key", b"v")
    assert table.approx_bytes == 3 + 1 + ENTRY_OVERHEAD
    table.delete(b"key")
    assert table.approx_bytes == 3 + ENTRY_OVERHEAD
    table.put(b"key", b"abc")
    assert table.approx_bytes == 3 + 3 + ENTRY_OVERHEAD


def test_input_validation() -> None:
    table = Memtable()
    with pytest.raises(ValueError):
        table.put(b"", b"v")
    with pytest.raises(ValueError):
        table.delete(b"k" * (MAX_KEY_SIZE + 1))
    with pytest.raises(TypeError):
        table.put("k", b"v")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        table.put(b"k", "v")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        table.get("k")  # type: ignore[arg-type]
    assert len(table) == 0


@pytest.mark.parametrize(("max_level", "p"), [(0, 0.25), (4, 0.0), (4, 1.0)])
def test_invalid_parameters(max_level: int, p: float) -> None:
    with pytest.raises(ValueError):
        Memtable(max_level=max_level, p=p)


def test_max_level_one_still_works() -> None:
    table = Memtable(max_level=1)
    for i in range(50):
        table.put(bytes([i + 1]), b"v")
    assert len(table) == 50
    assert table.get(bytes([25])) == b"v"


@given(_ops)
def test_matches_dict_model(ops: list[tuple[str, bytes, bytes | None]]) -> None:
    table = Memtable(seed=1)
    model: dict[bytes, bytes | None] = {}
    _apply(table, model, ops)

    assert len(table) == len(model)
    expected_bytes = sum(len(k) + len(v or b"") + ENTRY_OVERHEAD for k, v in model.items())
    assert table.approx_bytes == expected_bytes
    for key in {k for _, k, _ in ops} | {b"zzzzz"}:
        assert table.get(key) == model.get(key, MISSING)

    entries = list(table.items())
    assert entries == sorted(model.items())
    keys = [k for k, _ in entries]
    assert all(a < b for a, b in itertools.pairwise(keys))


@given(_ops, st.one_of(st.none(), _keys), st.one_of(st.none(), _keys))
def test_range_matches_model(
    ops: list[tuple[str, bytes, bytes | None]], start: bytes | None, end: bytes | None
) -> None:
    table = Memtable(seed=3)
    model: dict[bytes, bytes | None] = {}
    _apply(table, model, ops)
    expected = [
        (k, v)
        for k, v in sorted(model.items())
        if (start is None or k >= start) and (end is None or k < end)
    ]
    assert list(table.items(start, end)) == expected


@given(_ops, st.integers(min_value=0, max_value=2**32))
def test_same_seed_same_structure(ops: list[tuple[str, bytes, bytes | None]], seed: int) -> None:
    first, second = Memtable(seed=seed), Memtable(seed=seed)
    _apply(first, {}, ops)
    _apply(second, {}, ops)
    assert first._heights() == second._heights()
    assert list(first.items()) == list(second.items())


def test_different_seeds_change_structure() -> None:
    first, second = Memtable(seed=1), Memtable(seed=2)
    for i in range(200):
        first.put(i.to_bytes(2, "big"), b"")
        second.put(i.to_bytes(2, "big"), b"")
    assert first._heights() != second._heights()
