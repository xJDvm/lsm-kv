"""Tests for the k-way merge iterator."""

import itertools
from collections.abc import Iterator

import pytest
from hypothesis import given
from hypothesis import strategies as st

from lsmkv.merge import drop_tombstones, merge_iterators
from lsmkv.types import Entry

_keys = st.binary(min_size=1, max_size=4)
_values = st.one_of(st.none(), st.binary(max_size=4))
_sources = st.lists(st.dictionaries(_keys, _values, max_size=12), max_size=6)


def _as_entries(source: dict[bytes, bytes | None]) -> list[Entry]:
    return sorted(source.items())


@given(_sources)
def test_matches_dict_model(sources: list[dict[bytes, bytes | None]]) -> None:
    model: dict[bytes, bytes | None] = {}
    for source in reversed(sources):  # oldest -> newest
        model.update(source)

    result = list(merge_iterators([_as_entries(s) for s in sources]))

    assert result == sorted(model.items())
    keys = [k for k, _ in result]
    assert all(a < b for a, b in itertools.pairwise(keys))


@given(_sources)
def test_drop_tombstones_matches_live_model(sources: list[dict[bytes, bytes | None]]) -> None:
    model: dict[bytes, bytes | None] = {}
    for source in reversed(sources):
        model.update(source)

    result = list(drop_tombstones(merge_iterators([_as_entries(s) for s in sources])))

    assert result == sorted((k, v) for k, v in model.items() if v is not None)


def test_newest_source_wins() -> None:
    newest = [(b"a", b"new")]
    oldest = [(b"a", b"old"), (b"b", b"old")]
    assert list(merge_iterators([newest, oldest])) == [(b"a", b"new"), (b"b", b"old")]


def test_tombstone_shadows_older_value_and_is_passed_through() -> None:
    sources: list[list[Entry]] = [[(b"a", None)], [(b"a", b"old")]]
    assert list(merge_iterators(sources)) == [(b"a", None)]


def test_value_resurrects_over_older_tombstone() -> None:
    sources: list[list[Entry]] = [[(b"a", b"new")], [(b"a", None)]]
    assert list(merge_iterators(sources)) == [(b"a", b"new")]


def test_no_sources() -> None:
    assert list(merge_iterators([])) == []


def test_empty_sources() -> None:
    assert list(merge_iterators([[], [], []])) == []
    assert list(merge_iterators([[], [(b"a", b"1")], []])) == [(b"a", b"1")]


def test_single_source_is_unchanged() -> None:
    source: list[Entry] = [(b"a", b"1"), (b"b", None), (b"c", b"")]
    assert list(merge_iterators([source])) == source


def test_all_tombstones() -> None:
    sources: list[list[Entry]] = [[(b"a", None), (b"c", None)], [(b"b", None), (b"c", None)]]
    merged = list(merge_iterators(sources))
    assert merged == [(b"a", None), (b"b", None), (b"c", None)]
    assert list(drop_tombstones(merged)) == []


def test_drop_tombstones_keeps_empty_values() -> None:
    entries: list[Entry] = [(b"a", b""), (b"b", None), (b"c", b"x")]
    assert list(drop_tombstones(entries)) == [(b"a", b""), (b"c", b"x")]


def test_accepts_one_shot_iterators() -> None:
    sources = [iter([(b"a", b"1")]), iter([(b"b", b"2")])]
    assert list(merge_iterators(sources)) == [(b"a", b"1"), (b"b", b"2")]


def _counting_infinite(offset: int, step: int, pulled: list[int], index: int) -> Iterator[Entry]:
    n = 0
    while True:
        pulled[index] += 1
        yield (offset + n * step).to_bytes(8, "big"), b"v"
        n += 1


def test_is_lazy_with_infinite_sources() -> None:
    pulled = [0, 0, 0]
    sources = [_counting_infinite(i, 3, pulled, i) for i in range(3)]

    first = list(itertools.islice(merge_iterators(sources), 30))

    assert [int.from_bytes(k, "big") for k, _ in first] == list(range(30))
    # Each source is pulled at most one item beyond what was consumed.
    assert sum(pulled) <= 30 + len(sources)


def test_does_not_pull_before_iteration() -> None:
    pulled = [0]
    merged = merge_iterators([_counting_infinite(0, 1, pulled, 0)])
    assert pulled == [0]
    next(merged)
    assert pulled[0] <= 2


@pytest.mark.parametrize("bad", [[(b"b", b"1"), (b"a", b"2")], [(b"a", b"1"), (b"a", b"2")]])
def test_unordered_or_duplicate_source_raises(bad: list[Entry]) -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        list(merge_iterators([bad, [(b"z", b"0")]]))
