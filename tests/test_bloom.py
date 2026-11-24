"""Tests for the Bloom filter."""

from __future__ import annotations

import struct

import pytest
from hypothesis import given
from hypothesis import strategies as st

from lsmkv.bloom import BloomFilter
from lsmkv.errors import CorruptionError


@given(st.sets(st.binary(max_size=64), max_size=200))
def test_no_false_negatives(keys: set[bytes]) -> None:
    bf = BloomFilter.for_capacity(len(keys))
    for k in keys:
        bf.add(k)
    assert all(bf.might_contain(k) for k in keys)


@pytest.mark.parametrize("p", [0.01, 0.05])
def test_false_positive_rate(p: float) -> None:
    n = 10_000
    bf = BloomFilter.for_capacity(n, p)
    for i in range(n):
        bf.add(b"in-%d" % i)
    probes = 20_000
    fps = sum(bf.might_contain(b"out-%d" % i) for i in range(probes))
    rate = fps / probes
    assert rate < 2 * p


def test_for_capacity_parameters() -> None:
    bf = BloomFilter.for_capacity(10_000, 0.01)
    assert bf.num_bits % 8 == 0
    assert bf.num_bits >= 95_000
    assert bf.num_hashes == 7


def test_for_capacity_zero_is_minimal() -> None:
    bf = BloomFilter.for_capacity(0)
    assert bf.num_bits == 64
    assert bf.num_hashes == 1
    assert not bf.might_contain(b"x")


def test_for_capacity_small_n_has_min_bits() -> None:
    assert BloomFilter.for_capacity(1).num_bits == 64


def test_for_capacity_hash_clamp() -> None:
    assert BloomFilter.for_capacity(1, 1e-300).num_hashes == 30


@pytest.mark.parametrize("n,p", [(-1, 0.01), (10, 0.0), (10, 1.0), (10, -0.5), (10, 2.0)])
def test_for_capacity_rejects_bad_params(n: int, p: float) -> None:
    with pytest.raises(ValueError):
        BloomFilter.for_capacity(n, p)


@pytest.mark.parametrize(
    "m,k",
    [(0, 1), (7, 1), (12, 1), (64, 0), (64, 31), (2**32, 1)],
)
def test_constructor_rejects_bad_params(m: int, k: int) -> None:
    with pytest.raises(ValueError):
        BloomFilter(m, k)


def test_constructor_rejects_wrong_bits_length() -> None:
    with pytest.raises(ValueError):
        BloomFilter(64, 3, bytearray(7))


def test_empty_filter_contains_nothing() -> None:
    bf = BloomFilter(64, 3)
    assert not bf.might_contain(b"")
    assert not bf.might_contain(b"abc")


def test_hashing_matches_pinned_vector() -> None:
    # Persisted filters must stay readable: the on-disk bytes are pinned.
    bf = BloomFilter(64, 3)
    bf.add(b"stable")
    assert bf.to_bytes().hex() == "40000000030000080000010020"


@given(st.sets(st.binary(max_size=32), max_size=100), st.binary(max_size=32))
def test_roundtrip(keys: set[bytes], probe: bytes) -> None:
    bf = BloomFilter.for_capacity(len(keys), 0.05)
    for k in keys:
        bf.add(k)
    restored = BloomFilter.from_bytes(bf.to_bytes())
    assert restored.num_bits == bf.num_bits
    assert restored.num_hashes == bf.num_hashes
    assert restored.to_bytes() == bf.to_bytes()
    assert restored.might_contain(probe) == bf.might_contain(probe)
    assert all(restored.might_contain(k) for k in keys)


def _blob(m: int, k: int, payload_len: int) -> bytes:
    return struct.pack("<IB", m, k) + bytes(payload_len)


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"\x00\x00\x00\x00",
        _blob(0, 1, 0),
        _blob(4, 1, 0),
        _blob(12, 1, 1),
        _blob(64, 0, 8),
        _blob(64, 31, 8),
        _blob(64, 3, 7),
        _blob(64, 3, 9),
        _blob(64, 3, 0),
        _blob(2**32 - 1, 3, 8),
        _blob(2**32 - 8, 3, 8),
    ],
)
def test_from_bytes_rejects_malformed(data: bytes) -> None:
    with pytest.raises(CorruptionError):
        BloomFilter.from_bytes(data)


def test_from_bytes_accepts_minimal() -> None:
    bf = BloomFilter.from_bytes(_blob(8, 1, 1))
    assert bf.num_bits == 8


@given(st.binary(max_size=256))
def test_from_bytes_arbitrary_input(data: bytes) -> None:
    try:
        bf = BloomFilter.from_bytes(data)
    except CorruptionError:
        return
    assert bf.to_bytes() == data
