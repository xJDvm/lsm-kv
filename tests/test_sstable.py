"""Tests for the SSTable writer and reader."""

import struct
import tempfile
import zlib
from collections.abc import Iterator
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from lsmkv.bloom import BloomFilter
from lsmkv.errors import ClosedError, CorruptionError
from lsmkv.sstable import SSTableReader, write_sstable
from lsmkv.types import MISSING, Entry

keys_st = st.binary(min_size=1, max_size=20)
values_st = st.one_of(st.none(), st.binary(max_size=40))
model_st = st.dictionaries(keys_st, values_st, max_size=60)


def _sorted(model: dict[bytes, bytes | None]) -> list[Entry]:
    return sorted(model.items())


@pytest.fixture
def path(tmp_path: Path) -> Path:
    return tmp_path / "t.sst"


@given(model=model_st, block_size=st.sampled_from([1, 16, 64, 4096]), absent=st.lists(keys_st))
def test_roundtrip_matches_model(
    model: dict[bytes, bytes | None], block_size: int, absent: list[bytes]
) -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.sst"
        assert write_sstable(p, _sorted(model), block_size=block_size) == len(model)
        with SSTableReader(p) as r:
            assert len(r) == len(model)
            assert list(r.items()) == _sorted(model)
            for k, v in model.items():
                assert r.get(k) == v
            for k in absent:
                if k not in model:
                    assert r.get(k) is MISSING


@given(
    model=model_st,
    start=st.one_of(st.none(), keys_st),
    end=st.one_of(st.none(), keys_st),
)
def test_ranged_items_match_model(
    model: dict[bytes, bytes | None], start: bytes | None, end: bytes | None
) -> None:
    expected = [
        (k, v)
        for k, v in _sorted(model)
        if (start is None or k >= start) and (end is None or k < end)
    ]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.sst"
        write_sstable(p, _sorted(model), block_size=48)
        with SSTableReader(p) as r:
            assert list(r.items(start, end)) == expected


def test_empty_table(path: Path) -> None:
    assert write_sstable(path, []) == 0
    with SSTableReader(path) as r:
        assert len(r) == 0
        assert r.num_blocks == 0
        assert r.min_key is None
        assert r.max_key is None
        assert list(r.items()) == []
        assert r.get(b"a") is MISSING


def test_single_key(path: Path) -> None:
    write_sstable(path, [(b"k", b"v")])
    with SSTableReader(path) as r:
        assert (len(r), r.num_blocks) == (1, 1)
        assert r.min_key == b"k" == r.max_key
        assert r.get(b"k") == b"v"
        assert r.get(b"j") is MISSING
        assert r.get(b"l") is MISSING


def test_empty_value_and_tombstone_are_distinct(path: Path) -> None:
    write_sstable(path, [(b"a", b""), (b"b", None)])
    with SSTableReader(path) as r:
        assert r.get(b"a") == b""
        assert r.get(b"b") is None
        assert r.get(b"c") is MISSING


def test_keys_at_block_boundaries(path: Path) -> None:
    # Each entry is 9 + 1 + 6 = 16 bytes, so block_size=32 packs exactly two
    # entries per block and every boundary key is a block's last or first key.
    entries: list[Entry] = [(bytes([ord("a") + i]), b"vvvvvv") for i in range(8)]
    write_sstable(path, entries, block_size=32)
    with SSTableReader(path) as r:
        assert r.num_blocks == 4
        assert list(r.items()) == entries
        for k, v in entries:
            assert r.get(k) == v
        assert list(r.items(b"b", b"d")) == entries[1:3]
        assert list(r.items(b"c", b"c")) == []
        assert r.min_key == b"a"
        assert r.max_key == b"h"


def test_tiny_block_size_makes_one_block_per_entry(path: Path) -> None:
    entries: list[Entry] = [(f"key{i:03}".encode(), b"x" * i) for i in range(100)]
    write_sstable(path, entries, block_size=1)
    with SSTableReader(path) as r:
        assert r.num_blocks == 100
        assert list(r.items()) == entries
        assert r.get(b"key050") == b"x" * 50


def test_value_larger_than_block_size(path: Path) -> None:
    big = bytes(range(256)) * 100
    entries: list[Entry] = [(b"a", b"1"), (b"b", big), (b"c", b"3")]
    write_sstable(path, entries, block_size=64)
    with SSTableReader(path) as r:
        assert r.get(b"b") == big
        assert list(r.items()) == entries


def test_accepts_generator(path: Path) -> None:
    def gen() -> Iterator[Entry]:
        yield b"a", b"1"
        yield b"b", None

    assert write_sstable(path, gen()) == 2


@pytest.mark.parametrize(
    "entries",
    [
        [(b"b", b"1"), (b"a", b"2")],
        [(b"a", b"1"), (b"a", b"2")],
        [(b"", b"1")],
    ],
)
def test_invalid_keys_raise_value_error(path: Path, entries: list[Entry]) -> None:
    with pytest.raises(ValueError):
        write_sstable(path, entries)
    assert not path.exists()


def test_write_to_directory_keeps_it_and_raises_original_error(tmp_path: Path) -> None:
    target = tmp_path / "dir.sst"
    target.mkdir()
    with pytest.raises((IsADirectoryError, PermissionError)):
        write_sstable(target, [(b"a", b"1")])
    assert target.is_dir()


def test_mid_stream_failure_removes_partial_file(path: Path) -> None:
    def gen() -> Iterator[Entry]:
        yield b"b", b"1"
        yield b"a", b"2"

    with pytest.raises(ValueError, match="strictly increasing"):
        write_sstable(path, gen(), block_size=1)
    assert not path.exists()


def _forged_table(path: Path, blocks: list[list[bytes]]) -> None:
    """Write a table with valid CRCs whose blocks may overlap in key range."""

    def crc(b: bytes) -> bytes:
        return struct.pack("<I", zlib.crc32(b) & 0xFFFFFFFF)

    data = bytearray()
    index = bytearray()
    all_keys: list[bytes] = []
    for keys in blocks:
        body = b"".join(struct.pack("<BII", 1, len(k), 1) + k + b"v" for k in keys)
        block = body + crc(body)
        index += struct.pack("<I", len(keys[-1])) + keys[-1]
        index += struct.pack("<QI", len(data), len(block))
        data += block
        all_keys += keys
    index += crc(bytes(index))
    bloom = BloomFilter.for_capacity(len(all_keys), 0.01)
    for k in all_keys:
        bloom.add(k)
    bloom_body = bloom.to_bytes()
    bloom_block = bloom_body + crc(bloom_body)
    footer = struct.pack(
        "<QIQIQ8s",
        len(data),
        len(index),
        len(data) + len(index),
        len(bloom_block),
        len(all_keys),
        b"LSMKVT01",
    )
    path.write_bytes(bytes(data + index) + bloom_block + footer + crc(footer) + b"\x00" * 4)


def test_forged_cross_block_ordering_is_rejected(path: Path) -> None:
    # Block 1 starts with b"a", which is <= block 0's last key b"b".
    _forged_table(path, [[b"b"], [b"a", b"c"]])
    with SSTableReader(path) as r:
        assert r.get(b"b") == b"v"
        with pytest.raises(CorruptionError):
            r.get(b"c")
        with pytest.raises(CorruptionError):
            list(r.items())


def test_forged_table_without_overlap_is_accepted(path: Path) -> None:
    _forged_table(path, [[b"a", b"b"], [b"c"]])
    with SSTableReader(path) as r:
        assert [k for k, _ in r.items()] == [b"a", b"b", b"c"]
        assert r.get(b"c") == b"v"


def test_invalid_types_and_parameters(path: Path) -> None:
    with pytest.raises(TypeError):
        write_sstable(path, [("a", b"1")])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        write_sstable(path, [(b"a", "1")])  # type: ignore[list-item]
    with pytest.raises(ValueError):
        write_sstable(path, [], block_size=0)
    with pytest.raises(ValueError):
        write_sstable(path, [], fp_rate=1.5)
    assert not path.exists()


def test_get_rejects_invalid_key(path: Path) -> None:
    write_sstable(path, [(b"a", b"1")])
    with SSTableReader(path) as r, pytest.raises(ValueError):
        r.get(b"")


def test_closed_reader(path: Path) -> None:
    write_sstable(path, [(b"a", b"1")])
    r = SSTableReader(path)
    it = r.items()
    r.close()
    r.close()  # idempotent
    with pytest.raises(ClosedError):
        r.get(b"a")
    with pytest.raises(ClosedError):
        r.items()
    with pytest.raises(ClosedError):
        next(it)
    with pytest.raises(ClosedError):
        _ = r.min_key
    with pytest.raises(ClosedError):
        _ = r.max_key


def test_context_manager_closes(path: Path) -> None:
    write_sstable(path, [(b"a", b"1")])
    with SSTableReader(path) as r:
        pass
    with pytest.raises(ClosedError):
        r.get(b"a")


def test_items_is_lazy(path: Path) -> None:
    entries: list[Entry] = [(bytes([65 + i]), b"v") for i in range(10)]
    write_sstable(path, entries, block_size=1)
    with SSTableReader(path) as r:
        it = r.items()
        assert next(it) == entries[0]
        assert next(it) == entries[1]
