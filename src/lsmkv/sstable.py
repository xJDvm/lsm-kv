"""Immutable, block-based sorted string tables (SSTables).

File layout (all integers little-endian)::

    [data block]* [index block] [bloom block] [footer]

    data block  = entry... | crc32:u32
    entry       = kind:u8 (1 = value, 2 = tombstone) | klen:u32 | vlen:u32 | key | value
    index block = (klen:u32 | last_key | off:u64 | len:u32)... | crc32:u32
    bloom block = BloomFilter.to_bytes() | crc32:u32
    footer      = index_off:u64 | index_len:u32 | bloom_off:u64 | bloom_len:u32
                  | count:u64 | magic "LSMKVT01" | crc32:u32 | 4 zero bytes (48 B)

The index holds one entry per data block: the block's last key, its offset and
its length (including the block's trailing CRC). Every CRC covers all bytes of
its block that precede it; the footer CRC covers the first 40 footer bytes.

All content read from disk is untrusted: the reader validates every offset and
length against the real file size before reading and reports any malformed
content as :class:`~lsmkv.errors.CorruptionError`.
"""

from __future__ import annotations

import bisect
import contextlib
import os
import struct
import zlib
from collections.abc import Iterable, Iterator
from types import TracebackType
from typing import BinaryIO

from lsmkv.bloom import BloomFilter
from lsmkv.errors import ClosedError, CorruptionError
from lsmkv.types import (
    MAX_KEY_SIZE,
    MAX_VALUE_SIZE,
    MISSING,
    Entry,
    Missing,
    validate_key,
    validate_value,
)

__all__ = ["SSTableReader", "write_sstable"]

_KIND_VALUE = 1
_KIND_TOMBSTONE = 2

_MAGIC = b"LSMKVT01"
_ENTRY_HEADER = struct.Struct("<BII")
_U32 = struct.Struct("<I")
_INDEX_TAIL = struct.Struct("<QI")
_FOOTER_BODY = struct.Struct("<QIQIQ8s")  # 40 bytes, covered by the footer CRC
_FOOTER_SIZE = 48
_CRC_SIZE = 4
_MIN_ENTRY_SIZE = _ENTRY_HEADER.size + 1  # header plus a one-byte key
_MIN_BLOCK_SIZE = _MIN_ENTRY_SIZE + _CRC_SIZE
_MIN_BLOOM_SIZE = 5 + _CRC_SIZE  # bloom header plus CRC


def _crc(data: bytes | bytearray) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def write_sstable(
    path: str | os.PathLike[str],
    entries: Iterable[Entry],
    *,
    block_size: int = 4096,
    fp_rate: float = 0.01,
) -> int:
    """Write ``entries`` to a new SSTable at ``path`` and return the entry count.

    Entries must have strictly increasing keys; a value of ``None`` is stored
    as a tombstone. Data blocks are streamed to disk and cut as soon as their
    payload reaches ``block_size`` bytes. Because ``entries`` is a one-shot
    iterable and the Bloom filter must be sized for the final entry count, the
    keys are kept in memory until the stream ends; that is ``O(keys)`` memory,
    which is acceptable at this engine's scale.

    The file is written to ``path`` directly and fsynced before returning;
    callers wanting atomic publication should write to a temporary name and
    rename. If writing fails, the partial file is removed.

    Raises:
        ValueError: ``block_size`` or ``fp_rate`` is out of range, a key or
            value is invalid, or keys are not strictly increasing.
        TypeError: A key or value is not ``bytes``.
    """
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    if not 0.0 < fp_rate < 1.0:
        raise ValueError("fp_rate must be in (0, 1)")
    try:
        with open(path, "wb") as f:
            count = _write_stream(f, entries, block_size, fp_rate)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)
        raise
    return count


def _write_stream(f: BinaryIO, entries: Iterable[Entry], block_size: int, fp_rate: float) -> int:
    keys: list[bytes] = []
    index = bytearray()
    block = bytearray()
    offset = 0
    prev: bytes | None = None

    def cut() -> None:
        nonlocal offset
        block.extend(_U32.pack(_crc(block)))
        f.write(block)
        index.extend(_U32.pack(len(prev or b"")) + (prev or b""))
        index.extend(_INDEX_TAIL.pack(offset, len(block)))
        offset += len(block)
        block.clear()

    for key, value in entries:
        validate_key(key)
        if prev is not None and key <= prev:
            raise ValueError("keys must be strictly increasing")
        if value is None:
            kind, payload = _KIND_TOMBSTONE, b""
        else:
            kind, payload = _KIND_VALUE, validate_value(value)
        block.extend(_ENTRY_HEADER.pack(kind, len(key), len(payload)))
        block.extend(key)
        block.extend(payload)
        keys.append(key)
        prev = key
        if len(block) >= block_size:
            cut()
    if block:
        cut()

    bloom = BloomFilter.for_capacity(len(keys), fp_rate)
    for key in keys:
        bloom.add(key)
    bloom_block = bloom.to_bytes()
    bloom_block += _U32.pack(_crc(bloom_block))
    index.extend(_U32.pack(_crc(index)))

    footer = _FOOTER_BODY.pack(
        offset, len(index), offset + len(index), len(bloom_block), len(keys), _MAGIC
    )
    f.write(index)
    f.write(bloom_block)
    f.write(footer + _U32.pack(_crc(footer)) + b"\x00" * 4)
    return len(keys)


def _parse_block(data: bytes, last_key: bytes) -> list[Entry]:
    """Validate one data block (including its CRC) and decode its entries."""
    if len(data) < _MIN_BLOCK_SIZE:
        raise CorruptionError("data block too short")
    body_len = len(data) - _CRC_SIZE
    if _U32.unpack_from(data, body_len)[0] != _crc(data[:body_len]):
        raise CorruptionError("data block checksum mismatch")
    entries: list[Entry] = []
    pos = 0
    prev: bytes | None = None
    while pos < body_len:
        if body_len - pos < _ENTRY_HEADER.size:
            raise CorruptionError("truncated entry header")
        kind, klen, vlen = _ENTRY_HEADER.unpack_from(data, pos)
        pos += _ENTRY_HEADER.size
        if kind not in (_KIND_VALUE, _KIND_TOMBSTONE):
            raise CorruptionError(f"invalid entry kind {kind}")
        if not 1 <= klen <= MAX_KEY_SIZE or vlen > MAX_VALUE_SIZE:
            raise CorruptionError("entry length out of range")
        if kind == _KIND_TOMBSTONE and vlen != 0:
            raise CorruptionError("tombstone with a value")
        if klen + vlen > body_len - pos:
            raise CorruptionError("entry extends past block end")
        key = data[pos : pos + klen]
        pos += klen
        if prev is not None and key <= prev:
            raise CorruptionError("keys not strictly increasing within block")
        prev = key
        value = None if kind == _KIND_TOMBSTONE else data[pos : pos + vlen]
        pos += vlen
        entries.append((key, value))
    if prev != last_key:
        raise CorruptionError("block last key does not match the index")
    return entries


class SSTableReader:
    """Read-only access to an SSTable file.

    Opening validates the footer, index and Bloom filter eagerly; data blocks
    are read, checksummed and validated lazily, one at a time. The reader holds
    the file open until :meth:`close` (or the end of a ``with`` block).

    Raises:
        CorruptionError: (from the constructor and all reads) the file is
            malformed, truncated or fails a checksum.
        OSError: The file cannot be opened.
    """

    __slots__ = (
        "_bloom",
        "_closed",
        "_count",
        "_file",
        "_last_keys",
        "_lengths",
        "_min_key",
        "_offsets",
    )

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._closed = False
        self._min_key: bytes | None = None
        # Kept open for the reader's lifetime; closed by close() or on load failure.
        self._file: BinaryIO = open(path, "rb")  # noqa: SIM115
        try:
            self._load()
        except BaseException:
            self._file.close()
            self._closed = True
            raise

    def _read_at(self, offset: int, length: int) -> bytes:
        self._file.seek(offset)
        data = self._file.read(length)
        if len(data) != length:
            raise CorruptionError("unexpected end of file")
        return data

    def _load(self) -> None:
        size = os.fstat(self._file.fileno()).st_size
        if size < _FOOTER_SIZE:
            raise CorruptionError("file too small to hold a footer")
        footer = self._read_at(size - _FOOTER_SIZE, _FOOTER_SIZE)
        index_off, index_len, bloom_off, bloom_len, count, magic = _FOOTER_BODY.unpack_from(footer)
        if magic != _MAGIC:
            raise CorruptionError("bad magic")
        if _U32.unpack_from(footer, _FOOTER_BODY.size)[0] != _crc(footer[: _FOOTER_BODY.size]):
            raise CorruptionError("footer checksum mismatch")
        if footer[_FOOTER_BODY.size + _CRC_SIZE :] != b"\x00" * 4:
            raise CorruptionError("nonzero footer padding")
        footer_off = size - _FOOTER_SIZE
        if (
            index_len < _CRC_SIZE
            or bloom_len < _MIN_BLOOM_SIZE
            or bloom_off != index_off + index_len
            or bloom_off + bloom_len != footer_off
        ):
            raise CorruptionError("inconsistent footer layout")

        index = self._read_at(index_off, index_len)
        self._offsets, self._lengths, self._last_keys = self._parse_index(index, index_off, count)
        self._count: int = count

        bloom_block = self._read_at(bloom_off, bloom_len)
        bloom_body = bloom_block[:-_CRC_SIZE]
        if _U32.unpack_from(bloom_block, bloom_len - _CRC_SIZE)[0] != _crc(bloom_body):
            raise CorruptionError("bloom block checksum mismatch")
        self._bloom = BloomFilter.from_bytes(bloom_body)

    @staticmethod
    def _parse_index(
        index: bytes, data_end: int, count: int
    ) -> tuple[list[int], list[int], list[bytes]]:
        body_len = len(index) - _CRC_SIZE
        if _U32.unpack_from(index, body_len)[0] != _crc(index[:body_len]):
            raise CorruptionError("index checksum mismatch")
        offsets: list[int] = []
        lengths: list[int] = []
        last_keys: list[bytes] = []
        pos = 0
        expected_off = 0
        while pos < body_len:
            if body_len - pos < _U32.size:
                raise CorruptionError("truncated index entry")
            klen = _U32.unpack_from(index, pos)[0]
            pos += _U32.size
            if not 1 <= klen <= MAX_KEY_SIZE or body_len - pos < klen + _INDEX_TAIL.size:
                raise CorruptionError("invalid index key length")
            key = index[pos : pos + klen]
            pos += klen
            off, length = _INDEX_TAIL.unpack_from(index, pos)
            pos += _INDEX_TAIL.size
            if last_keys and key <= last_keys[-1]:
                raise CorruptionError("index keys not strictly increasing")
            if off != expected_off or length < _MIN_BLOCK_SIZE or off + length > data_end:
                raise CorruptionError("data blocks are not contiguous or out of bounds")
            expected_off = off + length
            offsets.append(off)
            lengths.append(length)
            last_keys.append(key)
        if expected_off != data_end:
            raise CorruptionError("data region does not match the index")
        if not len(last_keys) <= count <= data_end // _MIN_ENTRY_SIZE:
            raise CorruptionError("entry count out of range")
        return offsets, lengths, last_keys

    def _check_open(self) -> None:
        if self._closed:
            raise ClosedError("sstable reader is closed")

    def _read_block(self, i: int) -> list[Entry]:
        self._check_open()
        data = self._read_at(self._offsets[i], self._lengths[i])
        return _parse_block(data, self._last_keys[i])

    def __len__(self) -> int:
        """Number of entries (values and tombstones) recorded in the footer."""
        return self._count

    @property
    def num_blocks(self) -> int:
        """Number of data blocks."""
        return len(self._offsets)

    @property
    def min_key(self) -> bytes | None:
        """Smallest key, or ``None`` for an empty table (reads the first block once)."""
        self._check_open()
        if self._min_key is None and self._offsets:
            self._min_key = self._read_block(0)[0][0]
        return self._min_key

    @property
    def max_key(self) -> bytes | None:
        """Largest key (taken from the index), or ``None`` for an empty table."""
        self._check_open()
        return self._last_keys[-1] if self._last_keys else None

    def get(self, key: bytes) -> bytes | Missing | None:
        """Look up ``key``.

        Returns the value, ``None`` if the key is stored as a tombstone, or
        :data:`~lsmkv.types.MISSING` if the table has no record of it.
        """
        self._check_open()
        validate_key(key)
        if not self._bloom.might_contain(key):
            return MISSING
        i = bisect.bisect_left(self._last_keys, key)
        if i == len(self._last_keys):
            return MISSING
        for k, v in self._read_block(i):
            if k == key:
                return v
            if k > key:
                break
        return MISSING

    def items(self, start: bytes | None = None, end: bytes | None = None) -> Iterator[Entry]:
        """Iterate entries with ``start <= key < end`` in key order.

        Either bound may be ``None`` for an open end. Blocks are read lazily as
        the iterator advances.
        """
        self._check_open()
        return self._iter_range(start, end)

    def _iter_range(self, start: bytes | None, end: bytes | None) -> Iterator[Entry]:
        first = 0 if start is None else bisect.bisect_left(self._last_keys, start)
        prev: bytes | None = None
        for i in range(first, len(self._offsets)):
            for key, value in self._read_block(i):
                if prev is not None and key <= prev:
                    raise CorruptionError("keys not strictly increasing across blocks")
                prev = key
                if start is not None and key < start:
                    continue
                if end is not None and key >= end:
                    return
                yield key, value

    def close(self) -> None:
        """Close the underlying file; further reads raise :class:`ClosedError`."""
        if not self._closed:
            self._closed = True
            self._file.close()

    def __enter__(self) -> SSTableReader:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
