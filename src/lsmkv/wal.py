"""Checksummed write-ahead log with torn-tail recovery.

Record layout (little-endian)::

    crc32:u32 | len:u32 | payload
    payload = op:u8 | klen:u32 | vlen:u32 | key | value

The CRC covers ``len`` and the payload. ``op`` is 1 for a put and 2 for a
delete (which carries an empty value).

Recovery guarantee: a log is read front to back and reading stops at the
first record that is short, fails its checksum or is structurally invalid.
A crash mid-append therefore loses at most the record being written, but
corruption in the middle of the log discards every record after it. The
recovered entries are always a prefix of the appended operations.

CRC32 detects accidental corruption only; it offers no protection against
deliberate tampering.
"""

from __future__ import annotations

import os
import struct
import zlib
from dataclasses import dataclass
from types import TracebackType
from typing import BinaryIO, Final

from lsmkv.errors import ClosedError
from lsmkv.types import MAX_KEY_SIZE, MAX_VALUE_SIZE, Entry, validate_key, validate_value

OP_PUT: Final = 1
OP_DELETE: Final = 2

_HEADER: Final = struct.Struct("<II")  # crc32, len
_PAYLOAD_HEAD: Final = struct.Struct("<BII")  # op, klen, vlen
_LEN_ONLY: Final = struct.Struct("<I")
_MAX_PAYLOAD: Final = _PAYLOAD_HEAD.size + MAX_KEY_SIZE + MAX_VALUE_SIZE


def _encode(op: int, key: bytes, value: bytes) -> bytes:
    payload = _PAYLOAD_HEAD.pack(op, len(key), len(value)) + key + value
    length = _LEN_ONLY.pack(len(payload))
    crc = zlib.crc32(payload, zlib.crc32(length))
    return _HEADER.pack(crc, len(payload)) + payload


class WAL:
    """An append-only write-ahead log.

    Use :meth:`open` to create an instance. Not safe for concurrent use.
    """

    def __init__(self, file: BinaryIO, *, sync: bool) -> None:
        self._file: BinaryIO | None = file
        self._sync = sync

    @classmethod
    def open(cls, path: str | os.PathLike[str], *, sync: bool = False) -> WAL:
        """Open (creating if needed) the log at ``path`` for appending.

        Args:
            path: Location of the log file.
            sync: If true, ``fsync`` after every append.
        """
        return cls(open(path, "ab"), sync=sync)

    def _live(self) -> BinaryIO:
        if self._file is None:
            raise ClosedError("WAL is closed")
        return self._file

    def _append(self, record: bytes) -> None:
        file = self._live()
        file.write(record)
        file.flush()
        if self._sync:
            os.fsync(file.fileno())

    def append_put(self, key: bytes, value: bytes) -> None:
        """Append a put record.

        Raises:
            ClosedError: the log is closed.
            TypeError: ``key`` or ``value`` is not ``bytes``.
            ValueError: ``key`` or ``value`` violates the size limits.
        """
        self._live()
        self._append(_encode(OP_PUT, validate_key(key), validate_value(value)))

    def append_delete(self, key: bytes) -> None:
        """Append a delete record.

        Raises:
            ClosedError: the log is closed.
            TypeError: ``key`` is not ``bytes``.
            ValueError: ``key`` violates the size limits.
        """
        self._live()
        self._append(_encode(OP_DELETE, validate_key(key), b""))

    def flush(self) -> None:
        """Flush buffered data to the OS, and to disk if ``sync`` is enabled."""
        file = self._live()
        file.flush()
        if self._sync:
            os.fsync(file.fileno())

    def reset(self) -> None:
        """Discard all records (used after the memtable has been flushed)."""
        file = self._live()
        file.flush()
        file.truncate(0)
        os.fsync(file.fileno())

    def close(self) -> None:
        """Flush and close the log. Closing twice is a no-op."""
        if self._file is None:
            return
        file, self._file = self._file, None
        try:
            file.flush()
            if self._sync:
                os.fsync(file.fileno())
        finally:
            file.close()

    def __enter__(self) -> WAL:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


@dataclass(frozen=True)
class WALScan:
    """Result of scanning a log file.

    Attributes:
        entries: Operations in append order; deletes have a value of ``None``.
        valid_length: Byte offset just after the last intact record.
        torn: True if bytes follow ``valid_length`` (a torn or corrupt tail).
    """

    entries: list[Entry]
    valid_length: int
    torn: bool


def _parse_payload(payload: bytes) -> Entry | None:
    """Decode a payload, or return ``None`` if it is structurally invalid."""
    if len(payload) < _PAYLOAD_HEAD.size:
        return None
    op, klen, vlen = _PAYLOAD_HEAD.unpack_from(payload)
    if klen == 0 or klen > MAX_KEY_SIZE or vlen > MAX_VALUE_SIZE:
        return None
    if _PAYLOAD_HEAD.size + klen + vlen != len(payload):
        return None
    key_end = _PAYLOAD_HEAD.size + klen
    key = payload[_PAYLOAD_HEAD.size : key_end]
    if op == OP_PUT:
        return key, payload[key_end:]
    if op == OP_DELETE and vlen == 0:
        return key, None
    return None


def scan_wal(path: str | os.PathLike[str]) -> WALScan:
    """Read the log at ``path`` up to its first bad record.

    Never raises for malformed content; a missing file yields an empty scan.
    Scanning stops at the first record that is short, fails its CRC, has an
    unknown op, inconsistent lengths, an empty or oversized key, or a length
    beyond the largest possible record.
    """
    entries: list[Entry] = []
    offset = 0
    try:
        with open(path, "rb") as file:
            size = os.fstat(file.fileno()).st_size
            while True:
                header = file.read(_HEADER.size)
                if len(header) < _HEADER.size:
                    break
                crc, length = _HEADER.unpack(header)
                if length > _MAX_PAYLOAD or length > size - offset - _HEADER.size:
                    break
                payload = file.read(length)
                if len(payload) < length:
                    break
                if zlib.crc32(payload, zlib.crc32(header[4:])) != crc:
                    break
                entry = _parse_payload(payload)
                if entry is None:
                    break
                entries.append(entry)
                offset += _HEADER.size + length
    except FileNotFoundError:
        return WALScan(entries, 0, False)
    return WALScan(entries, offset, offset < size)


def recover_wal(path: str | os.PathLike[str]) -> list[Entry]:
    """Return the operations in the log, repairing a torn tail.

    If the scan finds trailing garbage, the file is truncated to the last
    intact record and synced, so later appends follow valid data. Corruption
    in the middle of the log discards every record after it.
    """
    scan = scan_wal(path)
    if scan.torn:
        with open(path, "r+b") as file:
            file.truncate(scan.valid_length)
            os.fsync(file.fileno())
    return scan.entries
