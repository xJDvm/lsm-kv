"""Bloom filter with Kirsch-Mitzenmacher double hashing.

Filters are persisted inside SSTables, so the hash function must be stable
across processes and Python versions.
"""

from __future__ import annotations

import hashlib
import math
import struct

from lsmkv.errors import CorruptionError

_HEADER = struct.Struct("<IB")
_MIN_BITS = 64
_MAX_BITS = 2**32 - 1
_MAX_HASHES = 30
_MASK64 = (1 << 64) - 1


class BloomFilter:
    """A fixed-size Bloom filter over ``bytes`` keys.

    Args:
        num_bits: Size of the bit array; must be a positive multiple of 8.
        num_hashes: Number of probe positions per key, in ``[1, 30]``.
        bits: Optional initial bit array of exactly ``num_bits // 8`` bytes.
            It is used directly (not copied).

    Raises:
        ValueError: If any argument is out of range.
    """

    __slots__ = ("_bits", "_num_bits", "_num_hashes")

    def __init__(self, num_bits: int, num_hashes: int, bits: bytearray | None = None) -> None:
        if not 8 <= num_bits <= _MAX_BITS or num_bits % 8 != 0:
            raise ValueError("num_bits must be a multiple of 8 in [8, 2**32 - 1]")
        if not 1 <= num_hashes <= _MAX_HASHES:
            raise ValueError(f"num_hashes must be in [1, {_MAX_HASHES}]")
        if bits is None:
            bits = bytearray(num_bits // 8)
        elif len(bits) != num_bits // 8:
            raise ValueError("bits length must equal num_bits // 8")
        self._num_bits = num_bits
        self._num_hashes = num_hashes
        self._bits = bits

    @classmethod
    def for_capacity(cls, n: int, fp_rate: float = 0.01) -> BloomFilter:
        """Build a filter sized for ``n`` keys at target false-positive rate ``fp_rate``.

        Raises:
            ValueError: If ``n < 0`` or ``fp_rate`` is not in ``(0, 1)``.
        """
        if n < 0:
            raise ValueError("n must be >= 0")
        if not 0.0 < fp_rate < 1.0:
            raise ValueError("fp_rate must be in (0, 1)")
        if n == 0:
            return cls(_MIN_BITS, 1)
        ln2 = math.log(2)
        m = math.ceil(-n * math.log(fp_rate) / (ln2 * ln2))
        m = min(max(m, _MIN_BITS), _MAX_BITS - 7)
        m = (m + 7) // 8 * 8
        k = min(max(round(m / n * ln2), 1), _MAX_HASHES)
        return cls(m, k)

    @property
    def num_bits(self) -> int:
        """Size of the bit array in bits."""
        return self._num_bits

    @property
    def num_hashes(self) -> int:
        """Number of probe positions per key."""
        return self._num_hashes

    def _positions(self, key: bytes) -> list[int]:
        # blake2b is deterministic across processes (unlike built-in hash(),
        # which is salted per run); this is required because filters are persisted.
        digest = hashlib.blake2b(key, digest_size=16).digest()
        h1 = int.from_bytes(digest[:8], "little")
        h2 = int.from_bytes(digest[8:], "little") | 1
        m = self._num_bits
        return [((h1 + i * h2) & _MASK64) % m for i in range(self._num_hashes)]

    def add(self, key: bytes) -> None:
        """Insert ``key`` into the filter."""
        for pos in self._positions(key):
            self._bits[pos >> 3] |= 1 << (pos & 7)

    def might_contain(self, key: bytes) -> bool:
        """Return ``False`` if ``key`` is definitely absent, ``True`` if possibly present."""
        bits = self._bits
        return all(bits[pos >> 3] & (1 << (pos & 7)) for pos in self._positions(key))

    def to_bytes(self) -> bytes:
        """Serialize as ``m_bits:u32 | k:u8 | bitarray`` (little-endian, no checksum)."""
        return _HEADER.pack(self._num_bits, self._num_hashes) + bytes(self._bits)

    @classmethod
    def from_bytes(cls, data: bytes) -> BloomFilter:
        """Parse the output of :meth:`to_bytes`.

        Raises:
            CorruptionError: If the header or length is invalid.
        """
        if len(data) < _HEADER.size:
            raise CorruptionError("bloom filter truncated")
        num_bits, num_hashes = _HEADER.unpack_from(data)
        if not 8 <= num_bits <= _MAX_BITS or num_bits % 8 != 0:
            raise CorruptionError(f"invalid bloom bit count {num_bits}")
        if not 1 <= num_hashes <= _MAX_HASHES:
            raise CorruptionError(f"invalid bloom hash count {num_hashes}")
        if len(data) - _HEADER.size != num_bits // 8:
            raise CorruptionError("bloom bit array length mismatch")
        return cls(num_bits, num_hashes, bytearray(data[_HEADER.size :]))
