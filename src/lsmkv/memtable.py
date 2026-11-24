"""In-memory sorted write buffer implemented as a skip list."""

from __future__ import annotations

import random
from collections.abc import Iterator

from lsmkv.types import MISSING, Entry, Missing, validate_key, validate_value

ENTRY_OVERHEAD = 48
"""Fixed bytes charged per entry on top of key and value lengths."""


class _Node:
    """A skip-list node; ``forward[i]`` is the next node at level ``i``."""

    __slots__ = ("forward", "key", "value")

    def __init__(self, key: bytes, value: bytes | None, height: int) -> None:
        self.key = key
        self.value = value
        self.forward: list[_Node | None] = [None] * height


class Memtable:
    """A sorted map from ``bytes`` keys to ``bytes`` values or tombstones.

    Deletions are stored as tombstones (value ``None``) so that they can
    shadow older data in SSTables. Not thread-safe.
    """

    def __init__(self, seed: int = 0, max_level: int = 16, p: float = 0.25) -> None:
        """Create an empty memtable.

        Args:
            seed: Seed for level generation; equal seeds and equal operation
                sequences give identical structures.
            max_level: Maximum node height.
            p: Probability of promoting a node to the next level.

        Raises:
            ValueError: ``max_level`` is below 1 or ``p`` is not in (0, 1).
        """
        if max_level < 1:
            raise ValueError("max_level must be at least 1")
        if not 0.0 < p < 1.0:
            raise ValueError("p must be in the open interval (0, 1)")
        # Used only to balance the list, not for anything security related.
        self._rng = random.Random(seed)
        self._max_level = max_level
        self._p = p
        self._head = _Node(b"", None, max_level)
        self._level = 1
        self._count = 0
        self._bytes = 0

    def __len__(self) -> int:
        """Return the number of distinct keys, tombstones included."""
        return self._count

    @property
    def approx_bytes(self) -> int:
        """Approximate memory footprint: key and value sizes plus overhead."""
        return self._bytes

    def put(self, key: bytes, value: bytes) -> None:
        """Insert or overwrite ``key`` with ``value``."""
        self._set(validate_key(key), validate_value(value))

    def delete(self, key: bytes) -> None:
        """Record a tombstone for ``key``, whether or not it is present."""
        self._set(validate_key(key), None)

    def get(self, key: bytes) -> bytes | Missing | None:
        """Return the value, ``None`` for a tombstone, or ``MISSING``."""
        node = self._find_ge(validate_key(key))
        if node is not None and node.key == key:
            return node.value
        return MISSING

    def items(self, start: bytes | None = None, end: bytes | None = None) -> Iterator[Entry]:
        """Yield entries with ``start <= key < end`` in increasing key order.

        Tombstones are yielded as ``(key, None)``. A ``None`` bound is open.
        The memtable must not be modified while iterating.
        """
        node = self._head.forward[0] if start is None else self._find_ge(start)
        while node is not None and (end is None or node.key < end):
            yield node.key, node.value
            node = node.forward[0]

    def _heights(self) -> list[int]:
        """Return node heights in key order (used to check determinism)."""
        heights: list[int] = []
        node = self._head.forward[0]
        while node is not None:
            heights.append(len(node.forward))
            node = node.forward[0]
        return heights

    def _find_ge(self, key: bytes) -> _Node | None:
        """Return the first node whose key is ``>= key``."""
        node = self._head
        for level in range(self._level - 1, -1, -1):
            while (nxt := node.forward[level]) is not None and nxt.key < key:
                node = nxt
        return node.forward[0]

    def _random_height(self) -> int:
        height = 1
        while height < self._max_level and self._rng.random() < self._p:
            height += 1
        return height

    def _set(self, key: bytes, value: bytes | None) -> None:
        update: list[_Node] = [self._head] * self._max_level
        node = self._head
        for level in range(self._level - 1, -1, -1):
            while (nxt := node.forward[level]) is not None and nxt.key < key:
                node = nxt
            update[level] = node
        existing = node.forward[0]
        new_size = len(value) if value is not None else 0
        if existing is not None and existing.key == key:
            old_size = len(existing.value) if existing.value is not None else 0
            self._bytes += new_size - old_size
            existing.value = value
            return
        height = self._random_height()
        self._level = max(self._level, height)
        new = _Node(key, value, height)
        for level in range(height):
            new.forward[level] = update[level].forward[level]
            update[level].forward[level] = new
        self._count += 1
        self._bytes += len(key) + new_size + ENTRY_OVERHEAD
