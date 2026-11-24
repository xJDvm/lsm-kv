"""K-way merge of sorted entry streams with newest-wins shadowing.

This is the core of both reads (``scan`` merges the memtable and every
SSTable) and compaction (which merges tables into a new one).
"""

import heapq
from collections.abc import Iterable, Iterator, Sequence

from lsmkv.types import Entry


def merge_iterators(sources: Sequence[Iterable[Entry]]) -> Iterator[Entry]:
    """Merge sorted sources into one sorted stream, newest source winning.

    Each source must yield entries with strictly increasing keys. Index 0 of
    ``sources`` is the newest source. The output has strictly increasing keys
    and contains every key exactly once, taken from the newest source that
    holds it. Tombstones (value ``None``) are passed through unchanged so that
    they keep shadowing older values; use :func:`drop_tombstones` to remove
    them.

    The merge is lazy: it holds one pending entry per source and pulls from a
    source only when its previous entry has been consumed. Each output entry
    costs O(log k) for ``k`` sources.

    Raises:
        ValueError: (while iterating) a source yields a key that is not
            strictly greater than its previous key.
    """
    iterators = [iter(source) for source in sources]
    # Heap items are (key, source_index, value); the unique source index makes
    # ties resolve to the newest source and never compares values.
    heap: list[tuple[bytes, int, bytes | None]] = []

    def push_next(index: int, previous: bytes | None) -> None:
        entry = next(iterators[index], None)
        if entry is None:
            return
        key, value = entry
        if previous is not None and key <= previous:
            raise ValueError(f"source {index} is not strictly increasing at key {key!r}")
        heapq.heappush(heap, (key, index, value))

    for index in range(len(iterators)):
        push_next(index, None)

    last_key: bytes | None = None
    while heap:
        key, index, value = heapq.heappop(heap)
        push_next(index, key)
        if key == last_key:
            continue
        last_key = key
        yield key, value


def drop_tombstones(entries: Iterable[Entry]) -> Iterator[tuple[bytes, bytes]]:
    """Yield only live entries, discarding tombstones."""
    for key, value in entries:
        if value is not None:
            yield key, value
