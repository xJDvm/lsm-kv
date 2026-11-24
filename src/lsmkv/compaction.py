"""Size-tiered compaction planning.

Compaction merges several SSTables into one so that reads touch fewer files and
shadowed values and tombstones are reclaimed. *Which* tables to merge is a pure
decision over their sizes, kept here so it can be tested exhaustively without
touching the disk; :class:`lsmkv.db.DB` performs the merge itself.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["pick_tiered_run"]


def pick_tiered_run(
    sizes: Sequence[int], *, min_tables: int, bucket_ratio: float = 2.0
) -> range | None:
    """Choose a run of similarly sized tables to compact together.

    ``sizes`` are file sizes in manifest order, newest table first. The result
    is the index range of the longest contiguous run of at least ``min_tables``
    tables whose sizes are similar, meaning ``max(run) <= bucket_ratio *
    min(run)`` (a size of 0 counts as 1). When several runs have the same
    length the newest one (smallest start index) is returned. ``None`` means no
    such run exists.

    The run must be contiguous in recency order. The merged table takes the
    run's place in the manifest, and a read that finds a key in a table must
    never see an older version than one held by a newer table. Merging tables
    that are not adjacent would put the output on one side of a table whose
    versions interleave with the run, so some key could resolve to a stale
    value (newest-wins ordering, invariant 2). A contiguous run can be replaced
    in place without changing the relative order of any other table.

    Similarity is hereditary (any sub-run of a similar run is similar), so the
    longest run is found by extending a window from every start index.

    Raises:
        ValueError: ``min_tables < 2`` or ``bucket_ratio < 1``.
    """
    if min_tables < 2:
        raise ValueError("min_tables must be >= 2")
    if bucket_ratio < 1.0:
        raise ValueError("bucket_ratio must be >= 1")
    clamped = [max(size, 1) for size in sizes]
    best: range | None = None
    for start in range(len(clamped)):
        low = high = clamped[start]
        stop = start + 1
        while stop < len(clamped):
            size = clamped[stop]
            if max(high, size) > bucket_ratio * min(low, size):
                break
            low, high = min(low, size), max(high, size)
            stop += 1
        length = stop - start
        if length >= min_tables and (best is None or length > len(best)):
            best = range(start, stop)
    return best
