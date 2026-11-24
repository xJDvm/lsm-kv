"""The database: a write-ahead log, a memtable and a stack of SSTables.

Writes go to the WAL and then the memtable. When the memtable grows past a
threshold it is flushed to a new immutable SSTable, which is published by
committing a new manifest. Reads consult the memtable and then the tables from
newest to oldest; the first record found wins, and a tombstone hides every
older value.
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType
from typing import Final

from lsmkv.compaction import pick_tiered_run
from lsmkv.errors import ClosedError, CorruptionError, LockedError
from lsmkv.manifest import MANIFEST_NAME, MANIFEST_TMP_NAME, Manifest, fsync_dir, table_path
from lsmkv.memtable import Memtable
from lsmkv.merge import drop_tombstones, merge_iterators
from lsmkv.sstable import SSTableReader, write_sstable
from lsmkv.types import MISSING, validate_key, validate_value
from lsmkv.wal import WAL, recover_wal

if sys.platform != "win32":
    import fcntl

__all__ = ["DB"]

_WAL_NAME: Final = "wal.log"
_LOCK_NAME: Final = "LOCK"
# Table names are zero-padded to six digits but grow past that (see table_path).
_TABLE_RE: Final = re.compile(r"\d{6,}\.sst")
_TABLE_TMP_RE: Final = re.compile(r"\d{6,}\.sst\.tmp")


class DB:
    """An embedded, persistent, ordered key-value store.

    Keys are non-empty ``bytes`` and values are ``bytes`` (possibly empty, which
    is distinct from a deleted key). Create instances with :meth:`open`.

    Thread safety: a single lock serializes every public method, so one
    instance may be shared between threads of one process. Opening takes an
    exclusive, non-blocking ``flock`` on the ``LOCK`` file in the directory, so
    a second open of the same directory, from this or another process, raises
    :class:`~lsmkv.errors.LockedError` until the first instance is closed. The
    lock is advisory and is skipped on platforms without ``fcntl`` (Windows),
    where callers must ensure a directory is opened only once.

    Durability: with ``sync=False`` (the default) an acknowledged write has
    reached the operating system but not necessarily the disk, so it survives a
    process crash but not a power failure. With ``sync=True`` every write is
    fsynced to the log before it is acknowledged.

    Failure handling: if a flush fails after the new table has been published,
    the in-memory state can no longer be trusted, so the database closes itself
    and later calls raise :class:`~lsmkv.errors.ClosedError`. Reopening the
    directory recovers every acknowledged write.
    """

    def __init__(
        self,
        path: Path,
        *,
        memtable_bytes: int,
        block_size: int,
        compaction_trigger: int,
        manifest: Manifest,
        readers: list[SSTableReader],
        memtable: Memtable,
        wal: WAL,
        lock_fd: int | None,
    ) -> None:
        self._path = path
        self._lock_fd = lock_fd
        self._memtable_bytes = memtable_bytes
        self._block_size = block_size
        self._compaction_trigger = compaction_trigger
        self._manifest = manifest
        self._readers = readers  # parallel to manifest.tables, newest first
        self._memtable = memtable
        self._wal: WAL | None = wal
        self._lock = threading.Lock()
        self._closed = False

    @classmethod
    def open(
        cls,
        path: str | os.PathLike[str],
        *,
        memtable_bytes: int = 4 * 1024 * 1024,
        sync: bool = False,
        block_size: int = 4096,
        compaction_trigger: int = 4,
    ) -> DB:
        """Open the database in directory ``path``, creating it if needed.

        Opening recovers from any earlier crash: table and temporary files the
        manifest does not reference (interrupted flushes) are deleted and the
        write-ahead log is replayed into a fresh memtable, discarding a torn
        tail. Files that do not follow the engine's naming are never touched.

        Args:
            path: Database directory.
            memtable_bytes: Approximate memtable size that triggers a flush.
            sync: If true, fsync the log on every write.
            block_size: Target data block size for new SSTables.
            compaction_trigger: Minimum number of similarly sized, adjacent
                tables that triggers an automatic compaction after a flush.

        Raises:
            ValueError: an option is out of range.
            CorruptionError: the manifest or a table it lists is damaged or
                missing, or the manifest is missing while table files exist.
            LockedError: the directory is already open.
            OSError: the directory cannot be created or read.
        """
        if memtable_bytes < 1:
            raise ValueError("memtable_bytes must be >= 1")
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        if compaction_trigger < 2:
            raise ValueError("compaction_trigger must be >= 2")
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)

        lock_fd = _lock_directory(directory)
        readers: list[SSTableReader] = []
        wal: WAL | None = None
        try:
            manifest = _load_or_create_manifest(directory)
            _remove_orphans(directory, manifest)
            for table_id in manifest.tables:
                readers.append(_open_table(directory, table_id))
            memtable = Memtable()
            for key, value in recover_wal(directory / _WAL_NAME):
                if value is None:
                    memtable.delete(key)
                else:
                    memtable.put(key, value)
            wal = WAL.open(directory / _WAL_NAME, sync=sync)
            fsync_dir(directory)  # make the log's directory entry durable
            return cls(
                directory,
                memtable_bytes=memtable_bytes,
                block_size=block_size,
                compaction_trigger=compaction_trigger,
                manifest=manifest,
                readers=readers,
                memtable=memtable,
                wal=wal,
                lock_fd=lock_fd,
            )
        except BaseException:
            # Whatever a close raises, the lock must still be released.
            try:
                if wal is not None:
                    wal.close()
            finally:
                try:
                    for reader in readers:
                        reader.close()
                finally:
                    _unlock(lock_fd)
            raise

    @property
    def table_ids(self) -> tuple[int, ...]:
        """Ids of the live SSTables, newest first."""
        with self._lock:
            self._check_open()
            return self._manifest.tables

    @property
    def memtable_entries(self) -> int:
        """Number of distinct keys (tombstones included) in the memtable."""
        with self._lock:
            self._check_open()
            return len(self._memtable)

    def put(self, key: bytes, value: bytes) -> None:
        """Store ``value`` under ``key``, replacing any previous value.

        The write is logged before it is applied, and may trigger a flush. If
        that flush fails the exception propagates but the write is durable in
        the log and visible after reopening.

        Raises:
            ClosedError: the database is closed.
            TypeError: ``key`` or ``value`` is not ``bytes``.
            ValueError: ``key`` or ``value`` violates the size limits.
        """
        with self._lock:
            wal = self._live_wal()
            validate_key(key)
            validate_value(value)
            wal.append_put(key, value)
            self._memtable.put(key, value)
            self._maybe_flush()

    def delete(self, key: bytes) -> None:
        """Delete ``key``; deleting an absent key is allowed.

        Raises:
            ClosedError: the database is closed.
            TypeError: ``key`` is not ``bytes``.
            ValueError: ``key`` is empty or too long.
        """
        with self._lock:
            wal = self._live_wal()
            validate_key(key)
            wal.append_delete(key)
            self._memtable.delete(key)
            self._maybe_flush()

    def get(self, key: bytes) -> bytes | None:
        """Return the value stored under ``key``, or ``None`` if absent or deleted.

        Raises:
            ClosedError: the database is closed.
            TypeError: ``key`` is not ``bytes``.
            ValueError: ``key`` is empty or too long.
            CorruptionError: a table needed for the lookup is damaged.
        """
        with self._lock:
            self._check_open()
            validate_key(key)
            found = self._memtable.get(key)
            if found is not MISSING:
                return found
            for reader in self._readers:
                found = reader.get(key)
                if found is not MISSING:
                    return found
            return None

    def scan(
        self, start: bytes | None = None, end: bytes | None = None
    ) -> Iterator[tuple[bytes, bytes]]:
        """Return the live entries with ``start <= key < end`` in key order.

        A ``None`` bound leaves that side open. The newest version of each key
        wins and deleted keys are omitted. If ``start >= end`` the result is
        empty.

        The result is a consistent snapshot: it is materialized while the lock
        is held, so later writes, flushes and compactions (which close and
        delete tables) cannot affect or invalidate an iteration in progress.
        The trade-off is O(result) memory, so bound the range when scanning a
        large database.

        Raises:
            ClosedError: the database is closed.
            TypeError: a bound is not ``bytes``.
            ValueError: a bound is empty or too long.
            CorruptionError: a table needed for the scan is damaged.
        """
        with self._lock:
            self._check_open()
            if start is not None:
                validate_key(start)
            if end is not None:
                validate_key(end)
            if start is not None and end is not None and start >= end:
                return iter(())
            sources = [
                self._memtable.items(start, end),
                *(reader.items(start, end) for reader in self._readers),
            ]
            return iter(list(drop_tombstones(merge_iterators(sources))))

    def flush(self) -> None:
        """Write the memtable to a new SSTable. Does nothing if it is empty.

        Raises:
            ClosedError: the database is closed.
            OSError: the table could not be written or published.
        """
        with self._lock:
            self._check_open()
            self._flush_locked()

    def compact(self) -> None:
        """Merge every SSTable into one (a manual major compaction).

        Shadowed values and all tombstones are discarded, since the output
        includes the oldest table. With no tables this does nothing; with a
        single table it is still rewritten so its tombstones are reclaimed. If
        every entry turns out to be a tombstone no table is written at all.
        The memtable is not touched; call :meth:`flush` first to include it.

        Raises:
            ClosedError: the database is closed.
            CorruptionError: a table is damaged.
            OSError: the merged table could not be written or published.
        """
        with self._lock:
            self._check_open()
            if self._readers:
                self._compact_locked(range(len(self._readers)))

    def close(self) -> None:
        """Close the log and every table. Closing twice is a no-op.

        The memtable is deliberately not flushed: every acknowledged write is
        already in the write-ahead log and is replayed on the next open, so
        closing stays cheap and cannot fail half way through writing a table.
        """
        with self._lock:
            self._shutdown()

    def __enter__(self) -> DB:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _check_open(self) -> None:
        if self._closed:
            raise ClosedError("database is closed")

    def _live_wal(self) -> WAL:
        self._check_open()
        assert self._wal is not None  # set whenever the database is open
        return self._wal

    def _maybe_flush(self) -> None:
        if self._memtable.approx_bytes >= self._memtable_bytes:
            self._flush_locked()

    def _flush_locked(self) -> None:
        """Flush the memtable; the caller holds the lock and has checked it is open.

        Crash analysis (DESIGN.md, write/flush protocol). The steps are:

        1. write ``<id>.sst.tmp`` and fsync it;
        2. rename it to ``<id>.sst`` and fsync the directory;
        3. commit a manifest that lists ``<id>``;
        4. reset (truncate) the WAL.

        A crash before step 3 completes leaves the old manifest in force: the
        ``.tmp`` or ``.sst`` file is an orphan that the next open deletes, and
        the untouched WAL restores the memtable. A crash after step 3 but
        before step 4 finishes leaves the new manifest and a WAL whose records
        are all also in the new table; replaying them into the memtable
        re-applies the same values on top of identical table contents, which
        changes no answer (invariant 4, idempotent recovery). Step 3 is a
        single atomic rename, so there is no state in between.
        """
        if len(self._memtable) == 0:
            return
        assert self._wal is not None  # callers have checked the database is open
        table_id = self._manifest.next_id
        final = table_path(self._path, table_id)
        tmp = final.with_name(final.name + ".tmp")
        # On failure write_sstable removes the temporary file and nothing else
        # has changed, so the database simply carries on.
        write_sstable(tmp, self._memtable.items(), block_size=self._block_size)
        try:
            os.replace(tmp, final)
            fsync_dir(self._path)
            manifest = Manifest(table_id + 1, (table_id, *self._manifest.tables))
            Manifest.commit(self._path, manifest)
            self._manifest = manifest
            self._readers.insert(0, SSTableReader(final))
            self._wal.reset()
            self._memtable = Memtable()
        except BaseException:
            # The published state may differ from memory; stop using it.
            self._shutdown()
            raise
        self._auto_compact_locked()

    def _auto_compact_locked(self) -> None:
        """Compact similarly sized adjacent tables until none qualify.

        Every pass shrinks the table list by at least one, so the loop is
        bounded by the current table count. A failure propagates to the caller
        of the flush that triggered it; the flushed data is already durable.
        """
        for _ in range(len(self._readers)):
            sizes = [table_path(self._path, i).stat().st_size for i in self._manifest.tables]
            run = pick_tiered_run(sizes, min_tables=self._compaction_trigger)
            if run is None:
                return
            self._compact_locked(run)

    def _compact_locked(self, run: range) -> None:
        """Merge the tables at manifest positions ``run`` into one table.

        The caller holds the lock, has checked the database is open and passes
        a non-empty, in-bounds, contiguous ``run`` (positions are newest first).
        The merged table takes the run's place, so the recency order of every
        other table is unchanged. Tombstones are dropped only if the run
        includes the oldest table: otherwise an older table outside the run may
        hold a value the tombstone still has to shadow (invariant 3). Recency is
        defined by manifest order, not by id, so the new table simply gets the
        next unused id.

        Crash analysis. The steps are:

        1. merge into ``<id>.sst.tmp`` and fsync it;
        2. rename it to ``<id>.sst`` and fsync the directory;
        3. commit a manifest with the run replaced by ``<id>``;
        4. close the old readers and unlink the old files.

        A crash before step 3 completes leaves the old manifest in force and
        every old table intact; the new file is an orphan deleted on the next
        open. A crash after step 3 leaves the new manifest in force and the old
        files are orphans, also deleted on open. Both states hold the same
        logical data. If the merged output is empty (only tombstones at the
        bottom) nothing is written and the manifest simply omits the run.

        A failure while reading or writing the temporary file leaves everything
        unchanged and the database usable. A failure after the rename follows
        the flush policy: the database shuts itself down and reopening
        recovers.
        """
        old_ids = self._manifest.tables[run.start : run.stop]
        old_readers = self._readers[run.start : run.stop]
        bottom = run.stop == len(self._readers)
        merged = merge_iterators([reader.items() for reader in old_readers])
        table_id = self._manifest.next_id
        final = table_path(self._path, table_id)
        tmp = final.with_name(final.name + ".tmp")
        count = write_sstable(
            tmp, drop_tombstones(merged) if bottom else merged, block_size=self._block_size
        )
        try:
            if count > 0:
                os.replace(tmp, final)
                fsync_dir(self._path)
                replacement: tuple[int, ...] = (table_id,)
            else:
                tmp.unlink()
                replacement = ()
            tables = (*self._manifest.tables[: run.start], *replacement)
            manifest = Manifest(
                table_id + len(replacement), (*tables, *self._manifest.tables[run.stop :])
            )
            Manifest.commit(self._path, manifest)
            self._manifest = manifest
            new_readers = [SSTableReader(final)] if count > 0 else []
            self._readers[run.start : run.stop] = new_readers
        except BaseException:
            self._shutdown()
            raise
        # The new manifest is committed, so failing to clean up is harmless:
        # leftover files are orphans that the next open removes.
        for reader in old_readers:
            with contextlib.suppress(OSError):
                reader.close()
        for old_id in old_ids:
            with contextlib.suppress(OSError):
                table_path(self._path, old_id).unlink()

    def _shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        wal, self._wal = self._wal, None
        lock_fd, self._lock_fd = self._lock_fd, None
        try:
            if wal is not None:
                wal.close()
        finally:
            try:
                for reader in self._readers:
                    reader.close()
                self._readers = []
            finally:
                _unlock(lock_fd)


def _lock_directory(directory: Path) -> int | None:
    """Take the exclusive process lock on ``directory`` and return its fd.

    The lock is an ``flock`` on the ``LOCK`` file. It belongs to the open file
    description, so it also excludes a second open in the same process, and the
    operating system drops it if the process dies. Returns ``None`` on
    platforms without ``fcntl``, where no lock is taken.

    Raises:
        LockedError: another instance holds the lock.
    """
    if sys.platform == "win32":
        return None
    fd = os.open(directory / _LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise LockedError("database is locked by another process") from None
    except BaseException:
        os.close(fd)
        raise
    return fd


def _unlock(fd: int | None) -> None:
    """Release a lock taken by :func:`_lock_directory` by closing its fd."""
    if fd is not None:
        os.close(fd)


def _load_or_create_manifest(directory: Path) -> Manifest:
    """Load the manifest, committing an empty one for a brand-new database.

    Every database that has ever been opened therefore has a MANIFEST. If it
    is missing while engine-named table files exist, the manifest was lost and
    guessing which tables are live could discard data, so open refuses.
    """
    if (directory / MANIFEST_NAME).exists():
        return Manifest.load(directory)
    if any(_TABLE_RE.fullmatch(entry.name) for entry in directory.iterdir()):
        raise CorruptionError("manifest missing but table files exist")
    manifest = Manifest(1, ())
    Manifest.commit(directory, manifest)
    return manifest


def _open_table(directory: Path, table_id: int) -> SSTableReader:
    """Open a table named by the manifest, mapping a missing file to corruption."""
    try:
        return SSTableReader(table_path(directory, table_id))
    except FileNotFoundError as exc:
        raise CorruptionError(f"table {table_id} listed in the manifest is missing") from exc


def _remove_orphans(directory: Path, manifest: Manifest) -> None:
    """Delete leftovers of interrupted flushes (invariant 6).

    Orphans are engine-named table files (``000042.sst``) the manifest does not
    list, engine-named temporaries (``000042.sst.tmp``) and ``MANIFEST.tmp``.
    Any other name, including ``LOCK`` and foreign ``.sst`` or ``.tmp`` files,
    is left alone. Only regular files directly inside ``directory`` are touched.
    """
    live = {table_path(directory, table_id).name for table_id in manifest.tables}
    for entry in directory.iterdir():
        name = entry.name
        orphan = (
            name == MANIFEST_TMP_NAME
            or _TABLE_TMP_RE.fullmatch(name) is not None
            or (_TABLE_RE.fullmatch(name) is not None and name not in live)
        )
        if orphan and entry.is_file():
            with contextlib.suppress(FileNotFoundError):
                entry.unlink()
