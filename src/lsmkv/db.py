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
import threading
from pathlib import Path
from types import TracebackType
from typing import Final

from lsmkv.errors import ClosedError, CorruptionError
from lsmkv.manifest import Manifest, fsync_dir, table_path
from lsmkv.memtable import Memtable
from lsmkv.sstable import SSTableReader, write_sstable
from lsmkv.types import MISSING, validate_key, validate_value
from lsmkv.wal import WAL, recover_wal

__all__ = ["DB"]

_WAL_NAME: Final = "wal.log"


class DB:
    """An embedded, persistent, ordered key-value store.

    Keys are non-empty ``bytes`` and values are ``bytes`` (possibly empty, which
    is distinct from a deleted key). Create instances with :meth:`open`.

    Thread safety: a single lock serializes every public method, so one
    instance may be shared between threads of one process. Separate processes
    must not open the same directory; there is no file lock.

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
        manifest: Manifest,
        readers: list[SSTableReader],
        memtable: Memtable,
        wal: WAL,
    ) -> None:
        self._path = path
        self._memtable_bytes = memtable_bytes
        self._block_size = block_size
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
    ) -> DB:
        """Open the database in directory ``path``, creating it if needed.

        Opening recovers from any earlier crash: files the manifest does not
        reference (interrupted flushes) are deleted and the write-ahead log is
        replayed into a fresh memtable, discarding a torn tail.

        Args:
            path: Database directory.
            memtable_bytes: Approximate memtable size that triggers a flush.
            sync: If true, fsync the log on every write.
            block_size: Target data block size for new SSTables.

        Raises:
            ValueError: an option is out of range.
            CorruptionError: the manifest or a table it lists is damaged or
                missing.
            OSError: the directory cannot be created or read.
        """
        if memtable_bytes < 1:
            raise ValueError("memtable_bytes must be >= 1")
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)

        manifest = Manifest.load(directory)
        _remove_orphans(directory, manifest)
        readers: list[SSTableReader] = []
        wal: WAL | None = None
        try:
            for table_id in manifest.tables:
                readers.append(_open_table(directory, table_id))
            memtable = Memtable()
            for key, value in recover_wal(directory / _WAL_NAME):
                if value is None:
                    memtable.delete(key)
                else:
                    memtable.put(key, value)
            wal = WAL.open(directory / _WAL_NAME, sync=sync)
            return cls(
                directory,
                memtable_bytes=memtable_bytes,
                block_size=block_size,
                manifest=manifest,
                readers=readers,
                memtable=memtable,
                wal=wal,
            )
        except BaseException:
            for reader in readers:
                reader.close()
            if wal is not None:
                wal.close()
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

    def flush(self) -> None:
        """Write the memtable to a new SSTable. Does nothing if it is empty.

        Raises:
            ClosedError: the database is closed.
            OSError: the table could not be written or published.
        """
        with self._lock:
            self._check_open()
            self._flush_locked()

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

    def _shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        wal, self._wal = self._wal, None
        try:
            if wal is not None:
                wal.close()
        finally:
            for reader in self._readers:
                reader.close()
            self._readers = []


def _open_table(directory: Path, table_id: int) -> SSTableReader:
    """Open a table named by the manifest, mapping a missing file to corruption."""
    try:
        return SSTableReader(table_path(directory, table_id))
    except FileNotFoundError as exc:
        raise CorruptionError(f"table {table_id} listed in the manifest is missing") from exc


def _remove_orphans(directory: Path, manifest: Manifest) -> None:
    """Delete leftovers of interrupted flushes (invariant 6).

    Orphans are ``*.sst`` files the manifest does not list and any ``*.tmp``
    file. Only regular files directly inside ``directory`` are touched.
    """
    live = {table_path(directory, table_id).name for table_id in manifest.tables}
    for entry in directory.iterdir():
        orphan = entry.suffix == ".tmp" or (entry.suffix == ".sst" and entry.name not in live)
        if orphan and entry.is_file():
            with contextlib.suppress(FileNotFoundError):
                entry.unlink()
