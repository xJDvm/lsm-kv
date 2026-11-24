"""The MANIFEST: the atomic, authoritative list of live SSTables.

On disk it is a small JSON document::

    {"version": 1, "next_id": 7, "tables": [6, 4, 3]}

``tables`` lists table ids newest first and ``next_id`` is the id the next
table will receive. The file is replaced atomically (write a temporary file,
fsync it, rename it over ``MANIFEST``, fsync the directory), so a reader sees
either the old or the new manifest, never a mix. Anything in the directory that
the manifest does not mention is an orphan of an interrupted operation.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from lsmkv.errors import CorruptionError

__all__ = ["MANIFEST_NAME", "Manifest", "fsync_dir", "table_path"]

MANIFEST_NAME: Final = "MANIFEST"
_TMP_NAME: Final = "MANIFEST.tmp"
_VERSION: Final = 1
_MAX_MANIFEST_SIZE: Final = 1024 * 1024


def table_path(directory: str | os.PathLike[str], table_id: int) -> Path:
    """Return the path of the SSTable with id ``table_id`` inside ``directory``.

    The file name is derived from an integer id (``000042.sst``), never from
    external text, so it cannot contain path separators or ``..`` components.
    """
    return Path(directory) / f"{table_id:06d}.sst"


def fsync_dir(directory: str | os.PathLike[str]) -> None:
    """Flush a directory's entries (creations, renames) to stable storage."""
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError:
        # Some platforms and filesystems refuse fsync on a directory handle
        # (for example Windows); the rename is then as durable as it can be.
        pass
    finally:
        os.close(fd)


def _problem(next_id: object, tables: object) -> str | None:
    """Describe what is wrong with a manifest's fields, or return ``None``."""
    # bool is a subclass of int, so it must be excluded explicitly.
    if type(next_id) is not int:
        return "next_id must be an integer"
    if not isinstance(tables, (list, tuple)):
        return "tables must be a list"
    for table_id in tables:
        if type(table_id) is not int:
            return "table ids must be integers"
        if table_id < 1:
            return "table ids must be positive"
    if len(set(tables)) != len(tables):
        return "table ids must be unique"
    if next_id < 1 or (tables and next_id <= max(tables)):
        return "next_id must exceed every table id"
    return None


@dataclass(frozen=True)
class Manifest:
    """The set of live SSTables.

    Attributes:
        next_id: Id that the next SSTable will be given.
        tables: Live table ids, newest first.
    """

    next_id: int
    tables: tuple[int, ...]

    @staticmethod
    def load(directory: str | os.PathLike[str]) -> Manifest:
        """Read the manifest in ``directory``.

        A missing file means a fresh database and yields ``Manifest(1, ())``.
        A leftover ``MANIFEST.tmp`` from an interrupted commit is ignored.

        Raises:
            CorruptionError: the file is larger than 1 MiB, is not valid JSON,
                has an unsupported version or any field of the wrong type or
                value.
        """
        try:
            with open(Path(directory) / MANIFEST_NAME, "rb") as f:
                raw = f.read(_MAX_MANIFEST_SIZE + 1)
        except FileNotFoundError:
            return Manifest(1, ())
        if len(raw) > _MAX_MANIFEST_SIZE:
            raise CorruptionError("manifest is larger than 1 MiB")
        try:
            doc = json.loads(raw)
        except (ValueError, RecursionError) as exc:
            raise CorruptionError(f"manifest is not valid JSON: {exc}") from exc
        if not isinstance(doc, dict) or set(doc) != {"version", "next_id", "tables"}:
            raise CorruptionError("manifest must be an object with version, next_id and tables")
        version = doc["version"]
        if type(version) is not int or version != _VERSION:
            raise CorruptionError(f"unsupported manifest version {version!r}")
        problem = _problem(doc["next_id"], doc["tables"])
        if problem is not None:
            raise CorruptionError(f"invalid manifest: {problem}")
        return Manifest(doc["next_id"], tuple(doc["tables"]))

    @staticmethod
    def commit(directory: str | os.PathLike[str], manifest: Manifest) -> None:
        """Atomically make ``manifest`` the current manifest of ``directory``.

        Raises:
            ValueError: ``manifest`` violates the manifest invariants.
            OSError: writing or renaming failed; the previous manifest is
                still in effect unless the rename itself succeeded.
        """
        problem = _problem(manifest.next_id, manifest.tables)
        if problem is not None:
            raise ValueError(f"invalid manifest: {problem}")
        doc = {"version": _VERSION, "next_id": manifest.next_id, "tables": list(manifest.tables)}
        base = Path(directory)
        tmp = base / _TMP_NAME
        with open(tmp, "wb") as f:
            f.write(json.dumps(doc).encode("ascii"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, base / MANIFEST_NAME)
        fsync_dir(base)
