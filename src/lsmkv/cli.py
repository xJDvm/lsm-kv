"""Command-line tools for inspecting lsm-kv files.

Exit codes: ``0`` success, ``1`` the file is corrupt, ``2`` usage error or the
file cannot be opened (missing, unreadable, a directory, ...).
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections.abc import Sequence

from lsmkv.errors import CorruptionError
from lsmkv.sstable import SSTableReader

__all__ = ["main"]

#: Largest accepted value of ``--entries``.
MAX_ENTRIES = 10_000
#: Default number of entries printed by ``inspect``.
DEFAULT_ENTRIES = 10
#: Values (and keys) longer than this many bytes are cut when displayed.
MAX_DISPLAY_BYTES = 64

_EXIT_OK = 0
_EXIT_CORRUPT = 1
_EXIT_USAGE = 2


def format_bytes(data: bytes, limit: int = MAX_DISPLAY_BYTES) -> str:
    """Render ``data`` for humans without ever emitting control characters.

    Printable ASCII is shown as-is; anything else is shown as ``0x`` followed by
    lowercase hex. Inputs longer than ``limit`` bytes are cut to ``limit`` bytes
    and suffixed with ``...(N bytes)`` where ``N`` is the full length.
    """
    shown = data[:limit]
    printable = all(0x20 <= b <= 0x7E for b in shown)
    text = shown.decode("ascii") if printable else "0x" + shown.hex()
    if len(data) > limit:
        text += f"...({len(data)} bytes)"
    return text


def _entries_arg(raw: str) -> int:
    try:
        n = int(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid integer: {raw!r}") from None
    if not 0 <= n <= MAX_ENTRIES:
        raise argparse.ArgumentTypeError(f"must be between 0 and {MAX_ENTRIES}")
    return n


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lsmkv", description="Inspect lsm-kv files.")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="print SSTable metadata and its first entries")
    inspect.add_argument("file", help="path to an .sst file")
    inspect.add_argument(
        "--entries",
        type=_entries_arg,
        default=DEFAULT_ENTRIES,
        metavar="N",
        help=f"number of entries to print, 0..{MAX_ENTRIES} (default: {DEFAULT_ENTRIES})",
    )

    verify = sub.add_parser("verify", help="check every checksum of an SSTable")
    verify.add_argument("file", help="path to an .sst file")
    return parser


def _inspect(path: str, entries: int) -> None:
    with SSTableReader(path) as reader:
        min_key, max_key = reader.min_key, reader.max_key
        print(f"file: {path}")
        print(f"entries: {len(reader)}")
        print(f"blocks: {reader.num_blocks}")
        print(f"min key: {'-' if min_key is None else format_bytes(min_key)}")
        print(f"max key: {'-' if max_key is None else format_bytes(max_key)}")
        for key, value in itertools.islice(reader.items(), entries):
            shown = "<tombstone>" if value is None else format_bytes(value)
            print(f"{format_bytes(key)} => {shown}")


def _verify(path: str) -> None:
    with SSTableReader(path) as reader:
        # Iterating reads every block, which checks each block CRC and ordering.
        count = sum(1 for _ in reader.items())
    print(f"ok: {count} entries")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``lsmkv`` command line and return the process exit code."""
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "inspect":
            _inspect(args.file, args.entries)
        else:
            _verify(args.file)
    except CorruptionError as exc:
        print(f"corrupt sstable: {exc}", file=sys.stderr)
        return _EXIT_CORRUPT
    except OSError as exc:
        print(f"lsmkv: cannot read {args.file}: {exc.strerror or exc}", file=sys.stderr)
        return _EXIT_USAGE
    return _EXIT_OK
