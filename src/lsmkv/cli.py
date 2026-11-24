"""Command-line tools for inspecting lsm-kv files.

Exit codes: ``0`` success (also when the output pipe is closed early, as in
``lsmkv inspect f | head``), ``1`` the file is corrupt, ``2`` usage error or the
file cannot be opened (missing, unreadable, a directory, ...).
"""

from __future__ import annotations

import argparse
import itertools
import os
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


def format_path(path: str) -> str:
    """Render a file path for humans without raising or emitting control characters.

    Printable ASCII is kept as-is; every other character (control characters,
    non-ASCII text and the surrogates Python uses for undecodable bytes) is
    shown as a backslash escape such as ``\\n``, ``\\xe9`` or ``\\udcff``.
    """
    return "".join(c if 0x20 <= ord(c) <= 0x7E else ascii(c)[1:-1] for c in path)


def _silence_stdout() -> None:
    """Point the stdout file descriptor at the null device.

    After a broken pipe the buffered output can never be delivered, and the
    interpreter would report a second error when it flushes stdout at exit.
    Redirecting the descriptor, as the Python documentation recommends, lets
    that final flush succeed silently.
    """
    try:
        fd = sys.stdout.fileno()
    except (OSError, ValueError):  # no real descriptor behind sys.stdout
        return
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, fd)
    finally:
        os.close(null)


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
        print(f"file: {format_path(path)}")
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
        sys.stdout.flush()  # surface a closed pipe here, not at interpreter exit
    except BrokenPipeError:
        # The reader went away (``lsmkv inspect f | head``); that is not an error.
        _silence_stdout()
        return _EXIT_OK
    except CorruptionError as exc:
        print(f"corrupt sstable: {exc}", file=sys.stderr)
        return _EXIT_CORRUPT
    except (OSError, ValueError) as exc:  # ValueError: NUL byte in the path
        reason = getattr(exc, "strerror", None) or str(exc) or type(exc).__name__
        print(f"lsmkv: cannot read {format_path(args.file)}: {reason}", file=sys.stderr)
        return _EXIT_USAGE
    return _EXIT_OK
