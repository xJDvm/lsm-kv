"""Micro-benchmarks for lsmkv.

Usage::

    uv run python benchmarks/bench.py [--n N] [--value-size B] [--seed S] [--sync]

Prints a Markdown table with the wall-clock time and throughput of each
workload. Every run happens in a temporary directory that is removed on exit.
"""

from __future__ import annotations

import argparse
import platform
import random
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from lsmkv import DB

MAX_N = 10_000_000
MAX_VALUE_SIZE = 1_048_576
_NOT_APPLICABLE = chr(0x2013)  # en dash, shown where a column does not apply


@dataclass(frozen=True)
class Result:
    """Timing of one workload."""

    name: str
    ops: int | None
    seconds: float

    @property
    def ops_per_second(self) -> float | None:
        """Throughput; ``None`` for one-shot workloads, infinite if no time passed."""
        if self.ops is None:
            return None
        return self.ops / self.seconds if self.seconds > 0 else float("inf")


def _key(i: int) -> bytes:
    return b"key%010d" % i


def _timed(name: str, ops: int, action: Callable[[], object]) -> Result:
    start = time.perf_counter()
    action()
    return Result(name, ops, time.perf_counter() - start)


def _write_workload(
    name: str, path: Path, order: Sequence[int], value: bytes, sync: bool
) -> tuple[Result, int]:
    """Put every index in ``order`` into a fresh DB; return timing and table count."""
    with DB.open(path, sync=sync) as db:

        def run() -> None:
            for i in order:
                db.put(_key(i), value)

        result = _timed(name, len(order), run)
        return result, len(db.table_ids)


def _positive_int(low: int, high: int) -> Callable[[str], int]:
    def parse(text: str) -> int:
        try:
            number = int(text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from exc
        if not low <= number <= high:
            raise argparse.ArgumentTypeError(f"must be between {low} and {high}")
        return number

    return parse


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark lsmkv.")
    parser.add_argument("--n", type=_positive_int(1, MAX_N), default=100_000, help="number of keys")
    parser.add_argument(
        "--value-size",
        type=_positive_int(0, MAX_VALUE_SIZE),
        default=100,
        help="value size in bytes",
    )
    parser.add_argument("--seed", type=int, default=42, help="seed for the shuffles and values")
    parser.add_argument("--sync", action="store_true", help="fsync the log on every write")
    return parser


def _table(results: Sequence[Result]) -> str:
    lines = ["| Workload | Ops | Seconds | Ops/s |", "| --- | ---: | ---: | ---: |"]
    for r in results:
        rate = r.ops_per_second
        ops = _NOT_APPLICABLE if r.ops is None else str(r.ops)
        rate_text = _NOT_APPLICABLE if rate is None else f"{rate:,.0f}"
        lines.append(f"| {r.name} | {ops} | {r.seconds:.3f} | {rate_text} |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run every workload and print the report; return the process exit code."""
    args = _parser().parse_args(argv)
    n: int = args.n
    sync: bool = args.sync
    # A seeded non-cryptographic RNG is right for a benchmark: runs must be
    # reproducible and nothing here is security sensitive.
    rng = random.Random(args.seed)
    value = rng.randbytes(args.value_size)
    sequential = list(range(n))
    shuffled = list(sequential)
    rng.shuffle(shuffled)
    missing = [n + i for i in shuffled]

    results: list[Result] = []
    notes: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        result, tables = _write_workload("sequential puts", root / "seq", sequential, value, sync)
        results.append(result)
        notes.append(f"SSTables after sequential puts: {tables}")

        result, tables = _write_workload("random puts", root / "rand", shuffled, value, sync)
        results.append(result)
        notes.append(f"SSTables after random puts: {tables}")

        with DB.open(root / "rand", sync=sync) as db:
            db.flush()

            def read_existing() -> None:
                for i in shuffled:
                    if db.get(_key(i)) is None:
                        raise AssertionError("existing key not found")

            def read_missing() -> None:
                for i in missing:
                    if db.get(_key(i)) is not None:
                        raise AssertionError("missing key found")

            def full_scan() -> None:
                count = sum(1 for _ in db.scan())
                if count != n:
                    raise AssertionError(f"scan returned {count} entries, expected {n}")

            results.append(_timed("random reads (existing)", n, read_existing))
            results.append(_timed("random reads (missing)", n, read_missing))
            results.append(_timed("full scan", n, full_scan))
            # Leave a non-empty memtable behind so reopening replays the log.
            # Keep it well under the default 4 MiB memtable so no flush occurs.
            tail = max(1, min(1000, 1_000_000 // (args.value_size + 20)))
            for i in range(tail):
                db.put(_key(n + i), value)

        start = time.perf_counter()
        reopened = DB.open(root / "rand", sync=sync)
        elapsed = time.perf_counter() - start
        with reopened:
            open_tables = len(reopened.table_ids)
        name = f"reopen (replay {tail} WAL records + open {open_tables} tables)"
        results.append(Result(name, None, elapsed))

    print(
        f"Python {platform.python_version()} ({platform.python_implementation()}), "
        f"{platform.machine()}, {platform.system()}; "
        f"n={n}, value_size={args.value_size}, seed={args.seed}, sync={sync}"
    )
    print()
    print(_table(results))
    print()
    for note in notes:
        print(note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
