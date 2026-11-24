"""Crash the engine at every file-system call of a workload and check recovery.

Each workload is first run once with a counting-only injector to learn the
number ``C`` of counted I/O calls (see :mod:`faults`). It is then replayed ``C``
times in a fresh directory, the ``N``-th run crashing at the ``N``-th call. The
files left behind are copied (so the dead instance's lock never gets in the
way), the copy is reopened without faults and must hold exactly the
acknowledged operations: every operation that returned is present, and the one
in flight is either fully applied or not at all. Recovery must also remove all
temporary and unlisted files, and the reopened database must stay writable.

A second pass crashes only at ``write`` calls and lets half of the data reach
the file first (a torn write).

The workload includes ``DB.open`` on the empty directory, so the initial
manifest and log creation are crash points too.

What is modelled: a *process* crash. The operating system survives, so bytes
handed to ``write`` stay visible even if never fsynced, while bytes still in
Python's userspace buffer are lost. What is not modelled: *power loss*, which
would additionally drop unsynced data and reorder unsynced directory entries.
That needs a fault layer that tracks which bytes were fsynced and is future work.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest

from faults import FaultInjector, SimulatedCrash, injected
from lsmkv import DB
from lsmkv.manifest import table_path

Kind = Literal["put", "delete", "flush", "compact"]
Model = dict[bytes, bytes]


@dataclass(frozen=True)
class Op:
    """One step of a workload."""

    kind: Kind
    key: bytes = b""
    value: bytes = b""


@dataclass(frozen=True)
class Workload:
    """A named sequence of operations plus the options to open the database with."""

    name: str
    ops: tuple[Op, ...]
    memtable_bytes: int
    sync: bool = False


def _put(key: str, value: str) -> Op:
    return Op("put", key.encode(), value.encode())


def _delete(key: str) -> Op:
    return Op("delete", key.encode())


_FLUSH = Op("flush")
_COMPACT = Op("compact")

_AUTO = Workload(
    "auto-flush-and-compaction",
    (
        *(_put(f"k{i:02d}", f"value-{i}") for i in range(5)),
        _delete("k03"),
        _put("k01", "rewritten"),
        *(_put(f"k{i:02d}", f"second-{i}") for i in range(5, 8)),
        _delete("k06"),
        _delete("k00"),
        _FLUSH,
        _COMPACT,
        _put("k20", "after-compaction"),
    ),
    memtable_bytes=48,
)

_MANUAL = Workload(
    "manual-flush-and-compact",
    (
        _put("a", "1"),
        _put("b", "2"),
        _put("c", "3"),
        _FLUSH,
        _put("d", "4"),
        _delete("a"),
        _FLUSH,
        _put("b", "22"),
        _FLUSH,
        _COMPACT,
        _delete("b"),
        _delete("c"),
        _delete("d"),
        _FLUSH,
        _COMPACT,
        _put("e", "5"),
    ),
    memtable_bytes=1 << 20,
)

_SYNCED = Workload(
    "synced-log",
    (
        _put("x", "1"),
        _put("y", "2"),
        _delete("x"),
        _FLUSH,
        _put("z", "3"),
        _put("y", "4"),
        _FLUSH,
        _COMPACT,
    ),
    memtable_bytes=1 << 20,
    sync=True,
)

WORKLOADS = (_AUTO, _MANUAL, _SYNCED)

_BLOCK_SIZE = 32
_COMPACTION_TRIGGER = 2
#: Above this many crash points a run is strided (every ``ceil(C / limit)``-th).
_MAX_POINTS = 1500


def _open(path: Path, workload: Workload) -> DB:
    return DB.open(
        path,
        memtable_bytes=workload.memtable_bytes,
        sync=workload.sync,
        block_size=_BLOCK_SIZE,
        compaction_trigger=_COMPACTION_TRIGGER,
    )


def _apply(model: Model, op: Op) -> None:
    if op.kind == "put":
        model[op.key] = op.value
    elif op.kind == "delete":
        model.pop(op.key, None)


def _model_after(ops: tuple[Op, ...], count: int) -> Model:
    model: Model = {}
    for op in ops[:count]:
        _apply(model, op)
    return model


def _run_op(db: DB, op: Op) -> None:
    if op.kind == "put":
        db.put(op.key, op.value)
    elif op.kind == "delete":
        db.delete(op.key)
    elif op.kind == "flush":
        db.flush()
    else:
        db.compact()


@dataclass
class Outcome:
    """What a (possibly crashed) run left behind."""

    directory: Path
    acknowledged: int
    in_flight: bool
    crashed: bool
    calls: list[str]


def _run(workload: Workload, root: Path, injector: FaultInjector) -> Outcome:
    """Run ``workload`` under ``injector`` and snapshot the directory afterwards.

    The snapshot is a copy made after the crash, so the dead instance (which
    still holds the lock) is irrelevant; the instance is closed only to
    release its descriptors once the copy exists. The database always lives in
    ``root/db`` so that file-system call labels do not depend on the run.
    """
    directory = root / "db"
    directory.mkdir(parents=True)
    db: DB | None = None
    acknowledged = 0
    in_flight = False
    crashed = False
    with injected(injector):
        try:
            db = _open(directory, workload)
            for op in workload.ops:
                in_flight = True
                _run_op(db, op)
                in_flight = False
                acknowledged += 1
        except SimulatedCrash:
            crashed = True
    snapshot = root / "recovered"
    shutil.copytree(directory, snapshot)
    if db is not None:
        db.close()
    return Outcome(snapshot, acknowledged, in_flight and crashed, crashed, injector.calls)


def _check_recovery(workload: Workload, outcome: Outcome) -> None:
    """Reopen the snapshot and assert it matches an allowed model."""
    allowed = [_model_after(workload.ops, outcome.acknowledged)]
    if outcome.in_flight:
        allowed.append(_model_after(workload.ops, outcome.acknowledged + 1))
    keys = {op.key for op in workload.ops if op.key} | {b"never-written"}
    path = outcome.directory

    with _open(path, workload) as db:
        got = list(db.scan())
        expected = next((m for m in allowed if sorted(m.items()) == got), None)
        assert expected is not None, f"scan {got!r} matches none of {allowed!r}"
        for key in sorted(keys):
            assert db.get(key) == expected.get(key), f"get({key!r})"
        low, high = b"b", b"k05"
        assert list(db.scan(low, high)) == [
            (k, v) for k, v in sorted(expected.items()) if low <= k < high
        ]

        names = {entry.name for entry in path.iterdir()}
        assert not [n for n in names if n.endswith(".tmp")], f"temporary files left: {names}"
        tables = {n for n in names if n.endswith(".sst")}
        assert tables == {table_path(path, i).name for i in db.table_ids}, (
            f"unlisted tables: {names}"
        )

        db.put(b"after-recovery", b"ok")
        db.flush()
        db.compact()
        assert db.get(b"after-recovery") == b"ok"
        expected = {**expected, b"after-recovery": b"ok"}
        assert list(db.scan()) == sorted(expected.items())

    with _open(path, workload) as db:
        assert list(db.scan()) == sorted(expected.items()), "state changed across a clean reopen"


def _points(calls: list[str], *, writes_only: bool) -> list[int]:
    """The 1-based crash points to try, strided if there are too many."""
    candidates = [
        n for n, label in enumerate(calls, 1) if not writes_only or label.startswith("write:")
    ]
    stride = -(-len(candidates) // _MAX_POINTS)
    return candidates[::stride]


def _reference(workload: Workload, tmp_path: Path) -> Outcome:
    """Run ``workload`` to completion under a counting-only injector."""
    outcome = _run(workload, tmp_path / "reference", FaultInjector())
    assert not outcome.crashed
    assert outcome.acknowledged == len(workload.ops)
    return outcome


@pytest.mark.parametrize("workload", WORKLOADS, ids=lambda w: w.name)
def test_reference_run_covers_flush_and_compaction(workload: Workload, tmp_path: Path) -> None:
    calls = _reference(workload, tmp_path).calls
    assert any(c.startswith("replace:") and c.endswith(".sst") for c in calls), "no flush"
    assert any(c.startswith("replace:MANIFEST.tmp") for c in calls), "no manifest commit"
    assert any(c.startswith("unlink:") and c.endswith(".sst") for c in calls), "no compaction"
    assert any(c.startswith("truncate:wal.log") for c in calls), "no log reset"
    assert any(c.startswith("write:") for c in calls)
    assert any(c.startswith("fsync") for c in calls)
    assert any(c.startswith("os.open") for c in calls)


def test_crash_leaves_orphans_for_recovery_to_remove(tmp_path: Path) -> None:
    """Guards the harness: crashes really do leave half-finished files behind."""
    calls = _reference(_MANUAL, tmp_path / "ref").calls
    publish = next(n for n, c in enumerate(calls, 1) if c.startswith("replace:000001.sst.tmp"))
    outcome = _run(_MANUAL, tmp_path / "flush", FaultInjector(publish))
    assert (outcome.directory / "000001.sst.tmp").exists()
    assert outcome.in_flight

    cleanup = next(
        n for n, c in enumerate(calls, 1) if c.startswith("unlink:") and c.endswith(".sst")
    )
    outcome = _run(_MANUAL, tmp_path / "compact", FaultInjector(cleanup))
    names = {p.name for p in outcome.directory.iterdir() if p.suffix == ".sst"}
    with _open(outcome.directory, _MANUAL) as db:
        assert len(names) > len(db.table_ids)
    assert {p.name for p in outcome.directory.iterdir() if p.suffix == ".sst"} != names


def test_crash_free_run_recovers_everything(tmp_path: Path) -> None:
    for workload in WORKLOADS:
        outcome = _reference(workload, tmp_path / workload.name)
        _check_recovery(workload, outcome)


@pytest.mark.parametrize("workload", WORKLOADS, ids=lambda w: w.name)
@pytest.mark.parametrize("torn", [False, True], ids=["before-call", "torn-write"])
def test_recovery_after_crash_at_every_io_call(
    workload: Workload, torn: bool, tmp_path: Path
) -> None:
    calls = _reference(workload, tmp_path).calls
    failures: list[str] = []
    for n in _points(calls, writes_only=torn):
        outcome = _run(workload, tmp_path / f"run-{n}", FaultInjector(n, torn=torn))
        assert outcome.crashed, (
            f"call {n} ({calls[n - 1]}) never happened: workload is not deterministic"
        )
        assert outcome.calls == calls[:n], f"call sequence diverged before call {n}"
        try:
            _check_recovery(workload, outcome)
        except AssertionError as exc:
            failures.append(f"N={n} ({calls[n - 1]}, {outcome.acknowledged} acked): {exc}")
    assert not failures, "\n".join(failures)
