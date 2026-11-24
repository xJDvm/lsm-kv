"""A deterministic fault injector that simulates a process crash at an I/O call.

Every call the engine makes that can change what is on disk is counted: the
``os`` functions ``fsync``, ``replace``, ``rename``, ``unlink``, ``remove``,
``truncate`` and ``open`` (which also covers ``Path.unlink``), and the
creation, ``write``, ``flush`` and ``truncate`` of every file the engine opens
for writing. When the counter reaches ``crash_at`` the call raises
:class:`SimulatedCrash` *instead of running*, and from then on every counted
call raises too, so cleanup handlers cannot repair anything: the process is
gone. Closing a file after the crash discards its unflushed buffer, as a dying
process would, but still releases the descriptor so a test run does not leak.

With ``torn=True`` a crash on a ``write`` first lets the first half of the
data reach the operating system, modelling a torn write.
"""

from __future__ import annotations

import builtins
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Any

import pytest

import lsmkv.manifest
import lsmkv.sstable
import lsmkv.wal

__all__ = ["FaultInjector", "SimulatedCrash", "injected"]

_WRITE_MODE_CHARS = frozenset("wax+")


class SimulatedCrash(BaseException):
    """The process died. A ``BaseException`` so no ``except Exception`` swallows it."""


def _name(arg: object) -> str:
    """A short label for a path-like argument (file descriptors have no name)."""
    return Path(arg).name if isinstance(arg, (str, os.PathLike)) else "fd"


class FaultInjector:
    """Counts I/O calls and crashes at the ``crash_at``-th (1-based).

    Attributes:
        calls: Label of every counted call so far, in order.
        crashed: Whether the crash point has been reached.
    """

    def __init__(self, crash_at: int | None = None, *, torn: bool = False) -> None:
        self._crash_at = crash_at
        self._torn = torn
        self.calls: list[str] = []
        self.crashed = False
        self._disarmed = False

    def disarm(self) -> None:
        """Stop counting and crashing, so files wrapped earlier pass calls through."""
        self._crash_at = None
        self.crashed = False
        self._disarmed = True

    def tick(self, label: str, torn_effect: Callable[[], None] | None = None) -> None:
        """Count one call; raise :class:`SimulatedCrash` if it is the crash point."""
        if self._disarmed:
            return
        if self.crashed:
            raise SimulatedCrash(label)
        self.calls.append(label)
        if len(self.calls) == self._crash_at:
            self.crashed = True
            if self._torn and torn_effect is not None:
                torn_effect()
            raise SimulatedCrash(label)


class _FaultyFile:
    """Wraps a file opened for writing so that its mutating calls are counted."""

    def __init__(self, inner: Any, injector: FaultInjector, name: str) -> None:
        self._inner = inner
        self._injector = injector
        self._name = name

    def write(self, data: bytes | bytearray) -> int:
        def half() -> None:
            self._inner.write(bytes(data[: len(data) // 2]))
            self._inner.flush()

        self._injector.tick(f"write:{self._name}", half)
        return int(self._inner.write(data))

    def flush(self) -> None:
        self._injector.tick(f"flush:{self._name}")
        self._inner.flush()

    def truncate(self, size: int | None = None) -> int:
        self._injector.tick(f"truncate:{self._name}")
        return int(self._inner.truncate(size))

    def close(self) -> None:
        if self._injector.crashed and not self._inner.closed:
            self._inner.detach().close()  # drop buffered bytes: the process is gone
        else:
            self._inner.close()

    def __enter__(self) -> _FaultyFile:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._inner, attr)


@contextmanager
def injected(injector: FaultInjector) -> Iterator[FaultInjector]:
    """Route the engine's file-system calls through ``injector`` for the block."""
    real_open = builtins.open

    def fake_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if not _WRITE_MODE_CHARS & set(mode):
            return real_open(file, mode, *args, **kwargs)
        injector.tick(f"open:{_name(file)}")
        return _FaultyFile(real_open(file, mode, *args, **kwargs), injector, _name(file))

    def counted(label: str, real: Callable[..., Any], *, named: bool) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if named:
                injector.tick(f"{label}:" + "->".join(_name(a) for a in args[:2]))
            else:
                injector.tick(label)
            return real(*args, **kwargs)

        return wrapper

    with pytest.MonkeyPatch.context() as patch:
        for module in (lsmkv.wal, lsmkv.sstable, lsmkv.manifest):
            patch.setattr(module, "open", fake_open, raising=False)
        patch.setattr(os, "fsync", counted("fsync", os.fsync, named=False))
        patch.setattr(os, "open", counted("os.open", os.open, named=True))
        for fn in ("replace", "rename", "unlink", "remove", "truncate"):
            patch.setattr(os, fn, counted(fn, getattr(os, fn), named=True))
        try:
            yield injector
        finally:
            injector.disarm()
