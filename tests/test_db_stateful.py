"""Stateful model test: a real ``DB`` is driven alongside a plain ``dict``."""

import shutil
import tempfile
from pathlib import Path

import hypothesis.strategies as st
from hypothesis import settings
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    invariant,
    precondition,
    rule,
)

from lsmkv import DB
from lsmkv.manifest import table_path

_MEMTABLE_BYTES = 512
_BLOCK_SIZE = 128

# A small pool makes overwrites, deletes of live keys and range hits frequent.
_POOL = [bytes([c]) for c in b"abcdefghij"] + [bytes([c, d]) for c in b"abc" for d in b"xyz"]
_KEYS = st.one_of(
    st.sampled_from(_POOL),
    st.sampled_from(_POOL),
    st.sampled_from(_POOL),
    st.binary(min_size=1, max_size=4),
)
_VALUES = st.binary(max_size=64)
_BOUNDS = st.one_of(st.none(), _KEYS)


class DBMachine(RuleBasedStateMachine):
    """Applies random operations to a ``DB`` and a ``dict`` and compares them."""

    def __init__(self) -> None:
        super().__init__()
        self.model: dict[bytes, bytes] = {}
        self.directory = Path(tempfile.mkdtemp(prefix="lsmkv-stateful-"))
        self.trash: list[Path] = [self.directory]
        self.compaction_trigger = 2
        self.db: DB | None = None

    @initialize(trigger=st.integers(min_value=2, max_value=4))
    def open_db(self, trigger: int) -> None:
        """Choose the compaction trigger once and open the database."""
        self.compaction_trigger = trigger
        self.db = self._open(self.directory)

    def _open(self, directory: Path) -> DB:
        return DB.open(
            directory,
            memtable_bytes=_MEMTABLE_BYTES,
            block_size=_BLOCK_SIZE,
            compaction_trigger=self.compaction_trigger,
        )

    @property
    def live(self) -> DB:
        assert self.db is not None
        return self.db

    @rule(key=_KEYS, value=_VALUES)
    def put(self, key: bytes, value: bytes) -> None:
        self.live.put(key, value)
        self.model[key] = value

    @rule(key=_KEYS)
    def delete(self, key: bytes) -> None:
        self.live.delete(key)
        self.model.pop(key, None)

    @precondition(lambda self: bool(self.model))
    @rule(data=st.data())
    def delete_existing(self, data: st.DataObject) -> None:
        key = data.draw(st.sampled_from(sorted(self.model)))
        self.live.delete(key)
        del self.model[key]

    @rule(key=_KEYS)
    def get(self, key: bytes) -> None:
        assert self.live.get(key) == self.model.get(key)

    @rule(start=_BOUNDS, end=_BOUNDS)
    def scan(self, start: bytes | None, end: bytes | None) -> None:
        expected = [
            (k, v)
            for k, v in sorted(self.model.items())
            if (start is None or k >= start) and (end is None or k < end)
        ]
        assert list(self.live.scan(start, end)) == expected

    @rule()
    def flush(self) -> None:
        self.live.flush()

    @rule()
    def compact(self) -> None:
        self.live.compact()

    @rule()
    def reopen(self) -> None:
        """Close cleanly and open the same directory again."""
        self.live.close()
        self.db = self._open(self.directory)

    @rule()
    def crash_reopen(self) -> None:
        """Copy the directory without closing, as a crash would leave it."""
        crashed = Path(tempfile.mkdtemp(prefix="lsmkv-stateful-"))
        self.trash.append(crashed)
        shutil.copytree(self.directory, crashed, dirs_exist_ok=True)
        self.live.close()
        self.directory = crashed
        self.db = self._open(crashed)

    @invariant()
    def no_stray_files(self) -> None:
        """No temporary files linger and every table on disk is live."""
        if self.db is None:
            return
        assert not list(self.directory.glob("*.tmp"))
        on_disk = {p.name for p in self.directory.glob("*.sst")}
        live = {table_path(self.directory, i).name for i in self.db.table_ids}
        assert on_disk == live

    def teardown(self) -> None:
        if self.db is not None:
            self.db.close()
        for path in self.trash:
            shutil.rmtree(path, ignore_errors=True)


# Inherit max_examples (and the rest) from the active profile.
DBMachine.TestCase.settings = settings(stateful_step_count=50, deadline=None)
TestDBStateful = DBMachine.TestCase
