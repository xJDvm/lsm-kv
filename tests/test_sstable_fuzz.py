"""Fuzz and exhaustive-corruption tests for the SSTable reader.

The reader must either return exactly the data that was written or raise
``CorruptionError``; no other exception and no silently wrong answer.
"""

import contextlib
import tempfile
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from lsmkv.errors import CorruptionError
from lsmkv.sstable import SSTableReader, write_sstable
from lsmkv.types import MISSING, Entry

ENTRIES: list[Entry] = [
    (b"alpha", b"1"),
    (b"bravo", None),
    (b"charlie", b""),
    (b"delta", b"d" * 20),
    (b"echo", b"5"),
    (b"foxtrot", None),
    (b"golf", b"gg"),
    (b"hotel", b"h" * 10),
]
ABSENT = [b"a", b"bravo2", b"zulu", b"\xff"]


@pytest.fixture(scope="module")
def valid_bytes(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    p = tmp_path_factory.mktemp("sst") / "valid.sst"
    write_sstable(p, ENTRIES, block_size=24)
    return p.read_bytes()


def _exercise(path: Path) -> None:
    """Open ``path`` and touch every read path; only CorruptionError may escape."""
    with SSTableReader(path) as r:
        _ = (len(r), r.min_key, r.max_key)
        list(r.items())
        list(r.items(b"c", b"g"))
        for k, _ in ENTRIES:
            r.get(k)
        for k in ABSENT:
            r.get(k)


def _exercise_expecting_correct(path: Path) -> None:
    """Like ``_exercise`` but asserts that any successful answer is the true one."""
    with SSTableReader(path) as r:
        assert len(r) == len(ENTRIES)
        assert list(r.items()) == ENTRIES
        for k, v in ENTRIES:
            assert r.get(k) == v
        for k in ABSENT:
            assert r.get(k) is MISSING


def test_valid_file_is_accepted(tmp_path: Path, valid_bytes: bytes) -> None:
    p = tmp_path / "t.sst"
    p.write_bytes(valid_bytes)
    _exercise_expecting_correct(p)
    with SSTableReader(p) as r:
        assert r.num_blocks > 2


def test_truncation_at_every_offset(tmp_path: Path, valid_bytes: bytes) -> None:
    p = tmp_path / "t.sst"
    for cut in range(len(valid_bytes)):
        p.write_bytes(valid_bytes[:cut])
        with pytest.raises(CorruptionError):
            _exercise(p)


def test_flipping_every_byte(tmp_path: Path, valid_bytes: bytes) -> None:
    p = tmp_path / "t.sst"
    for i in range(len(valid_bytes)):
        mutated = bytearray(valid_bytes)
        mutated[i] ^= 0xFF
        p.write_bytes(mutated)
        with contextlib.suppress(CorruptionError):
            _exercise_expecting_correct(p)


def test_appended_garbage_is_rejected(tmp_path: Path, valid_bytes: bytes) -> None:
    p = tmp_path / "t.sst"
    p.write_bytes(valid_bytes + b"\x00")
    with pytest.raises(CorruptionError):
        _exercise(p)


@given(data=st.binary(max_size=2048))
def test_arbitrary_bytes_only_raise_corruption(data: bytes) -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.sst"
        p.write_bytes(data)
        with contextlib.suppress(CorruptionError):
            _exercise(p)


@given(
    edits=st.lists(
        st.tuples(st.integers(min_value=0), st.integers(min_value=1, max_value=255)),
        min_size=1,
        max_size=4,
    ),
)
def test_random_mutations_never_give_wrong_answers(
    edits: list[tuple[int, int]], valid_bytes: bytes
) -> None:
    mutated = bytearray(valid_bytes)
    for pos, mask in edits:
        mutated[pos % len(mutated)] ^= mask
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.sst"
        p.write_bytes(mutated)
        with contextlib.suppress(CorruptionError):
            _exercise_expecting_correct(p)
