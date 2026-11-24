"""Exhaustive crash-recovery tests: truncation at every byte, flip of every bit."""

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from lsmkv.types import Entry
from lsmkv.wal import WAL, recover_wal, scan_wal

OPS: list[Entry] = [
    (b"alpha", b"1"),
    (b"beta", b""),
    (b"alpha", None),
    (b"gamma", b"x" * 40),
    (b"k", b"v"),
    (b"delta", None),
    (b"epsilon", b"\x00\xff" * 5),
    (b"beta", b"again"),
    (b"z" * 20, b"y" * 3),
    (b"omega", b"end"),
]


def _build(path: Path) -> tuple[bytes, list[int]]:
    """Write ``OPS`` and return the file bytes and each record's end offset."""
    ends: list[int] = []
    with WAL.open(path) as wal:
        for key, value in OPS:
            if value is None:
                wal.append_delete(key)
            else:
                wal.append_put(key, value)
            wal.flush()
            ends.append(path.stat().st_size)
    return path.read_bytes(), ends


def test_truncation_at_every_offset(tmp_path: Path) -> None:
    data, ends = _build(tmp_path / "wal")
    probe = tmp_path / "probe"
    for cut in range(len(data) + 1):
        probe.write_bytes(data[:cut])
        scan = scan_wal(probe)
        complete = sum(1 for end in ends if end <= cut)
        assert scan.entries == OPS[:complete], cut
        assert scan.valid_length == (ends[complete - 1] if complete else 0)
        assert scan.torn == (cut != scan.valid_length)


def test_recover_repairs_every_truncation(tmp_path: Path) -> None:
    data, ends = _build(tmp_path / "wal")
    probe = tmp_path / "probe"
    for cut in range(0, len(data) + 1, 3):
        probe.write_bytes(data[:cut])
        complete = sum(1 for end in ends if end <= cut)
        assert recover_wal(probe) == OPS[:complete]
        assert probe.stat().st_size == (ends[complete - 1] if complete else 0)
        assert not scan_wal(probe).torn


def test_every_single_bit_flip_yields_strict_prefix(tmp_path: Path) -> None:
    data, ends = _build(tmp_path / "wal")
    starts = [0, *ends[:-1]]
    probe = tmp_path / "probe"
    for bit in range(len(data) * 8):
        byte = bit // 8
        corrupted = bytearray(data)
        corrupted[byte] ^= 1 << (bit % 8)
        probe.write_bytes(corrupted)
        scan = scan_wal(probe)
        index = next(i for i, end in enumerate(ends) if byte < end)
        assert scan.entries == OPS[:index], (byte, bit % 8)
        assert scan.valid_length == starts[index]
        assert scan.torn


@given(tail=st.binary(max_size=300))
def test_arbitrary_tail_after_valid_records(
    tail: bytes, tmp_path_factory: pytest.TempPathFactory
) -> None:
    path = tmp_path_factory.mktemp("tail") / "wal"
    data, _ = _build(path)
    path.write_bytes(data + tail)
    scan = scan_wal(path)
    assert scan.entries[: len(OPS)] == OPS
    assert len(data) <= scan.valid_length <= len(data) + len(tail)


@given(content=st.binary(max_size=600))
def test_arbitrary_file_never_raises(
    content: bytes, tmp_path_factory: pytest.TempPathFactory
) -> None:
    path = tmp_path_factory.mktemp("arb") / "wal"
    path.write_bytes(content)
    scan = scan_wal(path)
    assert 0 <= scan.valid_length <= len(content)
    assert scan.torn == (scan.valid_length < len(content))
