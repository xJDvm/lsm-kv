"""Tests for the ``lsmkv`` command line."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

from lsmkv.cli import MAX_DISPLAY_BYTES, MAX_ENTRIES, format_bytes, main
from lsmkv.sstable import write_sstable
from lsmkv.types import Entry

ENTRIES: list[Entry] = [
    (b"\x00\xffbin", b"\x01\x02"),
    (b"alpha", b"one"),
    (b"beta", None),
    (b"gamma", b"x" * 200),
    (b"line\nbreak", b"tab\tvalue"),
]


@pytest.fixture
def sst(tmp_path: Path) -> Path:
    path = tmp_path / "table.sst"
    write_sstable(path, ENTRIES, block_size=64)
    return path


def test_format_bytes_printable_hex_and_truncation() -> None:
    assert format_bytes(b"hello world~") == "hello world~"
    assert format_bytes(b"") == ""
    assert format_bytes(b"a\x7fb") == "0x61" + "7f" + "62"
    assert format_bytes("é".encode()) == "0xc3a9"
    long = b"y" * (MAX_DISPLAY_BYTES + 1)
    assert format_bytes(long) == "y" * MAX_DISPLAY_BYTES + f"...({MAX_DISPLAY_BYTES + 1} bytes)"
    assert format_bytes(b"z" * MAX_DISPLAY_BYTES) == "z" * MAX_DISPLAY_BYTES


def test_inspect_output(sst: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["inspect", str(sst)]) == 0
    out, err = capsys.readouterr()
    assert err == ""
    lines = out.splitlines()
    assert lines[0] == f"file: {sst}"
    assert lines[1] == "entries: 5"
    assert lines[2].startswith("blocks: ") and int(lines[2].split()[1]) > 1
    assert lines[3] == "min key: 0x00ff62696e"
    assert lines[4] == "max key: 0x6c696e650a627265616b"
    assert lines[5:] == [
        "0x00ff62696e => 0x0102",
        "alpha => one",
        "beta => <tombstone>",
        "gamma => " + "x" * 64 + "...(200 bytes)",
        "0x6c696e650a627265616b => 0x7461620976616c7565",
    ]


def test_inspect_uses_path_as_given(
    sst: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(sst.parent)
    assert main(["inspect", "table.sst"]) == 0
    assert capsys.readouterr().out.startswith("file: table.sst\n")


def test_inspect_entries_limit(sst: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["inspect", str(sst), "--entries", "2"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 5 + 2
    assert main(["inspect", str(sst), "--entries", "0"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 5
    assert main(["inspect", str(sst), "--entries", str(MAX_ENTRIES)]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 5 + len(ENTRIES)


@pytest.mark.parametrize("bad", ["-1", str(MAX_ENTRIES + 1), "ten", "1.5", ""])
def test_inspect_entries_out_of_range(
    sst: Path, bad: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as info:
        main(["inspect", str(sst), "--entries", bad])
    assert info.value.code == 2
    assert "--entries" in capsys.readouterr().err


def test_inspect_empty_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "empty.sst"
    write_sstable(path, [])
    assert main(["inspect", str(path)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[1:] == ["entries: 0", "blocks: 0", "min key: -", "max key: -"]


def test_verify_ok(sst: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["verify", str(sst)]) == 0
    assert capsys.readouterr().out == "ok: 5 entries\n"


def _corrupt_first_byte(path: Path) -> None:
    data = bytearray(path.read_bytes())
    data[0] ^= 0xFF
    path.write_bytes(bytes(data))


def test_verify_detects_corruption(sst: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _corrupt_first_byte(sst)
    assert main(["verify", str(sst)]) == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith("corrupt sstable: ")
    assert "Traceback" not in err


def test_inspect_detects_corruption_in_first_block(
    sst: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _corrupt_first_byte(sst)
    assert main(["inspect", str(sst)]) == 1
    assert capsys.readouterr().err.startswith("corrupt sstable: ")


def test_garbage_file_is_corruption(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "junk.sst"
    path.write_bytes(b"not an sstable")
    assert main(["inspect", str(path)]) == 1
    assert "corrupt sstable: " in capsys.readouterr().err


@pytest.mark.parametrize("command", ["inspect", "verify"])
def test_missing_file_exits_2_without_traceback(
    command: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([command, str(tmp_path / "nope.sst")]) == 2
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith("lsmkv: cannot read ")
    assert "Traceback" not in err


def test_directory_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["verify", str(tmp_path)]) == 2
    assert "Traceback" not in capsys.readouterr().err


def test_usage_errors(capsys: pytest.CaptureFixture[str]) -> None:
    for argv in ([], ["bogus"], ["inspect"], ["verify"]):
        with pytest.raises(SystemExit) as info:
            main(argv)
        assert info.value.code == 2
    assert "usage:" in capsys.readouterr().err


def test_python_dash_m_entry_point(
    sst: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["lsmkv", "verify", str(sst)])
    with pytest.raises(SystemExit) as info:
        runpy.run_module("lsmkv", run_name="__main__")
    assert info.value.code == 0
    assert capsys.readouterr().out == "ok: 5 entries\n"
