"""Tests for the ``lsmkv`` command line."""

from __future__ import annotations

import contextlib
import errno
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from lsmkv.cli import MAX_DISPLAY_BYTES, MAX_ENTRIES, format_bytes, format_path, main
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


@pytest.mark.parametrize(
    ("raw", "shown"),
    [
        ("plain/dir/table.sst", "plain/dir/table.sst"),
        ("", ""),
        ("a\nb", "a\\nb"),
        ("a\x1b[31mred", "a\\x1b[31mred"),
        ("café", "caf\\xe9"),
        ("snow☃", "snow\\u2603"),
        ("bad\udcff", "bad\\udcff"),
        ("del\x7f", "del\\x7f"),
        ("nul\x00", "nul\\x00"),
    ],
)
def test_format_path_escapes_unsafe_characters(raw: str, shown: str) -> None:
    assert format_path(raw) == shown


@given(st.text())
def test_format_path_is_always_printable_ascii(raw: str) -> None:
    shown = format_path(raw)
    assert all(0x20 <= ord(c) <= 0x7E for c in shown)


@pytest.mark.parametrize("command", ["inspect", "verify"])
def test_odd_paths_are_reported_safely(command: str, capsys: pytest.CaptureFixture[str]) -> None:
    for odd in ("no\nsuch\x1b[2Jfile\udcff.sst", "nul\x00byte.sst"):
        assert main([command, odd]) == 2
        err = capsys.readouterr().err
        assert err.startswith("lsmkv: cannot read ")
        assert err.count("\n") == 1
        assert "\x1b" not in err
        assert err.isascii()


def test_inspect_prints_the_path_safely(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "café\ttable.sst"
    write_sstable(path, ENTRIES)
    assert main(["inspect", str(path)]) == 0
    first = capsys.readouterr().out.splitlines()[0]
    assert first.endswith("caf\\xe9\\ttable.sst")
    assert first.isascii()


class _ClosedPipe:
    """A stdout whose reader has gone away, backed by a real descriptor."""

    def __init__(self) -> None:
        read_end, write_end = os.pipe()
        os.close(read_end)
        self.fd = write_end
        self.stream = os.fdopen(write_end, "w", closefd=False)

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self.stream.close()
        os.close(self.fd)


def test_broken_pipe_exits_quietly_and_redirects_stdout(
    sst: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pipe = _ClosedPipe()
    try:
        monkeypatch.setattr(sys, "stdout", pipe.stream)
        assert main(["inspect", str(sst)]) == 0
        monkeypatch.undo()
        assert capsys.readouterr().err == ""
        # the descriptor now leads to the null device, so a later flush is silent
        assert os.stat(pipe.fd).st_rdev == os.stat(os.devnull).st_rdev
        pipe.stream.flush()
    finally:
        pipe.close()


def test_broken_pipe_during_print_is_not_an_error(
    sst: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def explode(*args: object, **kwargs: object) -> None:
        raise BrokenPipeError(errno.EPIPE, "Broken pipe")

    monkeypatch.setattr("builtins.print", explode)
    assert main(["verify", str(sst)]) == 0
    assert capsys.readouterr().err == ""


def test_broken_pipe_in_a_real_process_leaves_no_message(sst: Path) -> None:
    script = (
        "import os, sys\n"
        "from lsmkv.cli import main\n"
        "r, w = os.pipe()\n"
        "os.close(r)\n"
        "os.dup2(w, 1)\n"
        "sys.exit(main(['inspect', sys.argv[1]]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(sst)],
        check=False,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == b""


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
