"""Tests for the directory lock that keeps two instances off one database."""

from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from lsmkv import DB, LockedError, LSMError

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="flock is POSIX only")


def test_second_open_in_same_process_fails(tmp_path: Path) -> None:
    with DB.open(tmp_path) as first:
        first.put(b"k", b"v")
        with pytest.raises(LockedError, match="locked by another process"):
            DB.open(tmp_path)
        assert first.get(b"k") == b"v"


def test_locked_error_is_an_lsm_error() -> None:
    assert issubclass(LockedError, LSMError)


def test_reopen_after_close_works(tmp_path: Path) -> None:
    DB.open(tmp_path).close()
    with DB.open(tmp_path) as db:
        db.put(b"k", b"v")
    with DB.open(tmp_path) as db:
        assert db.get(b"k") == b"v"


def test_lock_is_released_by_a_failed_flush_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = DB.open(tmp_path)
    db.put(b"k", b"v")

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("simulated")

    with monkeypatch.context() as patch:
        patch.setattr("lsmkv.db.Manifest.commit", staticmethod(boom))
        with pytest.raises(OSError, match="simulated"):
            db.flush()
    with DB.open(tmp_path) as reopened:
        assert reopened.get(b"k") == b"v"


def test_failed_open_does_not_keep_the_lock(tmp_path: Path) -> None:
    DB.open(tmp_path).close()
    (tmp_path / "MANIFEST").write_bytes(b"{broken")
    with pytest.raises(LSMError):
        DB.open(tmp_path)
    (tmp_path / "MANIFEST").unlink()
    with DB.open(tmp_path):
        pass


def test_lock_file_survives_orphan_cleanup(tmp_path: Path) -> None:
    DB.open(tmp_path).close()
    with DB.open(tmp_path):
        assert (tmp_path / "LOCK").is_file()
    assert (tmp_path / "LOCK").is_file()


def test_copy_of_a_live_directory_can_be_opened(tmp_path: Path) -> None:
    with DB.open(tmp_path / "live") as db:
        db.put(b"k", b"v")
        shutil.copytree(tmp_path / "live", tmp_path / "copy")
    with DB.open(tmp_path / "copy") as copy:
        assert copy.get(b"k") == b"v"


def test_other_process_is_locked_out(tmp_path: Path) -> None:
    script = textwrap.dedent(
        """
        import sys
        from lsmkv import DB, LockedError
        try:
            DB.open(sys.argv[1])
        except LockedError:
            sys.exit(7)
        """
    )

    def attempt() -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [sys.executable, "-I", "-c", script, str(tmp_path)],
            check=False,
            capture_output=True,
            timeout=60,
        )

    with DB.open(tmp_path):
        held = attempt()
        assert held.returncode == 7, held.stderr
    free = attempt()
    assert free.returncode == 0, free.stderr
