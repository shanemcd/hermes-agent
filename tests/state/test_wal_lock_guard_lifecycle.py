"""The WAL lock guard keeps the live writer's WAL generation from being unlinked.

POSIX advisory locks are per process, so any ``open()``/``close()`` of ``state.db`` or
``state.db-shm`` inside the holder cancels them (howtocorrupt §2.2) and the next sibling
close may unlink ``-wal``/``-shm`` under the live writer — stranding it on a deleted
generation and failing writes with SQLITE_CANTOPEN. ``hermes_state_lockguard`` re-takes
the same ranges as OFD locks, which a stray close cannot cancel and which refuse the
foreign EXCLUSIVE a close-time unlink needs.

Linux-only: OFD locks + ``/proc`` descriptor enumeration.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys

import pytest

import hermes_state_lockguard as lg
import hermes_state_wal
from hermes_state import SessionDB

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="OFD locks + /proc")


def pin_wal(monkeypatch) -> None:
    monkeypatch.setattr(
        hermes_state_wal, "is_sqlite_wal_reset_vulnerable", lambda version_info=None: False
    )
    monkeypatch.setattr(hermes_state_wal, "resolve_journal_mode", lambda: "wal")


def make_db(path, session_id: str, content: str) -> SessionDB:
    db = SessionDB(db_path=path)
    db.create_session(session_id, "cli")
    db.append_message(session_id, role="user", content=content)
    return db


def require_wal(db: SessionDB) -> None:
    if not db._wal_active or not os.path.exists(os.fspath(db.db_path) + "-wal"):
        db.close()
        pytest.skip("WAL not active on this filesystem/SQLite build")


def _foreign_exclusive_ok(path: str) -> bool:
    """Another process tries the EXCLUSIVE a close-time WAL reset needs; True = nothing guards."""
    code = (
        "import fcntl, os, struct, sys\n"
        f"fd = os.open({path!r}, os.O_RDWR)\n"
        "lk = struct.pack('@hhqqi', fcntl.F_WRLCK, 0, 0x40000002, 510, 0)\n"
        "try:\n    fcntl.fcntl(fd, 37, lk); print('EXCLUSIVE_ACQUIRED')\n"
        "except BlockingIOError:\n    print('REFUSED')\n"
    )
    return "EXCLUSIVE_ACQUIRED" in subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout


def test_writer_guard_survives_a_stray_open_close(tmp_path, monkeypatch):
    pin_wal(monkeypatch)
    db = make_db(tmp_path / "state.db", "s", "seed")
    require_wal(db)
    try:
        assert db._wal_lock_guard, "open WAL writer came back unguarded"
        assert not _foreign_exclusive_ok(str(db.db_path)), "guard did not refuse a foreign unlink"
        # The stray close the guard exists for: raw open()/close() of both sidecars.
        for name in ("state.db", "state.db-shm"):
            p = tmp_path / name
            if p.exists():
                os.close(os.open(p, os.O_RDONLY))
        assert not _foreign_exclusive_ok(str(db.db_path)), "guard cancelled by a stray open/close"
        db.append_message("s", role="user", content="after stray close")
    finally:
        db.close()
    assert _foreign_exclusive_ok(str(db.db_path)), "a true last close must lift the guard"
    assert not lg._HANDLES


def test_guard_never_outlives_the_handle_under_fd_reuse(tmp_path, monkeypatch):
    """A+B live -> close A (its fd number is recycled by C) -> close B: C must still be guarded,
    and once C closes nothing may be left locked."""
    pin_wal(monkeypatch)
    path = tmp_path / "state.db"
    a = make_db(path, "s", "seed")
    require_wal(a)
    b = SessionDB(db_path=path)
    a.close()
    c = SessionDB(db_path=path)
    b.close()
    try:
        assert not _foreign_exclusive_ok(str(path)), "C recorded as guarded while nothing locks"
        c.append_message("s", role="user", content="still writes")
    finally:
        c.close()
    assert _foreign_exclusive_ok(str(path)), "a lock survived the last handle's close"
    assert not lg._HANDLES
    assert sqlite3.connect(path).execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
