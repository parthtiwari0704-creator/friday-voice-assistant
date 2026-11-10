"""SQLite persistence: notes (long-term memory) and reminders that survive restarts."""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, due_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending', created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders(status, due_at);
"""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(SCHEMA)

    def _all(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def _exec(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        with self._lock, self._conn:
            return self._conn.execute(sql, params)

    def close(self) -> None:
        self._conn.close()

    # --- notes -------------------------------------------------------------------------------------
    def add_note(self, text: str) -> int:
        return int(
            self._exec("INSERT INTO notes(text, created_at) VALUES (?,?)", (text, utcnow().isoformat())).lastrowid or 0
        )

    def notes(self, query: str = "", limit: int = 10) -> list[dict[str, Any]]:
        words = [w for w in query.lower().split() if len(w) > 2]
        rows = self._all("SELECT id, text, created_at FROM notes ORDER BY id DESC LIMIT 500")
        if words:
            scored = [(sum(w in r["text"].lower() for w in words), r) for r in rows]
            rows = [r for score, r in sorted(scored, key=lambda x: -x[0]) if score > 0]
        return rows[:limit]

    # --- reminders ---------------------------------------------------------------------------------
    def add_reminder(self, text: str, due_at: datetime) -> int:
        due = due_at.astimezone(timezone.utc).isoformat()
        cur = self._exec(
            "INSERT INTO reminders(text, due_at, created_at) VALUES (?,?,?)", (text, due, utcnow().isoformat())
        )
        return int(cur.lastrowid or 0)

    def pending_reminders(self) -> list[dict[str, Any]]:
        return self._all("SELECT id, text, due_at FROM reminders WHERE status='pending' ORDER BY due_at")

    def cancel_reminder(self, reminder_id: int) -> bool:
        cur = self._exec("UPDATE reminders SET status='cancelled' WHERE id=? AND status='pending'", (reminder_id,))
        return cur.rowcount == 1

    def pop_due_reminders(self, now: datetime | None = None) -> list[dict[str, Any]]:
        """Atomically claim reminders that are due, so each fires exactly once."""
        now_iso = (now or utcnow()).astimezone(timezone.utc).isoformat()
        with self._lock, self._conn:
            rows = [dict(r) for r in self._conn.execute(
                "SELECT id, text, due_at FROM reminders WHERE status='pending' AND due_at <= ? ORDER BY due_at", (now_iso,)
            )]  # fmt: skip
            for r in rows:
                self._conn.execute("UPDATE reminders SET status='fired' WHERE id=?", (r["id"],))
        return rows
