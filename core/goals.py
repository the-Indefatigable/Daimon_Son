"""Open-ended goals DAIMON creates for itself. Not a task list — a set of
open questions and projects it's actively working on."""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from . import config


@dataclass
class Goal:
    id: int
    ts_created: float
    title: str
    rationale: str            # why DAIMON set this goal
    horizon: str              # 'today' | 'week' | 'month' | 'open'
    status: str               # 'active' | 'completed' | 'abandoned' | 'paused'
    progress_notes: str
    ts_resolved: float | None
    resolution: str           # how it ended — what DAIMON learned


class Goals:
    def __init__(self, db_path: Path = config.DB_PATH):
        self._conn = sqlite3.connect(db_path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_created REAL NOT NULL,
                title TEXT NOT NULL,
                rationale TEXT NOT NULL,
                horizon TEXT NOT NULL DEFAULT 'open',
                status TEXT NOT NULL DEFAULT 'active',
                progress_notes TEXT NOT NULL DEFAULT '',
                ts_resolved REAL,
                resolution TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status);
            """
        )

    def create(self, title: str, rationale: str, horizon: str = "open") -> int:
        cur = self._conn.execute(
            "INSERT INTO goals (ts_created, title, rationale, horizon) "
            "VALUES (?, ?, ?, ?)",
            (time.time(), title, rationale, horizon),
        )
        return cur.lastrowid

    def add_progress(self, goal_id: int, note: str) -> None:
        row = self._conn.execute(
            "SELECT progress_notes FROM goals WHERE id=?", (goal_id,)
        ).fetchone()
        if not row:
            return
        existing = row["progress_notes"]
        stamped = f"[{time.strftime('%Y-%m-%d %H:%M')}] {note}"
        combined = f"{existing}\n{stamped}".strip()
        self._conn.execute(
            "UPDATE goals SET progress_notes=? WHERE id=?",
            (combined, goal_id),
        )

    def resolve(self, goal_id: int, status: str, resolution: str) -> None:
        self._conn.execute(
            "UPDATE goals SET status=?, ts_resolved=?, resolution=? WHERE id=?",
            (status, time.time(), resolution, goal_id),
        )

    def active(self, limit: int = 10) -> list[Goal]:
        rows = self._conn.execute(
            "SELECT * FROM goals WHERE status='active' ORDER BY ts_created DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row(r) for r in rows]

    def recent(self, limit: int = 20) -> list[Goal]:
        rows = self._conn.execute(
            "SELECT * FROM goals ORDER BY ts_created DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row(r) for r in rows]

    @staticmethod
    def _row(r: sqlite3.Row) -> Goal:
        return Goal(
            id=r["id"], ts_created=r["ts_created"], title=r["title"],
            rationale=r["rationale"], horizon=r["horizon"], status=r["status"],
            progress_notes=r["progress_notes"], ts_resolved=r["ts_resolved"],
            resolution=r["resolution"],
        )

    def format_active_for_prompt(self) -> str:
        active = self.active()
        if not active:
            return "(no active goals — you haven't set any yet)"
        lines: list[str] = []
        for g in active:
            lines.append(f"- [{g.horizon}] {g.title}")
            lines.append(f"    why: {g.rationale}")
            if g.progress_notes:
                last_line = g.progress_notes.strip().split("\n")[-1]
                lines.append(f"    latest: {last_line}")
        return "\n".join(lines)

    def close(self) -> None:
        self._conn.close()
