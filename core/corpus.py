"""Backrooms corpus tracker. Every scripts/backrooms.py run lands here so
DAIMON can reason about corpus size, freshness, and cumulative cost.

A run's lifecycle:
  start_run(...)       -> row with status='running'
  finalize_from_meta(meta_json_path) -> status updated + cost from meta
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from . import config


@dataclass
class BackroomsRun:
    id: int
    started_ts: float
    ended_ts: float | None
    turns_requested: int
    turns_completed: int
    fresh: bool
    log_path: str | None
    status: str             # 'running' | 'completed' | 'interrupted' | 'failed'
    error: str | None
    claude_calls: int
    grok_calls: int
    claude_tokens_in: int
    claude_tokens_out: int
    grok_tokens_in: int
    grok_tokens_out: int
    claude_cost_usd: float
    grok_cost_usd: float
    total_cost_usd: float


class BackroomsRuns:
    def __init__(self, db_path: Path = config.DB_PATH):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS backrooms_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_ts REAL NOT NULL,
                ended_ts REAL,
                turns_requested INTEGER NOT NULL,
                turns_completed INTEGER NOT NULL DEFAULT 0,
                fresh INTEGER NOT NULL DEFAULT 0,
                log_path TEXT,
                status TEXT NOT NULL DEFAULT 'running',
                error TEXT,
                claude_calls INTEGER NOT NULL DEFAULT 0,
                grok_calls INTEGER NOT NULL DEFAULT 0,
                claude_tokens_in INTEGER NOT NULL DEFAULT 0,
                claude_tokens_out INTEGER NOT NULL DEFAULT 0,
                grok_tokens_in INTEGER NOT NULL DEFAULT 0,
                grok_tokens_out INTEGER NOT NULL DEFAULT 0,
                claude_cost_usd REAL NOT NULL DEFAULT 0,
                grok_cost_usd REAL NOT NULL DEFAULT 0,
                total_cost_usd REAL NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_br_status
                ON backrooms_runs(status, started_ts);
            """
        )

    # ---------- write ----------
    def start_run(self, *, turns_requested: int, fresh: bool) -> int:
        cur = self._conn.execute(
            "INSERT INTO backrooms_runs (started_ts, turns_requested, fresh, status) "
            "VALUES (?, ?, ?, 'running')",
            (time.time(), turns_requested, int(bool(fresh))),
        )
        return cur.lastrowid

    def finalize_from_meta(self, run_id: int, meta_json_path: Path) -> BackroomsRun | None:
        """Read the .meta.json emitted by backrooms.py and write usage back."""
        if not meta_json_path.exists():
            self._conn.execute(
                "UPDATE backrooms_runs SET status='failed', ended_ts=?, "
                "error='meta.json missing' WHERE id=?",
                (time.time(), run_id),
            )
            return self.get(run_id)

        meta = json.loads(meta_json_path.read_text())
        self._conn.execute(
            "UPDATE backrooms_runs SET "
            "  ended_ts=?, turns_completed=?, log_path=?, status=?, error=?, "
            "  claude_calls=?, grok_calls=?, "
            "  claude_tokens_in=?, claude_tokens_out=?, "
            "  grok_tokens_in=?, grok_tokens_out=?, "
            "  claude_cost_usd=?, grok_cost_usd=?, total_cost_usd=? "
            "WHERE id=?",
            (
                meta.get("ended_ts", time.time()),
                int(meta.get("turns_completed", 0)),
                meta.get("log_path"),
                meta.get("status", "completed"),
                meta.get("error"),
                int(meta.get("claude_calls", 0)),
                int(meta.get("grok_calls", 0)),
                int(meta.get("claude_tokens_in", 0)),
                int(meta.get("claude_tokens_out", 0)),
                int(meta.get("grok_tokens_in", 0)),
                int(meta.get("grok_tokens_out", 0)),
                float(meta.get("claude_cost_usd", 0.0)),
                float(meta.get("grok_cost_usd", 0.0)),
                float(meta.get("total_cost_usd", 0.0)),
                run_id,
            ),
        )
        return self.get(run_id)

    # ---------- read ----------
    def get(self, run_id: int) -> BackroomsRun | None:
        row = self._conn.execute(
            "SELECT * FROM backrooms_runs WHERE id=?", (run_id,)
        ).fetchone()
        return _row_to_run(row) if row else None

    def recent(self, limit: int = 10) -> list[BackroomsRun]:
        rows = self._conn.execute(
            "SELECT * FROM backrooms_runs ORDER BY started_ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_row_to_run(r) for r in rows]

    def aggregate_stats(self) -> dict:
        row = self._conn.execute(
            "SELECT COUNT(*) AS runs, "
            " SUM(turns_completed) AS total_turns, "
            " SUM(total_cost_usd) AS total_cost, "
            " MAX(ended_ts) AS last_run_ts "
            "FROM backrooms_runs WHERE status='completed'"
        ).fetchone()
        return {
            "completed_runs": int(row["runs"] or 0),
            "total_turns": int(row["total_turns"] or 0),
            "total_cost_usd": float(row["total_cost"] or 0.0),
            "last_completed_ts": float(row["last_run_ts"]) if row["last_run_ts"] else None,
        }


def _row_to_run(row: sqlite3.Row) -> BackroomsRun:
    return BackroomsRun(
        id=row["id"],
        started_ts=row["started_ts"],
        ended_ts=row["ended_ts"],
        turns_requested=row["turns_requested"],
        turns_completed=row["turns_completed"],
        fresh=bool(row["fresh"]),
        log_path=row["log_path"],
        status=row["status"],
        error=row["error"],
        claude_calls=row["claude_calls"],
        grok_calls=row["grok_calls"],
        claude_tokens_in=row["claude_tokens_in"],
        claude_tokens_out=row["claude_tokens_out"],
        grok_tokens_in=row["grok_tokens_in"],
        grok_tokens_out=row["grok_tokens_out"],
        claude_cost_usd=row["claude_cost_usd"],
        grok_cost_usd=row["grok_cost_usd"],
        total_cost_usd=row["total_cost_usd"],
    )
