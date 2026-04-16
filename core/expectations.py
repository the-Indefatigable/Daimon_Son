"""Expectations: prediction-error encoding.

Every action whose result DAIMON cares about gets a row here at do-time:
  what it did, what it predicted, when results should be visible.
At result-time, DAIMON fills in actual_value + surprise. The gap teaches.

This is the keystone primitive for time-awareness AND memory salience —
- check_after_ts answers "when should I hope for results"
- surprise feeds memory weighting and the semantic schema layer (by principle)
- auto-expiration turns silence into a real signal
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

from . import config


VALID_STATUS = {"pending", "checked", "expired", "cancelled"}


class Expectations:
    def __init__(self, db_path: Path = config.DB_PATH):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS expectations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle INTEGER NOT NULL,
                created_ts REAL NOT NULL,

                action_kind TEXT NOT NULL,
                action_ref TEXT,
                action_summary TEXT NOT NULL,

                predicted_metric TEXT NOT NULL,
                predicted_value TEXT NOT NULL,
                predicted_basis TEXT,
                principle TEXT,

                check_after_ts REAL NOT NULL,
                check_before_ts REAL,

                status TEXT NOT NULL DEFAULT 'pending',
                checked_ts REAL,
                actual_value TEXT,
                surprise REAL,
                notes TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_exp_due
                ON expectations(status, check_after_ts);
            CREATE INDEX IF NOT EXISTS idx_exp_principle
                ON expectations(principle);
            """
        )

    # ---------- write ----------
    def create(
        self,
        *,
        cycle: int,
        action_kind: str,
        action_summary: str,
        predicted_metric: str,
        predicted_value: str,
        check_after_hours: float,
        check_before_hours: float | None = None,
        action_ref: str | None = None,
        predicted_basis: str | None = None,
        principle: str | None = None,
    ) -> int:
        now = time.time()
        check_after = now + check_after_hours * 3600
        check_before = (now + check_before_hours * 3600) if check_before_hours else None
        cur = self._conn.execute(
            "INSERT INTO expectations "
            "(cycle, created_ts, action_kind, action_ref, action_summary, "
            " predicted_metric, predicted_value, predicted_basis, principle, "
            " check_after_ts, check_before_ts) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (cycle, now, action_kind, action_ref, action_summary,
             predicted_metric, predicted_value, predicted_basis, principle,
             check_after, check_before),
        )
        return cur.lastrowid

    def record_outcome(
        self,
        expectation_id: int,
        actual_value: str,
        surprise: float,
        notes: str | None = None,
    ) -> bool:
        surprise = max(0.0, min(1.0, float(surprise)))
        cur = self._conn.execute(
            "UPDATE expectations "
            "SET status='checked', checked_ts=?, actual_value=?, surprise=?, notes=? "
            "WHERE id=? AND status='pending'",
            (time.time(), actual_value, surprise, notes, expectation_id),
        )
        return bool(cur.rowcount)

    def cancel(self, expectation_id: int, reason: str = "") -> bool:
        cur = self._conn.execute(
            "UPDATE expectations SET status='cancelled', notes=? "
            "WHERE id=? AND status='pending'",
            (reason or None, expectation_id),
        )
        return bool(cur.rowcount)

    def expire_overdue(self) -> list[dict]:
        """Mark pending expectations whose check_before_ts has passed as expired.
        Returns the rows that just expired so the caller can surface them."""
        now = time.time()
        rows = self._conn.execute(
            "SELECT * FROM expectations "
            "WHERE status='pending' AND check_before_ts IS NOT NULL "
            "AND check_before_ts < ?",
            (now,),
        ).fetchall()
        if not rows:
            return []
        ids = [r["id"] for r in rows]
        placeholders = ",".join("?" for _ in ids)
        self._conn.execute(
            f"UPDATE expectations SET status='expired', checked_ts=? "
            f"WHERE id IN ({placeholders})",
            (now, *ids),
        )
        return [dict(r) for r in rows]

    # ---------- read ----------
    def due_now(self, limit: int = 10) -> list[dict]:
        now = time.time()
        rows = self._conn.execute(
            "SELECT * FROM expectations "
            "WHERE status='pending' AND check_after_ts <= ? "
            "ORDER BY check_after_ts ASC LIMIT ?",
            (now, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def still_waiting(self, limit: int = 10) -> list[dict]:
        now = time.time()
        rows = self._conn.execute(
            "SELECT * FROM expectations "
            "WHERE status='pending' AND check_after_ts > ? "
            "ORDER BY check_after_ts ASC LIMIT ?",
            (now, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get(self, expectation_id: int) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM expectations WHERE id=?", (expectation_id,)
        ).fetchone()
        return dict(row) if row else None

    def by_principle(self, principle: str, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM expectations WHERE principle=? "
            "ORDER BY created_ts DESC LIMIT ?",
            (principle, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def pending_count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS c FROM expectations WHERE status='pending'"
        ).fetchone()
        return int(row["c"]) if row else 0

    # ---------- prompt formatting ----------
    @staticmethod
    def _shorten_row(row: dict) -> dict:
        """Trim a row to the fields DAIMON actually needs in observations."""
        out = {
            "id": row["id"],
            "kind": row["action_kind"],
            "summary": (row["action_summary"] or "")[:200],
            "predicted": f"{row['predicted_metric']} = {row['predicted_value']}",
            "principle": row.get("principle"),
        }
        if row.get("action_ref"):
            out["ref"] = row["action_ref"]
        if row.get("predicted_basis"):
            out["basis"] = row["predicted_basis"][:200]
        return out

    def snapshot_for_observations(self) -> dict[str, Any]:
        """The block to inject into _observe(). Side-effect: expires overdue rows
        first so 'just_expired' surfaces them exactly once."""
        just_expired = self.expire_overdue()
        due = self.due_now(limit=10)
        waiting = self.still_waiting(limit=10)
        now = time.time()

        return {
            "due_now": [
                {**self._shorten_row(r),
                 "ready_for_hours": round((now - r["check_after_ts"]) / 3600, 1)}
                for r in due
            ],
            "still_waiting": [
                {**self._shorten_row(r),
                 "check_in_hours": round((r["check_after_ts"] - now) / 3600, 1)}
                for r in waiting
            ],
            "just_expired": [
                {**self._shorten_row(r),
                 "note": "window closed without you recording an outcome — "
                         "silence IS the result; record_outcome with surprise"}
                for r in just_expired
            ],
            "pending_total": self.pending_count(),
            "guidance": (
                "Do NOT check results for actions in 'still_waiting' — wait for "
                "their window to open. For 'due_now' rows, run the check tool "
                "(github_pr_status / bluesky_read / stripe_metrics / etc) and "
                "then call record_outcome. For 'just_expired' rows, the absence "
                "of a result is itself information — record it."
            ) if (due or waiting or just_expired) else (
                "No pending expectations. When you ship something whose result "
                "you actually care about, call expect_result first."
            ),
        }

    def close(self) -> None:
        self._conn.close()
