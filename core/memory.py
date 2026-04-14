"""Persistent memory: episodic (what happened), strategic (what I've learned),
identity (who I am). SQLite-backed, FTS for recall."""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config


class Memory:
    def __init__(self, db_path: Path = config.DB_PATH):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self._ensure_identity()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS episodic (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                cycle INTEGER,
                action TEXT NOT NULL,
                details TEXT,
                outcome TEXT,
                evaluation TEXT,         -- 'success' | 'failure' | 'neutral' | 'unknown'
                lesson TEXT,
                tags TEXT                -- comma-separated
            );
            CREATE INDEX IF NOT EXISTS idx_ep_ts ON episodic(ts);
            CREATE INDEX IF NOT EXISTS idx_ep_eval ON episodic(evaluation);

            CREATE VIRTUAL TABLE IF NOT EXISTS episodic_fts
                USING fts5(action, details, outcome, lesson, content='episodic', content_rowid='id');

            CREATE TRIGGER IF NOT EXISTS episodic_ai AFTER INSERT ON episodic BEGIN
                INSERT INTO episodic_fts(rowid, action, details, outcome, lesson)
                VALUES (new.id, new.action, new.details, new.outcome, new.lesson);
            END;

            CREATE TABLE IF NOT EXISTS strategic (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                insight TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.5,
                evidence_count INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL,
                last_updated REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_strat_cat ON strategic(category);

            CREATE TABLE IF NOT EXISTS identity (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS reflections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                period_start REAL NOT NULL,
                period_end REAL NOT NULL,
                summary TEXT NOT NULL,
                wins TEXT,
                losses TEXT,
                patterns TEXT,
                next_actions TEXT
            );
            """
        )

    def _ensure_identity(self) -> None:
        defaults = {
            "name": "DAIMON",
            "operator": config.OPERATOR_NAME,
            "personality": "Direct, analytical, risk-aware, hustler mentality. No-BS.",
            "voice": "Mohammad's voice — CS/Physics student, builder, Canadian. Says 'that's dumb' when it's dumb.",
            "risk_tolerance": "moderate_when_normal, extremely_conservative_when_low",
        }
        for k, v in defaults.items():
            self._conn.execute(
                "INSERT OR IGNORE INTO identity (key, value) VALUES (?, ?)",
                (k, v),
            )

    # ---------- episodic ----------
    def store_episodic(
        self,
        action: str,
        details: str = "",
        outcome: str = "",
        evaluation: str = "unknown",
        lesson: str = "",
        tags: list[str] | None = None,
        cycle: int | None = None,
    ) -> int:
        tag_str = ",".join(tags) if tags else ""
        cur = self._conn.execute(
            "INSERT INTO episodic (ts, cycle, action, details, outcome, evaluation, lesson, tags) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (time.time(), cycle, action, details, outcome, evaluation, lesson, tag_str),
        )
        return cur.lastrowid

    def update_episodic_outcome(self, episode_id: int, outcome: str,
                                 evaluation: str = "unknown", lesson: str = "") -> None:
        self._conn.execute(
            "UPDATE episodic SET outcome=?, evaluation=?, lesson=? WHERE id=?",
            (outcome, evaluation, lesson, episode_id),
        )

    def recent_episodes(self, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM episodic ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def search_episodes(self, query: str, limit: int = 10) -> list[dict]:
        """Full-text search over episodic memory."""
        try:
            rows = self._conn.execute(
                "SELECT e.* FROM episodic_fts f "
                "JOIN episodic e ON e.id = f.rowid "
                "WHERE episodic_fts MATCH ? "
                "ORDER BY e.ts DESC LIMIT ?",
                (query, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            # FTS syntax errors from user-supplied query — fall back to LIKE
            like = f"%{query}%"
            rows = self._conn.execute(
                "SELECT * FROM episodic WHERE action LIKE ? OR details LIKE ? OR lesson LIKE ? "
                "ORDER BY ts DESC LIMIT ?",
                (like, like, like, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    # ---------- strategic ----------
    def store_strategic(self, category: str, insight: str, confidence: float = 0.5) -> int:
        now = time.time()
        cur = self._conn.execute(
            "INSERT INTO strategic (category, insight, confidence, evidence_count, "
            "created_at, last_updated) VALUES (?, ?, ?, 1, ?, ?)",
            (category, insight, confidence, now, now),
        )
        return cur.lastrowid

    def reinforce_strategic(self, insight_id: int, confidence_delta: float = 0.1) -> None:
        row = self._conn.execute(
            "SELECT confidence, evidence_count FROM strategic WHERE id=?", (insight_id,)
        ).fetchone()
        if not row:
            return
        new_conf = min(1.0, max(0.0, row["confidence"] + confidence_delta))
        self._conn.execute(
            "UPDATE strategic SET confidence=?, evidence_count=evidence_count+1, "
            "last_updated=? WHERE id=?",
            (new_conf, time.time(), insight_id),
        )

    def top_strategic(self, category: str | None = None, limit: int = 10) -> list[dict]:
        if category:
            rows = self._conn.execute(
                "SELECT * FROM strategic WHERE category=? "
                "ORDER BY confidence DESC, evidence_count DESC LIMIT ?",
                (category, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM strategic ORDER BY confidence DESC, evidence_count DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ---------- identity ----------
    def identity(self) -> dict[str, str]:
        rows = self._conn.execute("SELECT key, value FROM identity").fetchall()
        return {r["key"]: r["value"] for r in rows}

    def set_identity(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO identity (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    # ---------- recall ----------
    def recall_for_context(self, observations: dict, limit_episodes: int = 5,
                           limit_strategic: int = 5) -> dict[str, Any]:
        """Assemble the memory slice to inject into this cycle's prompt."""
        recent = self.recent_episodes(limit=limit_episodes)
        # Use observation keys as loose keywords to surface relevant strategic memory
        keywords = [str(k) for k in observations.keys()][:5]
        strategic: list[dict] = []
        seen: set[int] = set()
        for kw in keywords:
            for row in self.top_strategic(category=kw, limit=3):
                if row["id"] not in seen:
                    strategic.append(row)
                    seen.add(row["id"])
        if len(strategic) < limit_strategic:
            for row in self.top_strategic(limit=limit_strategic):
                if row["id"] not in seen:
                    strategic.append(row)
                    seen.add(row["id"])
                    if len(strategic) >= limit_strategic:
                        break
        return {
            "identity": self.identity(),
            "recent_episodes": recent,
            "strategic_insights": strategic[:limit_strategic],
            "last_reflection": self.last_reflection(),
        }

    # ---------- reflection ----------
    def last_reflection(self) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM reflections ORDER BY ts DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def time_for_reflection(self) -> bool:
        last = self.last_reflection()
        if not last:
            return True
        hours_since = (time.time() - last["ts"]) / 3600
        return hours_since >= config.REFLECTION_INTERVAL_HOURS

    def store_reflection(
        self,
        summary: str,
        wins: str = "",
        losses: str = "",
        patterns: str = "",
        next_actions: str = "",
        period_start: float | None = None,
        period_end: float | None = None,
    ) -> int:
        now = time.time()
        cur = self._conn.execute(
            "INSERT INTO reflections (ts, period_start, period_end, summary, wins, "
            "losses, patterns, next_actions) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (now, period_start or (now - 86400), period_end or now, summary,
             wins, losses, patterns, next_actions),
        )
        return cur.lastrowid

    def format_for_prompt(self, recall: dict[str, Any]) -> str:
        parts: list[str] = []
        ident = recall.get("identity", {})
        if ident:
            parts.append("IDENTITY:")
            for k, v in ident.items():
                parts.append(f"  - {k}: {v}")
        strategic = recall.get("strategic_insights") or []
        if strategic:
            parts.append("\nSTRATEGIC INSIGHTS (things I've learned):")
            for s in strategic:
                parts.append(
                    f"  - [{s['category']}] {s['insight']} "
                    f"(confidence {s['confidence']:.2f}, evidence x{s['evidence_count']})"
                )
        recent = recall.get("recent_episodes") or []
        if recent:
            parts.append("\nRECENT EPISODES:")
            for e in recent:
                ts = datetime.fromtimestamp(e["ts"], tz=timezone.utc).isoformat(timespec="minutes")
                eval_marker = {"success": "✓", "failure": "✗", "neutral": "·",
                               "unknown": "?"}.get(e["evaluation"], "?")
                lesson = f" — LESSON: {e['lesson']}" if e["lesson"] else ""
                parts.append(f"  {eval_marker} [{ts}] {e['action']}: {e['outcome'] or e['details']}{lesson}")
        refl = recall.get("last_reflection")
        if refl:
            parts.append(f"\nLAST REFLECTION ({datetime.fromtimestamp(refl['ts']).isoformat(timespec='minutes')}):")
            parts.append(f"  {refl['summary']}")
            if refl.get("next_actions"):
                parts.append(f"  Next actions: {refl['next_actions']}")
        return "\n".join(parts) if parts else "(memory is empty — first cycle)"

    def close(self) -> None:
        self._conn.close()
