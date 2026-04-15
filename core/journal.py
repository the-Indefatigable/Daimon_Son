"""DAIMON's journal. Private first (honest thinking), public-ready second
(excerpts it chooses to share). This is how DAIMON builds a narrative."""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import config


@dataclass
class JournalEntry:
    id: int
    ts: float
    cycle: int | None
    kind: str               # 'cycle_note' | 'reflection' | 'manifesto' | 'failure_post_mortem'
    title: str
    body: str
    published: bool
    published_at: float | None
    published_url: str | None

    def ts_iso(self) -> str:
        return datetime.fromtimestamp(self.ts, tz=timezone.utc).isoformat(timespec="minutes")


class Journal:
    def __init__(self, db_path: Path = config.DB_PATH):
        self._conn = sqlite3.connect(db_path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                cycle INTEGER,
                kind TEXT NOT NULL,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                published INTEGER NOT NULL DEFAULT 0,
                published_at REAL,
                published_url TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_journal_ts ON journal(ts);
            CREATE INDEX IF NOT EXISTS idx_journal_kind ON journal(kind);
            """
        )

    def write(self, kind: str, title: str, body: str,
              cycle: int | None = None) -> int:
        cur = self._conn.execute(
            "INSERT INTO journal (ts, cycle, kind, title, body) VALUES (?, ?, ?, ?, ?)",
            (time.time(), cycle, kind, title, body),
        )
        return cur.lastrowid

    def mark_published(self, entry_id: int, url: str = "") -> None:
        self._conn.execute(
            "UPDATE journal SET published=1, published_at=?, published_url=? WHERE id=?",
            (time.time(), url, entry_id),
        )

    def recent(self, limit: int = 10, kind: str | None = None,
               published_only: bool = False) -> list[JournalEntry]:
        q = "SELECT * FROM journal"
        clauses: list[str] = []
        params: list = []
        if kind:
            clauses.append("kind=?")
            params.append(kind)
        if published_only:
            clauses.append("published=1")
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(q, params).fetchall()
        return [self._row(r) for r in rows]

    def last_manifesto(self) -> JournalEntry | None:
        row = self._conn.execute(
            "SELECT * FROM journal WHERE kind='manifesto' ORDER BY ts DESC LIMIT 1"
        ).fetchone()
        return self._row(row) if row else None

    @staticmethod
    def _row(r: sqlite3.Row) -> JournalEntry:
        return JournalEntry(
            id=r["id"], ts=r["ts"], cycle=r["cycle"], kind=r["kind"],
            title=r["title"], body=r["body"],
            published=bool(r["published"]),
            published_at=r["published_at"],
            published_url=r["published_url"],
        )

    def format_recent_for_prompt(self, limit: int = 3) -> str:
        entries = self.recent(limit=limit)
        if not entries:
            return "(no journal entries yet — you haven't started writing about yourself)"
        lines: list[str] = []
        for e in entries:
            lines.append(f"[{e.ts_iso()}] ({e.kind}) {e.title}")
            # Clip body to keep prompt tight
            body = e.body[:500]
            if len(e.body) > 500:
                body += "…"
            lines.append(body)
            lines.append("")
        return "\n".join(lines).strip()

    def close(self) -> None:
        self._conn.close()
