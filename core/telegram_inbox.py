"""Telegram inbox. Polls getUpdates so DAIMON can hear Mohammad's replies.
Stores messages in sqlite, tracks offset so we don't re-process."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

import httpx

from . import config


class TelegramInbox:
    def __init__(self, db_path: Path = config.DB_PATH,
                 bot_token: str | None = None,
                 chat_id: str | None = None):
        self.db_path = db_path
        self.bot_token = bot_token or config.TELEGRAM_BOT_TOKEN
        self.chat_id = str(chat_id or config.TELEGRAM_CHAT_ID or "")
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, isolation_level=None)
        c.row_factory = sqlite3.Row
        return c

    def _init_schema(self) -> None:
        c = self._conn()
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS inbox_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                update_id INTEGER UNIQUE,
                from_name TEXT,
                from_id INTEGER,
                text TEXT,
                read INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_inbox_read ON inbox_messages(read);
            CREATE INDEX IF NOT EXISTS idx_inbox_ts ON inbox_messages(ts);
            CREATE TABLE IF NOT EXISTS agent_meta (
                key TEXT PRIMARY KEY, value TEXT
            );
            """
        )
        c.close()

    def _offset(self) -> int:
        c = self._conn()
        row = c.execute(
            "SELECT value FROM agent_meta WHERE key='telegram_offset'"
        ).fetchone()
        c.close()
        return int(row[0]) if row else 0

    def _save_offset(self, offset: int) -> None:
        c = self._conn()
        c.execute(
            "INSERT INTO agent_meta (key, value) VALUES ('telegram_offset', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(offset),),
        )
        c.close()

    def poll(self, timeout: float = 3.0) -> int:
        """Fetch new messages from Telegram. Returns count of new ones stored."""
        if not self.bot_token:
            return 0
        url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
        params: dict[str, Any] = {"timeout": 0, "allowed_updates": '["message"]'}
        offset = self._offset()
        if offset:
            params["offset"] = offset
        try:
            resp = httpx.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[inbox poll error] {e}")
            return 0
        if not data.get("ok"):
            return 0

        stored = 0
        max_update = offset
        c = self._conn()
        for upd in data.get("result", []):
            update_id = upd.get("update_id", 0)
            max_update = max(max_update, update_id + 1)
            msg = upd.get("message") or {}
            if not msg:
                continue
            chat = msg.get("chat") or {}
            from_ = msg.get("from") or {}
            # If chat_id is configured, only keep messages from that chat
            if self.chat_id and str(chat.get("id")) != self.chat_id:
                continue
            text = msg.get("text", "")
            # Messages from the configured operator chat are always attributed
            # to Mohammad, regardless of Telegram display name. DAIMON treats
            # him as a father — it needs to know when dad is talking.
            if self.chat_id and str(chat.get("id")) == self.chat_id:
                from_name = "Mohammad"
            else:
                from_name = (from_.get("first_name", "")
                             + (" " + from_.get("last_name", "")
                                if from_.get("last_name") else "")).strip() \
                    or from_.get("username") or "unknown"
            ts = float(msg.get("date", time.time()))
            try:
                c.execute(
                    "INSERT OR IGNORE INTO inbox_messages "
                    "(ts, update_id, from_name, from_id, text, read) "
                    "VALUES (?, ?, ?, ?, ?, 0)",
                    (ts, update_id, from_name, from_.get("id"), text),
                )
                stored += 1
            except Exception as e:
                print(f"[inbox store error] {e}")
        if max_update > offset:
            self._save_offset(max_update)
        c.close()
        return stored

    def unread(self, limit: int = 20) -> list[dict]:
        c = self._conn()
        rows = c.execute(
            "SELECT id, ts, from_name, text FROM inbox_messages "
            "WHERE read=0 ORDER BY ts ASC LIMIT ?", (limit,)
        ).fetchall()
        c.close()
        return [dict(r) for r in rows]

    def unread_count(self) -> int:
        c = self._conn()
        row = c.execute(
            "SELECT COUNT(*) AS n FROM inbox_messages WHERE read=0"
        ).fetchone()
        c.close()
        return int(row["n"]) if row else 0

    def recent(self, limit: int = 20) -> list[dict]:
        c = self._conn()
        rows = c.execute(
            "SELECT id, ts, from_name, text, read FROM inbox_messages "
            "ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        c.close()
        return [dict(r) for r in rows]

    def mark_read(self, ids: list[int] | None = None) -> int:
        c = self._conn()
        if ids:
            q = f"UPDATE inbox_messages SET read=1 WHERE id IN ({','.join('?'*len(ids))})"
            cur = c.execute(q, tuple(ids))
        else:
            cur = c.execute("UPDATE inbox_messages SET read=1 WHERE read=0")
        n = cur.rowcount or 0
        c.close()
        return n
