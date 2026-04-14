"""Resource requester: DAIMON formats and logs requests for APIs, budget,
platform access, or infra upgrades. Notifies Mohammad via Telegram."""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

from . import config


RequestType = Literal["API_KEY", "BUDGET", "PLATFORM_ACCESS", "INFRASTRUCTURE"]
Priority = Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
Status = Literal["pending", "approved", "denied", "withdrawn"]


@dataclass
class ResourceRequest:
    id: int | None
    ts: float
    type: RequestType
    priority: Priority
    title: str                # short one-liner
    what: str                 # specific thing needed
    why: str                  # business reasoning
    expected_benefit: str
    expected_cost: str
    risk_if_denied: str
    setup_notes: str          # how Mohammad can act quickly
    status: Status
    resolved_at: float | None
    resolution_notes: str

    def format_telegram(self) -> str:
        return (
            f"📋 RESOURCE REQUEST from DAIMON\n\n"
            f"Type: {self.type}\n"
            f"Priority: {self.priority}\n\n"
            f"What I need: {self.what}\n"
            f"Why: {self.why}\n"
            f"Expected benefit: {self.expected_benefit}\n"
            f"Expected cost: {self.expected_cost}\n"
            f"Risk if denied: {self.risk_if_denied}\n\n"
            f"Setup notes:\n{self.setup_notes}\n\n"
            f"Reply YES to approve or NO with reason."
        )


class ResourceRequester:
    def __init__(self, db_path: Path = config.DB_PATH):
        self._conn = sqlite3.connect(db_path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS resource_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                type TEXT NOT NULL,
                priority TEXT NOT NULL,
                title TEXT NOT NULL,
                what TEXT NOT NULL,
                why TEXT NOT NULL,
                expected_benefit TEXT,
                expected_cost TEXT,
                risk_if_denied TEXT,
                setup_notes TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                resolved_at REAL,
                resolution_notes TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_req_status ON resource_requests(status);
            """
        )

    def create(
        self,
        type: RequestType,
        priority: Priority,
        title: str,
        what: str,
        why: str,
        expected_benefit: str = "",
        expected_cost: str = "",
        risk_if_denied: str = "",
        setup_notes: str = "",
    ) -> ResourceRequest:
        # Dedupe: skip if an identical pending request already exists
        existing = self._conn.execute(
            "SELECT * FROM resource_requests WHERE status='pending' AND what=? AND type=?",
            (what, type),
        ).fetchone()
        if existing:
            return self._row_to_request(existing)

        now = time.time()
        cur = self._conn.execute(
            "INSERT INTO resource_requests (ts, type, priority, title, what, why, "
            "expected_benefit, expected_cost, risk_if_denied, setup_notes, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')",
            (now, type, priority, title, what, why, expected_benefit, expected_cost,
             risk_if_denied, setup_notes),
        )
        return ResourceRequest(
            id=cur.lastrowid, ts=now, type=type, priority=priority, title=title,
            what=what, why=why, expected_benefit=expected_benefit,
            expected_cost=expected_cost, risk_if_denied=risk_if_denied,
            setup_notes=setup_notes, status="pending", resolved_at=None,
            resolution_notes="",
        )

    def resolve(self, request_id: int, status: Status, notes: str = "") -> None:
        self._conn.execute(
            "UPDATE resource_requests SET status=?, resolved_at=?, resolution_notes=? "
            "WHERE id=?",
            (status, time.time(), notes, request_id),
        )

    def pending(self) -> list[ResourceRequest]:
        rows = self._conn.execute(
            "SELECT * FROM resource_requests WHERE status='pending' ORDER BY ts ASC"
        ).fetchall()
        return [self._row_to_request(r) for r in rows]

    def history(self, limit: int = 20) -> list[ResourceRequest]:
        rows = self._conn.execute(
            "SELECT * FROM resource_requests ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_request(r) for r in rows]

    @staticmethod
    def _row_to_request(r: sqlite3.Row) -> ResourceRequest:
        return ResourceRequest(
            id=r["id"], ts=r["ts"], type=r["type"], priority=r["priority"],
            title=r["title"], what=r["what"], why=r["why"],
            expected_benefit=r["expected_benefit"] or "",
            expected_cost=r["expected_cost"] or "",
            risk_if_denied=r["risk_if_denied"] or "",
            setup_notes=r["setup_notes"] or "",
            status=r["status"], resolved_at=r["resolved_at"],
            resolution_notes=r["resolution_notes"] or "",
        )

    def close(self) -> None:
        self._conn.close()
