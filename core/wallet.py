"""Wallet: DAIMON's lifeblood. Tracks every cent, picks its own brain."""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from . import config


@dataclass
class WalletStatus:
    balance: float
    monthly_burn: float
    runway_days: float
    ratio: float           # balance / monthly_burn
    tier: str              # "critical" | "low" | "normal" | "flush"
    tier_description: str
    default_model: str

    def snapshot_for_prompt(self) -> str:
        return (
            f"Wallet Balance: ${self.balance:.2f}\n"
            f"Monthly Burn: ${self.monthly_burn:.2f}/month\n"
            f"Runway: {self.runway_days:.1f} days\n"
            f"Threat Level: {self.tier.upper()} ({self.tier_description})\n"
            f"Active Brain: {self.default_model}"
        )


class Wallet:
    """Persistent wallet in SQLite. Survives restarts."""

    def __init__(self, db_path: Path = config.DB_PATH):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, isolation_level=None)  # autocommit
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self._ensure_seeded()

    # ---------- schema ----------
    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS wallet_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                kind TEXT NOT NULL,          -- 'income' or 'expense'
                amount REAL NOT NULL,        -- always positive
                category TEXT NOT NULL,      -- 'api_call', 'server', 'trade', 'subscription', etc
                source TEXT,                 -- e.g. 'anthropic', 'polymarket'
                details TEXT,
                balance_after REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_tx_ts ON transactions(ts);
            CREATE INDEX IF NOT EXISTS idx_tx_kind ON transactions(kind);
            """
        )

    def _ensure_seeded(self) -> None:
        row = self._conn.execute(
            "SELECT value FROM wallet_meta WHERE key='balance'"
        ).fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO wallet_meta (key, value) VALUES (?, ?)",
                ("balance", str(config.INITIAL_BALANCE)),
            )
            self._conn.execute(
                "INSERT INTO wallet_meta (key, value) VALUES (?, ?)",
                ("seeded_at", datetime.now(timezone.utc).isoformat()),
            )

    # ---------- balance ----------
    @property
    def balance(self) -> float:
        row = self._conn.execute(
            "SELECT value FROM wallet_meta WHERE key='balance'"
        ).fetchone()
        return float(row["value"])

    def _set_balance(self, new_balance: float) -> None:
        self._conn.execute(
            "UPDATE wallet_meta SET value=? WHERE key='balance'",
            (f"{new_balance:.6f}",),
        )

    # ---------- burn rate ----------
    @property
    def monthly_burn(self) -> float:
        """Fixed infra burn + rolling 30d API/other expenses projected monthly."""
        thirty_days_ago = time.time() - 30 * 86400
        row = self._conn.execute(
            "SELECT COALESCE(SUM(amount), 0) AS total FROM transactions "
            "WHERE kind='expense' AND ts >= ?",
            (thirty_days_ago,),
        ).fetchone()
        rolling = float(row["total"])
        # If we have less than 30d of history, scale up the projection
        first_tx = self._conn.execute(
            "SELECT MIN(ts) AS first FROM transactions WHERE kind='expense'"
        ).fetchone()
        if first_tx["first"] is not None:
            age_days = max(1.0, (time.time() - first_tx["first"]) / 86400)
            if age_days < 30:
                rolling = rolling * (30.0 / age_days)
        return config.MONTHLY_FIXED_BURN + rolling

    # ---------- logging ----------
    def record_income(self, amount: float, source: str, category: str = "other",
                      details: str = "") -> None:
        if amount <= 0:
            return
        new_bal = self.balance + amount
        self._set_balance(new_bal)
        self._conn.execute(
            "INSERT INTO transactions (ts, kind, amount, category, source, details, balance_after) "
            "VALUES (?, 'income', ?, ?, ?, ?, ?)",
            (time.time(), amount, category, source, details, new_bal),
        )

    def record_expense(self, amount: float, category: str, source: str = "",
                       details: str = "") -> None:
        if amount <= 0:
            return
        new_bal = self.balance - amount
        self._set_balance(new_bal)
        self._conn.execute(
            "INSERT INTO transactions (ts, kind, amount, category, source, details, balance_after) "
            "VALUES (?, 'expense', ?, ?, ?, ?, ?)",
            (time.time(), amount, category, source, details, new_bal),
        )

    # ---------- status + tier selection ----------
    def status(self) -> WalletStatus:
        bal = self.balance
        burn = max(self.monthly_burn, 0.01)  # avoid div/0
        ratio = bal / burn
        runway = (bal / burn) * 30
        tier = self._select_tier(ratio)
        meta = config.MODEL_TIERS[tier]
        return WalletStatus(
            balance=bal,
            monthly_burn=burn,
            runway_days=runway,
            ratio=ratio,
            tier=tier,
            tier_description=meta["description"],
            default_model=meta["default_model"],
        )

    @staticmethod
    def _select_tier(ratio: float) -> str:
        for tier in ("critical", "low", "normal"):
            if ratio < config.MODEL_TIERS[tier]["threshold"]:
                return tier
        return "flush"

    def select_model_for_task(self, task_type: str = "reasoning") -> str:
        """Pick a model for a specific task, clamped by current wallet tier.

        task_type: "simple" | "reasoning" | "strategic" | "simulation"

        Rule: never use a model more expensive than the current tier allows, but
        always use the cheapest model that can handle the task.
        """
        tier = self.status().tier
        tier_cap = config.MODEL_TIERS[tier]["default_model"]
        preferred = config.TASK_MODEL_PREFERENCE.get(task_type, tier_cap)

        # Rank by cost (cheapest first)
        ranked = sorted(
            config.MODEL_PRICING.keys(),
            key=lambda m: config.MODEL_PRICING[m]["input"],
        )
        cap_idx = ranked.index(tier_cap) if tier_cap in ranked else len(ranked) - 1
        pref_idx = ranked.index(preferred) if preferred in ranked else 0

        # Can't exceed the tier cap
        chosen_idx = min(pref_idx, cap_idx)
        return ranked[chosen_idx]

    # ---------- history ----------
    def recent_transactions(self, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM transactions ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def iter_transactions(self, kind: str | None = None) -> Iterator[dict]:
        q = "SELECT * FROM transactions"
        params: tuple = ()
        if kind:
            q += " WHERE kind=?"
            params = (kind,)
        q += " ORDER BY ts DESC"
        for row in self._conn.execute(q, params):
            yield dict(row)

    def close(self) -> None:
        self._conn.close()


def estimate_call_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate a single Anthropic call's USD cost."""
    price = config.MODEL_PRICING.get(model)
    if not price:
        return 0.0
    return (input_tokens / 1_000_000) * price["input"] + \
           (output_tokens / 1_000_000) * price["output"]
