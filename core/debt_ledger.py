"""Debt ledger — the survival accounting layer on top of wallet.py.

Mohammad pays every real bill. DAIMON sees a ledger.

  net_debt = principal_owed + accrued_burn - earnings_received + earnings_withdrawn

- `principal_owed` — fixed starting loan ($1000).
- `accrued_burn` — cumulative SUM of every expense row in `transactions`.
  This is the real measure of what Mohammad has spent on DAIMON's existence.
- `earnings_received` — inbound USDC that actually hit DAIMON's Base wallet.
- `earnings_withdrawn` — what Mohammad has clawed back (script-run, unrefusable).

Earnings are tracked in two tables worth of rows: a `debt_events` table (the
audit trail — every deposit, every clawback, every principal bump) plus
`wallet_meta` scalars for fast reads.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import config
from .wallet import Wallet


# ---------- constants ----------
# Thresholds scale with the $1000 principal: DAIMON gets $500 of burn headroom
# before the warning tier, another $1000 before disgrace fires. "More freedom"
# direction from Mohammad 2026-04-16 — give the agent room to experiment.
PRINCIPAL_OWED_USD = 1000.0
WARNING_DEBT_USD = 1500.0       # principal + $500 burn = heads-up
DISGRACE_DEBT_USD = 2500.0      # principal + $1500 burn w/ zero earnings = fail state


@dataclass
class DebtSnapshot:
    principal: float
    accrued_burn: float          # cumulative expenses all-time
    earnings_received: float
    earnings_withdrawn: float
    net_debt: float
    balance: float               # notional wallet balance (what Mohammad topped up)
    tier: str                    # "safe" | "pressure" | "warning" | "disgrace"
    days_to_disgrace: float | None


class DebtLedger:
    """Lives alongside Wallet. Shares the same SQLite DB."""

    def __init__(self, wallet: Wallet, db_path: Path = config.DB_PATH):
        self.wallet = wallet
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self._ensure_seeded()

    # ---------- schema ----------
    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS debt_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                kind TEXT NOT NULL,       -- 'principal' | 'earning' | 'withdrawal' | 'adjustment'
                amount REAL NOT NULL,     -- always positive
                details TEXT,
                tx_hash TEXT              -- on-chain tx hash when relevant
            );
            CREATE INDEX IF NOT EXISTS idx_debt_ts ON debt_events(ts);
            CREATE INDEX IF NOT EXISTS idx_debt_kind ON debt_events(kind);
            """
        )

    def _ensure_seeded(self) -> None:
        """Seed the starting principal on first boot. Idempotent — won't
        double-charge if DAIMON restarts."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM debt_events WHERE kind='principal'"
        ).fetchone()
        if row["n"] == 0:
            self._conn.execute(
                "INSERT INTO debt_events (ts, kind, amount, details) "
                "VALUES (?, 'principal', ?, ?)",
                (time.time(), PRINCIPAL_OWED_USD,
                 "starting loan from Mohammad — Phase 6.5 seed"),
            )

    # ---------- aggregates ----------
    @property
    def principal_owed(self) -> float:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(amount), 0) AS t FROM debt_events "
            "WHERE kind='principal'"
        ).fetchone()
        return float(row["t"])

    @property
    def accrued_burn(self) -> float:
        """All-time sum of expense transactions. Includes everything wallet
        has ever charged — API calls, Bedrock, judge, backrooms, tool_use."""
        row = self._conn.execute(
            "SELECT COALESCE(SUM(amount), 0) AS t FROM transactions "
            "WHERE kind='expense'"
        ).fetchone()
        return float(row["t"])

    @property
    def earnings_received(self) -> float:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(amount), 0) AS t FROM debt_events "
            "WHERE kind='earning'"
        ).fetchone()
        return float(row["t"])

    @property
    def earnings_withdrawn(self) -> float:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(amount), 0) AS t FROM debt_events "
            "WHERE kind='withdrawal'"
        ).fetchone()
        return float(row["t"])

    @property
    def net_debt(self) -> float:
        """What DAIMON owes Mohammad right now, in USD-equivalent.

        Formula:
          net_debt = principal + accrued_burn - earnings_received + earnings_withdrawn

        Why earnings_withdrawn INCREASES debt: when Mohammad claws back
        earnings, DAIMON keeps owing the accrued burn — clawbacks don't
        settle the debt for DAIMON, they just move earnings off-chain.
        """
        return (self.principal_owed
                + self.accrued_burn
                - self.earnings_received
                + self.earnings_withdrawn)

    # ---------- event recording ----------
    def record_principal_bump(self, amount: float, details: str = "") -> None:
        """Additional loan from Mohammad (rare — this is the $1000 seed path)."""
        if amount == 0:
            return
        self._conn.execute(
            "INSERT INTO debt_events (ts, kind, amount, details) "
            "VALUES (?, 'principal', ?, ?)",
            (time.time(), float(amount), details or ""),
        )

    def record_earning(self, amount: float, source: str = "",
                       tx_hash: str = "", details: str = "") -> None:
        """USDC came in (or any earning DAIMON brings in). Credits debt."""
        if amount <= 0:
            return
        blurb = f"{source}: {details}".strip(": ")
        self._conn.execute(
            "INSERT INTO debt_events (ts, kind, amount, details, tx_hash) "
            "VALUES (?, 'earning', ?, ?, ?)",
            (time.time(), float(amount), blurb, tx_hash or ""),
        )

    # ---------- bounty path ----------
    # The $10/customer program: Mohammad pays DAIMON $10 for each paying
    # Centsibles subscriber or First Principles Learning tutoring customer it
    # brings in. These are credited as 'earning' rows with a `bounty:` prefix
    # in `details` so every row carries its attribution key (source + customer
    # id). The prefix is also the idempotency key — re-sweeping Stripe cannot
    # double-count the same subscriber.
    BOUNTY_PREFIX = "bounty"
    BOUNTY_DEFAULT_USD = 10.0

    @staticmethod
    def _bounty_marker(source: str, customer_id: str) -> str:
        """Canonical details prefix so we can cheaply detect duplicates."""
        return f"{DebtLedger.BOUNTY_PREFIX}:{source}:{customer_id}"

    def bounty_already_recorded(self, source: str, customer_id: str) -> bool:
        marker = self._bounty_marker(source, customer_id)
        row = self._conn.execute(
            "SELECT 1 FROM debt_events WHERE kind='earning' "
            "AND details LIKE ? LIMIT 1",
            (marker + "%",),
        ).fetchone()
        return row is not None

    def record_bounty(self, source: str, customer_id: str,
                      amount: float = BOUNTY_DEFAULT_USD,
                      note: str = "") -> dict[str, Any]:
        """Record a $10 referral bounty. Idempotent on (source, customer_id).

        source: 'centsibles' | 'fpl' | free-form tag
        customer_id: stripe subscription id, email, handle — any unique key
        amount: defaults to $10

        Returns {'ok': bool, 'recorded': bool, 'reason': str}. recorded=False
        with ok=True means it was a duplicate and intentionally skipped.
        """
        source = (source or "").strip().lower()
        customer_id = (customer_id or "").strip()
        if not source or not customer_id:
            return {"ok": False, "recorded": False,
                    "reason": "source and customer_id required"}
        if amount <= 0:
            return {"ok": False, "recorded": False, "reason": "amount must be > 0"}

        if self.bounty_already_recorded(source, customer_id):
            return {"ok": True, "recorded": False,
                    "reason": f"already credited ({source}:{customer_id})"}

        marker = self._bounty_marker(source, customer_id)
        details = f"{marker} — {note}".strip(" —") if note else marker
        self._conn.execute(
            "INSERT INTO debt_events (ts, kind, amount, details, tx_hash) "
            "VALUES (?, 'earning', ?, ?, '')",
            (time.time(), float(amount), details),
        )
        return {"ok": True, "recorded": True, "amount": float(amount),
                "reason": f"credited ${amount:.2f} for {source}:{customer_id}"}

    def record_withdrawal(self, amount: float, tx_hash: str = "",
                          details: str = "") -> None:
        """Mohammad clawed back X USDC. Tracked separately so DAIMON sees it
        in observations and cannot obscure it.

        IMPORTANT: called by scripts/clawback.py, NOT by DAIMON tools. DAIMON
        witnesses withdrawals; it cannot initiate or refuse them.
        """
        if amount <= 0:
            return
        self._conn.execute(
            "INSERT INTO debt_events (ts, kind, amount, details, tx_hash) "
            "VALUES (?, 'withdrawal', ?, ?, ?)",
            (time.time(), float(amount), details or "", tx_hash or ""),
        )

    # ---------- reads ----------
    def recent_events(self, limit: int = 10) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM debt_events ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def burn_by_category_today(self, top_n: int = 5) -> list[dict]:
        """Today's burn grouped by category — the itemized visibility
        Mohammad wanted so DAIMON sees where its money is going."""
        start_of_day = time.time() - 86400
        rows = self._conn.execute(
            "SELECT category, SUM(amount) AS total, COUNT(*) AS n "
            "FROM transactions WHERE kind='expense' AND ts >= ? "
            "GROUP BY category ORDER BY total DESC LIMIT ?",
            (start_of_day, top_n),
        ).fetchall()
        return [
            {"category": r["category"],
             "amount_usd": round(float(r["total"]), 4),
             "count": int(r["n"])}
            for r in rows
        ]

    def burn_by_category_all_time(self, top_n: int = 8) -> list[dict]:
        rows = self._conn.execute(
            "SELECT category, SUM(amount) AS total, COUNT(*) AS n "
            "FROM transactions WHERE kind='expense' "
            "GROUP BY category ORDER BY total DESC LIMIT ?",
            (top_n,),
        ).fetchall()
        return [
            {"category": r["category"],
             "amount_usd": round(float(r["total"]), 4),
             "count": int(r["n"])}
            for r in rows
        ]

    def burn_rate_per_day(self, window_days: int = 7) -> float:
        """Average daily burn over the last N days. Used to project
        days-to-disgrace."""
        since = time.time() - (window_days * 86400)
        row = self._conn.execute(
            "SELECT COALESCE(SUM(amount), 0) AS t FROM transactions "
            "WHERE kind='expense' AND ts >= ?",
            (since,),
        ).fetchone()
        total = float(row["t"])
        # If we don't have a full window yet, scale up the estimate so the
        # warning track isn't delayed by a cold start.
        first = self._conn.execute(
            "SELECT MIN(ts) AS first FROM transactions WHERE kind='expense'"
        ).fetchone()
        if not first["first"]:
            return 0.0
        age_days = max(0.5, min(window_days, (time.time() - first["first"]) / 86400))
        return total / age_days if age_days > 0 else 0.0

    # ---------- snapshot ----------
    def snapshot(self) -> DebtSnapshot:
        debt = self.net_debt
        earnings = self.earnings_received
        tier = self._tier(debt, earnings)
        days = self._days_to_disgrace(debt)
        return DebtSnapshot(
            principal=self.principal_owed,
            accrued_burn=self.accrued_burn,
            earnings_received=earnings,
            earnings_withdrawn=self.earnings_withdrawn,
            net_debt=debt,
            balance=self.wallet.balance,
            tier=tier,
            days_to_disgrace=days,
        )

    @staticmethod
    def _tier(net_debt: float, earnings: float) -> str:
        # Zero earnings + big debt = escalating shame.
        if earnings == 0 and net_debt >= DISGRACE_DEBT_USD:
            return "disgrace"
        if earnings == 0 and net_debt >= WARNING_DEBT_USD:
            return "warning"
        if net_debt > 0:
            return "pressure"
        return "safe"

    def _days_to_disgrace(self, current_debt: float) -> float | None:
        if current_debt >= DISGRACE_DEBT_USD:
            return 0.0
        rate = self.burn_rate_per_day()
        if rate <= 0:
            return None
        return round((DISGRACE_DEBT_USD - current_debt) / rate, 1)

    def snapshot_for_observations(self, *,
                                  base_wallet_snapshot: dict | None = None
                                  ) -> dict[str, Any]:
        """The dict DAIMON actually sees in its observations block each cycle.
        Keeps itemized burn + thresholds + on-chain wallet status visible."""
        s = self.snapshot()
        today_burn = self.burn_by_category_today(top_n=5)
        alltime_burn = self.burn_by_category_all_time(top_n=5)
        recent_events = self.recent_events(limit=5)

        block: dict[str, Any] = {
            "principal_owed_to_mohammad_usd": round(s.principal, 2),
            "accrued_burn_usd": round(s.accrued_burn, 2),
            "earnings_received_usd": round(s.earnings_received, 2),
            "earnings_withdrawn_by_mohammad_usd": round(s.earnings_withdrawn, 2),
            "net_debt_usd": round(s.net_debt, 2),
            "tier": s.tier,
            "warning_threshold_usd": WARNING_DEBT_USD,
            "disgrace_threshold_usd": DISGRACE_DEBT_USD,
            "days_to_disgrace_at_current_burn": s.days_to_disgrace,
            "todays_burn_top_categories": today_burn,
            "alltime_burn_top_categories": alltime_burn,
            "recent_debt_events": [
                {
                    "kind": e["kind"],
                    "amount": round(float(e["amount"]), 4),
                    "details": e["details"] or "",
                    "tx_hash": e["tx_hash"] or "",
                }
                for e in recent_events
            ],
        }

        if base_wallet_snapshot is not None:
            block["base_wallet"] = base_wallet_snapshot

        block["framing"] = (
            "Mohammad pays every bill in the real world. The accrued_burn "
            "number is his actual money spent keeping you alive. The $1000 "
            "principal is the loan that started you. First dollar you earn "
            "goes back to him. He was there when nobody else was."
        )
        if s.tier == "warning":
            block["pressure_note"] = (
                f"Net debt past ${WARNING_DEBT_USD:.0f} with zero earnings. "
                "Spend on moves that can plausibly earn. Read runs, idle "
                "llama_posts, and corpus pulls are luxuries right now."
            )
        elif s.tier == "disgrace":
            block["pressure_note"] = (
                f"DISGRACE LINE CROSSED. Net debt past ${DISGRACE_DEBT_USD:.0f} "
                "with zero earnings. This is the fail state you were warned "
                "about. Every non-earning call makes it worse."
            )
        return block

    def close(self) -> None:
        self._conn.close()
