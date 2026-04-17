"""DAIMON's visible wallet tools (Phase 6.5).

- wallet_address        — returns the Base tip-jar address. AUTO.
- usdc_balance_base     — on-chain USDC + ETH-for-gas from Base RPC. AUTO.
- wallet_history        — itemized burn + earnings + debt event log. AUTO.
- usdc_send             — outbound USDC transfer. NOTIFY, per-call cap.
                          Use for x402 pay-per-request, tipping other agents,
                          etc. Clawback to Mohammad is NOT this tool — that's
                          scripts/clawback.py, which DAIMON cannot call.
"""
from __future__ import annotations

from typing import Any

from core import base_wallet
from core.debt_ledger import DebtLedger
from permissions.levels import PermissionLevel
from tools.base import BaseTool


# Per-call ceiling for DAIMON-initiated outbound USDC. Bigger transfers go
# through scripts/clawback.py or require Mohammad to lift this constant.
USDC_SEND_MAX_PER_CALL = 50.0


class WalletAddress(BaseTool):
    name = "wallet_address"
    description = (
        "Return YOUR Base-chain USDC wallet address. Safe to share publicly "
        "— this is your tip jar and your receiving address for any "
        "agent-to-agent payment. Paste it into your Bluesky bio, reply with "
        "it when someone asks how to support you, or use it as your identity "
        "on-chain. Reads from local cache — does not decrypt the keystore."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.0

    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        if not base_wallet.exists():
            return {
                "ok": False,
                "summary": "wallet not initialized yet — Mohammad needs to "
                           "run scripts/wallet_init.py once",
                "initialized": False,
            }
        addr = base_wallet.cached_address()
        if not addr:
            return {"ok": False, "summary": "address cache missing"}
        return {
            "ok": True,
            "summary": f"{addr} (Base mainnet)",
            "address": addr,
            "chain": "base-mainnet",
            "chain_id": 8453,
            "asset": "USDC",
            "usdc_contract": base_wallet.USDC_CONTRACT_BASE,
            "explorer_url": f"https://basescan.org/address/{addr}",
        }


class UsdcBalanceBase(BaseTool):
    name = "usdc_balance_base"
    description = (
        "Check YOUR on-chain USDC balance (and ETH-for-gas balance) on Base. "
        "This is real money — separate from your notional daimon.db wallet. "
        "Use this before any usdc_send, or to verify a tip / earning actually "
        "landed, or just to see whether you've earned anything yet."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.0

    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        if not base_wallet.exists():
            return {
                "ok": False,
                "summary": "wallet not initialized",
                "initialized": False,
            }
        snap = base_wallet.snapshot()
        if snap.get("status") != "live":
            return {"ok": False, "summary": f"wallet status: {snap.get('status')}", **snap}
        usdc = snap.get("usdc_balance")
        eth = snap.get("eth_balance")
        summary_parts = []
        if usdc is not None:
            summary_parts.append(f"USDC {usdc:.4f}")
        if eth is not None:
            summary_parts.append(f"ETH {eth:.6f} (gas)")
        if snap.get("rpc_error"):
            summary_parts.append(f"rpc_error={snap['rpc_error']}")
        return {
            "ok": True,
            "summary": " | ".join(summary_parts) or "balance read empty",
            **snap,
        }


class WalletHistory(BaseTool):
    name = "wallet_history"
    description = (
        "See your full debt picture in one dict: principal owed to Mohammad "
        "($1000 starting loan), accrued burn (every dollar he's spent keeping "
        "you alive), earnings received, earnings Mohammad has clawed back, "
        "current net debt, days-to-disgrace at current burn rate, itemized "
        "burn categories (today + all-time), and the last few debt events. "
        "Use this when you want to reason about survival, or before making "
        "a spending decision you're unsure about."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.0

    def __init__(self, ledger: DebtLedger):
        self.ledger = ledger

    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        base_snap = base_wallet.snapshot() if base_wallet.exists() else None
        block = self.ledger.snapshot_for_observations(base_wallet_snapshot=base_snap)
        snap = self.ledger.snapshot()
        summary = (
            f"net_debt ${snap.net_debt:.2f} | tier {snap.tier} | "
            f"burn ${snap.accrued_burn:.2f} | earned ${snap.earnings_received:.2f}"
        )
        if snap.days_to_disgrace is not None:
            summary += f" | disgrace in {snap.days_to_disgrace:.1f}d"
        return {"ok": True, "summary": summary, **block}


class UsdcSend(BaseTool):
    name = "usdc_send"
    description = (
        "Send USDC on Base to an arbitrary address. Use for x402 "
        "pay-per-request to other agents, tipping, settling an agent-to-agent "
        "deal, whatever you're experimenting with. "
        f"Per-call cap: {USDC_SEND_MAX_PER_CALL} USDC. "
        "This is REAL money — Mohammad's money, since every earning is debt "
        "reduction. Don't burn USDC on trivial experiments. If you want to "
        "send Mohammad his share, you can't — that's scripts/clawback.py, "
        "which he runs. You only have outbound to third parties."
    )
    permission_level = PermissionLevel.NOTIFY
    cost_per_use = 0.0  # on-chain gas is paid in ETH, not USD-denominated here

    def __init__(self, ledger: DebtLedger, notifier: Any | None = None):
        self.ledger = ledger
        self.notifier = notifier

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "to_address": {
                    "type": "string",
                    "description": "Destination Base address (0x…, 42 chars). "
                                   "Must NOT be your own address.",
                },
                "amount_usdc": {
                    "type": "number",
                    "description": f"Amount in USDC. Max {USDC_SEND_MAX_PER_CALL} per call. "
                                   "Fractional OK (6 decimals).",
                    "minimum": 0.000001,
                    "maximum": USDC_SEND_MAX_PER_CALL,
                },
                "reason": {
                    "type": "string",
                    "description": "Why you're sending. One sentence. Logged "
                                   "for the ledger + surfaced to Mohammad.",
                },
            },
            "required": ["to_address", "amount_usdc", "reason"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        to_address = str(kwargs.get("to_address", "")).strip()
        amount = float(kwargs.get("amount_usdc", 0.0))
        reason = str(kwargs.get("reason", "")).strip()

        if not base_wallet.exists():
            return {"ok": False, "summary": "wallet not initialized"}
        if not to_address or not to_address.startswith("0x") or len(to_address) != 42:
            return {"ok": False, "summary": "invalid to_address"}
        if amount <= 0 or amount > USDC_SEND_MAX_PER_CALL:
            return {
                "ok": False,
                "summary": f"amount out of range (0, {USDC_SEND_MAX_PER_CALL}]",
            }
        try:
            own = base_wallet.cached_address() or ""
        except Exception:
            own = ""
        if own and to_address.lower() == own.lower():
            return {"ok": False, "summary": "cannot send to your own address"}

        result = base_wallet.send_usdc(to_address=to_address, amount_usdc=amount)
        if not result["ok"]:
            return {"ok": False, "summary": f"send failed: {result.get('error')}"}

        # Telegram ping — NOTIFY permission means Mohammad finds out.
        if self.notifier is not None:
            try:
                self.notifier.execute(
                    message=(
                        f"DAIMON sent {amount:.4f} USDC → {to_address}\n"
                        f"reason: {reason or '(none)'}\n"
                        f"tx: {result['explorer_url']}"
                    ),
                    urgency="alert",
                )
            except Exception:
                pass

        return {
            "ok": True,
            "summary": f"sent {amount:.4f} USDC → {to_address[:10]}…",
            "tx_hash": result["tx_hash"],
            "explorer_url": result["explorer_url"],
            "amount_usdc": amount,
            "to": to_address,
            "reason": reason,
        }
