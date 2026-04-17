#!/usr/bin/env python
"""MOHAMMAD-ONLY: claw back USDC from DAIMON's Base wallet to yours.

Not a DAIMON tool. Not exposed in any registry. DAIMON cannot call this,
cannot refuse it, cannot see it coming. It witnesses the withdrawal in its
next observations block because record_withdrawal writes to debt_events.

Usage:
  .venv/bin/python scripts/clawback.py <amount_usdc>
  .venv/bin/python scripts/clawback.py --all
  .venv/bin/python scripts/clawback.py <amount_usdc> --to 0xYourOtherAddress

The destination defaults to $MOHAMMAD_BASE_ADDRESS from .env. Override with
--to if you want to send elsewhere.

Requires in .env:
  DAIMON_WALLET_PASSWORD  — decrypts the keystore
  MOHAMMAD_BASE_ADDRESS   — default clawback destination
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import base_wallet
from core.debt_ledger import DebtLedger
from core.wallet import Wallet


def main() -> int:
    parser = argparse.ArgumentParser(description="Mohammad's clawback tool.")
    parser.add_argument("amount", nargs="?", type=float, default=None,
                        help="USDC amount to withdraw (human units). Omit if --all.")
    parser.add_argument("--all", action="store_true",
                        help="Withdraw the full USDC balance.")
    parser.add_argument("--to", type=str, default=None,
                        help="Destination Base address. Defaults to MOHAMMAD_BASE_ADDRESS.")
    parser.add_argument("--yes", action="store_true",
                        help="Skip confirmation prompt.")
    args = parser.parse_args()

    if not base_wallet.exists():
        print("❌ No keystore. Run scripts/wallet_init.py first.")
        return 1

    password = os.getenv("DAIMON_WALLET_PASSWORD", "").strip()
    if not password:
        print("❌ DAIMON_WALLET_PASSWORD not set in env.")
        return 1

    to_address = (args.to or os.getenv("MOHAMMAD_BASE_ADDRESS", "")).strip()
    if not to_address:
        print("❌ No destination. Pass --to 0x... or set MOHAMMAD_BASE_ADDRESS in .env.")
        return 1

    try:
        current = base_wallet.usdc_balance()
    except Exception as e:
        print(f"❌ Could not read on-chain balance: {e}")
        return 1

    if args.all:
        amount = current
    elif args.amount is not None:
        amount = float(args.amount)
    else:
        print("❌ Specify an amount or pass --all.")
        parser.print_help()
        return 1

    if amount <= 0:
        print("❌ Amount must be positive.")
        return 1
    if amount > current:
        print(f"❌ Insufficient USDC. On-chain balance: {current:.6f}, requested: {amount:.6f}")
        return 1

    print("─" * 60)
    print("  CLAWBACK")
    print(f"    from DAIMON:  {base_wallet.address()}")
    print(f"    to Mohammad:  {to_address}")
    print(f"    amount:       {amount:.6f} USDC")
    print(f"    current bal:  {current:.6f} USDC")
    print("─" * 60)

    if not args.yes:
        confirm = input("  Proceed? [y/N] ").strip().lower()
        if confirm not in ("y", "yes"):
            print("  aborted.")
            return 0

    print("  Broadcasting tx…")
    result = base_wallet.send_usdc(
        to_address=to_address, amount_usdc=amount, password=password,
    )
    if not result["ok"]:
        print(f"  ❌ failed: {result.get('error')}")
        return 1

    tx_hash = result["tx_hash"]
    print(f"  ✓ tx: {tx_hash}")
    print(f"    {result['explorer_url']}")

    # Record the clawback in the debt ledger. DAIMON sees this next cycle.
    ledger = DebtLedger(Wallet())
    ledger.record_withdrawal(
        amount=amount, tx_hash=tx_hash,
        details=f"Mohammad clawback → {to_address}",
    )
    print(f"  ✓ debt_events row written. DAIMON will witness this next cycle.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
