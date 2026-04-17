#!/usr/bin/env python
"""MOHAMMAD-ONLY: credit or reverse a referral bounty on DAIMON's ledger.

Not a DAIMON tool. Use this when:
  - A tutoring (FPL) customer signed up through DAIMON — credit $10.
  - DAIMON claimed a bounty manually and you want to reverse a fake claim.
  - You're crediting a non-Stripe Centsibles referral that the automatic
    `bounty_sweep_centsibles` tool won't see.

DAIMON can't write to `debt_events` in the withdrawal/adjustment kinds. It
witnesses the new row on its next cycle.

Usage:
  python scripts/record_bounty.py --source fpl --customer-id "alice@x.com"
  python scripts/record_bounty.py --source fpl --customer-id bob --amount 10 \
      --note "signed up after DAIMON's reply thread on 2026-04-18"
  python scripts/record_bounty.py --reverse --source fpl --customer-id bob
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.debt_ledger import DebtLedger
from core.wallet import Wallet


def main() -> int:
    parser = argparse.ArgumentParser(description="Mohammad's bounty ledger tool.")
    parser.add_argument("--source", type=str, required=True,
                        help="Bounty source: 'fpl', 'centsibles', 'other'.")
    parser.add_argument("--customer-id", type=str, required=True,
                        help="Unique customer identifier — email, handle, name.")
    parser.add_argument("--amount", type=float, default=10.0,
                        help="USD bounty amount. Default $10.")
    parser.add_argument("--note", type=str, default="",
                        help="Free-text context written into the ledger row.")
    parser.add_argument("--reverse", action="store_true",
                        help="Reverse a previously credited bounty (writes an "
                             "offsetting 'adjustment' row).")
    args = parser.parse_args()

    ledger = DebtLedger(Wallet())

    if args.reverse:
        # Write an offsetting 'adjustment' row so the earning stays visible
        # as a receipt but the net is zeroed. DAIMON sees both rows.
        marker = DebtLedger._bounty_marker(args.source.lower(),
                                           args.customer_id.strip())
        # Check the bounty actually exists first.
        row = ledger._conn.execute(
            "SELECT amount, id FROM debt_events WHERE kind='earning' "
            "AND details LIKE ? ORDER BY ts ASC LIMIT 1",
            (marker + "%",),
        ).fetchone()
        if not row:
            print(f"❌ No bounty found for source={args.source} customer={args.customer_id}")
            return 1
        orig_amount = float(row["amount"])
        # Adjustment row: negative-impact 'earning' isn't allowed (amount>0 check),
        # so we insert a 'withdrawal' kind to subtract from the net — that's the
        # same mechanism clawback uses.
        ledger._conn.execute(
            "INSERT INTO debt_events (ts, kind, amount, details) "
            "VALUES (?, 'withdrawal', ?, ?)",
            (time.time(), orig_amount,
             f"bounty_reversal:{marker} — {args.note or 'Mohammad reversal'}"),
        )
        print(f"✓ reversed ${orig_amount:.2f} bounty for {args.source}:{args.customer_id}")
        print(f"  (DAIMON sees this as a clawback-shaped debt event next cycle)")
        return 0

    result = ledger.record_bounty(
        source=args.source.lower(),
        customer_id=args.customer_id.strip(),
        amount=float(args.amount),
        note=args.note.strip(),
    )
    if not result.get("ok"):
        print(f"❌ {result.get('reason')}")
        return 1
    if not result.get("recorded"):
        print(f"⚠ duplicate — already credited. {result.get('reason')}")
        return 0
    print(f"✓ {result.get('reason')}")
    print(f"  (DAIMON will see this in its debt_status block next cycle)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
