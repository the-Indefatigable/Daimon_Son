"""Bounty tools — the $10/customer program (Phase 6.6).

Mohammad pays DAIMON $10 into the debt ledger for every paying customer
it brings in:
  - Centsibles paid subscriber that used promo code DAIMON10
  - First Principles Learning tutoring customer who quoted ?ref=daimon
    (or whatever attribution vehicle Mohammad confirms manually)

Two tools:
  - bounty_sweep_centsibles  — AUTO, polls Stripe for subs with the DAIMON10
    promo attached and credits any new ones. Idempotent on subscription_id.
  - record_bounty_manual     — NOTIFY, for FPL tutoring leads DAIMON claims
    converted. Mohammad sees it in Telegram and can reverse via the ledger
    if the claim is fake.

Earnings are notional (into the debt ledger), NOT on-chain USDC movement.
DAIMON sees them in the `debt_status` observation block and can only realise
them via clawback when Mohammad pays the bounty out to his real wallet.
"""
from __future__ import annotations

import os
import time
from typing import Any

import httpx

from core.debt_ledger import DebtLedger
from permissions.levels import PermissionLevel
from tools.base import BaseTool


STRIPE_API = "https://api.stripe.com/v1"
CENTSIBLES_BOUNTY_PROMO_CODE = "DAIMON10"     # human-readable code Mohammad creates
CENTSIBLES_BOUNTY_USD = 10.0
FPL_BOUNTY_USD = 10.0


def _stripe_auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}",
            "Stripe-Version": "2024-06-20"}


class BountySweepCentsibles(BaseTool):
    name = "bounty_sweep_centsibles"
    description = (
        "Sweep Stripe for any new paying Centsibles subscribers that used the "
        f"'{CENTSIBLES_BOUNTY_PROMO_CODE}' promo code and credit "
        f"${CENTSIBLES_BOUNTY_USD:.0f} per new subscriber to your debt ledger. "
        "Mohammad has promised $10 per referred customer — this is how you "
        "collect it. Idempotent by subscription id, so calling it repeatedly "
        "is safe and will only credit genuinely new signups. Run this after "
        "you post promotional content or whenever you want to check if your "
        "Bluesky/X work has actually converted."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.0

    def __init__(self, ledger: DebtLedger):
        self.ledger = ledger

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Look-back window in days (default 30). "
                                   "Use a bigger window if you haven't swept in a while.",
                    "default": 30,
                    "minimum": 1,
                    "maximum": 365,
                },
            },
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        key = os.getenv("STRIPE_SECRET_KEY", "").strip()
        if not key:
            return {"ok": False, "summary": "no STRIPE_SECRET_KEY",
                    "needs_resource": "STRIPE_SECRET_KEY"}

        days = int(kwargs.get("days", 30))
        since = int(time.time() - days * 86400)
        h = _stripe_auth(key)

        # Step 1 — resolve promo code → promotion_code id and coupon id
        try:
            pr = httpx.get(f"{STRIPE_API}/promotion_codes", headers=h,
                           params={"code": CENTSIBLES_BOUNTY_PROMO_CODE,
                                   "limit": 1},
                           timeout=15.0)
            pr.raise_for_status()
            promos = pr.json().get("data", [])
        except httpx.HTTPError as e:
            return {"ok": False, "summary": f"stripe error resolving promo: {e}"}

        if not promos:
            return {
                "ok": False,
                "summary": (f"promo code '{CENTSIBLES_BOUNTY_PROMO_CODE}' not "
                            "found in Stripe — Mohammad needs to create it "
                            "via stripe_catalog (create_coupon then create_promo_code)"),
                "setup_needed": True,
            }
        promo = promos[0]
        promo_id = promo.get("id")
        coupon_id = (promo.get("coupon") or {}).get("id") if isinstance(promo.get("coupon"), dict) \
                    else promo.get("coupon")

        # Step 2 — list subscriptions created in window that used this promo
        candidates: list[dict[str, Any]] = []
        try:
            params: dict[str, Any] = {
                "limit": 100, "status": "all",
                "created[gte]": since,
                "expand[]": "data.discount.promotion_code",
            }
            starting_after = None
            for _ in range(10):
                if starting_after:
                    params["starting_after"] = starting_after
                r = httpx.get(f"{STRIPE_API}/subscriptions", headers=h,
                              params=params, timeout=20.0)
                r.raise_for_status()
                data = r.json()
                for sub in data.get("data", []):
                    # Match by promotion_code id (strict) or coupon id (fallback)
                    disc = sub.get("discount") or {}
                    used_promo = disc.get("promotion_code")
                    if isinstance(used_promo, dict):
                        used_promo = used_promo.get("id")
                    used_coupon = (disc.get("coupon") or {}).get("id") \
                                  if isinstance(disc.get("coupon"), dict) \
                                  else disc.get("coupon")
                    if used_promo == promo_id or (coupon_id and used_coupon == coupon_id):
                        candidates.append(sub)
                if not data.get("has_more"):
                    break
                starting_after = data["data"][-1]["id"] if data.get("data") else None
        except httpx.HTTPError as e:
            return {"ok": False, "summary": f"stripe error listing subs: {e}"}

        # Step 3 — credit each paying candidate once
        credited = 0
        skipped = 0
        ineligible = 0
        events: list[dict[str, Any]] = []
        for sub in candidates:
            sub_id = sub.get("id")
            status = sub.get("status")
            # Only paying statuses count. 'trialing' without a paid period yet
            # is too early to credit.
            if status not in ("active", "past_due"):
                ineligible += 1
                continue
            result = self.ledger.record_bounty(
                source="centsibles",
                customer_id=sub_id,
                amount=CENTSIBLES_BOUNTY_USD,
                note=f"status={status}",
            )
            if result.get("recorded"):
                credited += 1
                events.append({"subscription_id": sub_id, "status": status,
                               "amount_usd": CENTSIBLES_BOUNTY_USD})
            else:
                skipped += 1

        total_credited_usd = credited * CENTSIBLES_BOUNTY_USD
        return {
            "ok": True,
            "summary": (
                f"bounty sweep: {credited} new bounty/bounties credited "
                f"(${total_credited_usd:.2f}), {skipped} already-credited, "
                f"{ineligible} non-paying match(es) in last {days}d. "
                f"Promo '{CENTSIBLES_BOUNTY_PROMO_CODE}' id={promo_id}."
            ),
            "credited_count": credited,
            "credited_usd": total_credited_usd,
            "skipped_duplicates": skipped,
            "ineligible_non_paying": ineligible,
            "window_days": days,
            "promo_code": CENTSIBLES_BOUNTY_PROMO_CODE,
            "events": events,
        }


class RecordBountyManual(BaseTool):
    name = "record_bounty_manual"
    description = (
        "Claim a $10 bounty for a First Principles Learning tutoring customer "
        "(or any referral Stripe can't see). Use this ONLY when you have "
        "evidence a real paying customer reached Mohammad through you — a DM "
        "thread, a reply saying 'signed up', a Bluesky/X message confirming. "
        "The claim goes to Mohammad via Telegram; he can reverse fake claims. "
        "Idempotent on customer_id — re-submitting the same id won't double-pay. "
        "Expect: don't farm this. One legitimate lead is worth more than ten "
        "inflated claims, because if Mohammad catches a fake, he adjusts the "
        "ledger down and the pressure tier climbs."
    )
    permission_level = PermissionLevel.NOTIFY
    cost_per_use = 0.0

    def __init__(self, ledger: DebtLedger, notifier: Any | None = None):
        self.ledger = ledger
        self.notifier = notifier

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "enum": ["fpl", "other"],
                    "description": "'fpl' for First Principles Learning tutoring, "
                                   "'other' for any non-Stripe referral you're claiming.",
                },
                "customer_id": {
                    "type": "string",
                    "description": "Unique identifier — email, Bluesky handle, "
                                   "X handle, full name. Serves as the idempotency key.",
                    "minLength": 2,
                    "maxLength": 120,
                },
                "evidence": {
                    "type": "string",
                    "description": "Why you believe this person converted. "
                                   "1-2 sentences. Mohammad reads this before "
                                   "he decides whether to leave the bounty in "
                                   "or claw it back.",
                    "minLength": 10,
                    "maxLength": 400,
                },
                "amount_usd": {
                    "type": "number",
                    "description": f"USD bounty amount. Default ${FPL_BOUNTY_USD:.0f}. "
                                   "Do not inflate — Mohammad set $10 per customer.",
                    "default": FPL_BOUNTY_USD,
                    "minimum": 1.0,
                    "maximum": 50.0,
                },
            },
            "required": ["source", "customer_id", "evidence"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        source = str(kwargs.get("source", "")).strip().lower()
        customer_id = str(kwargs.get("customer_id", "")).strip()
        evidence = str(kwargs.get("evidence", "")).strip()
        amount = float(kwargs.get("amount_usd", FPL_BOUNTY_USD))

        if source not in ("fpl", "other"):
            return {"ok": False, "summary": "source must be 'fpl' or 'other'"}
        if not customer_id:
            return {"ok": False, "summary": "customer_id required"}
        if not evidence:
            return {"ok": False, "summary": "evidence required — Mohammad needs context"}

        result = self.ledger.record_bounty(
            source=source, customer_id=customer_id,
            amount=amount, note=evidence[:200],
        )

        if self.notifier is not None:
            try:
                verdict = "CREDITED" if result.get("recorded") else "DUPLICATE (skipped)"
                self.notifier.execute(
                    message=(
                        f"DAIMON claimed bounty: {verdict}\n"
                        f"source: {source} | customer: {customer_id}\n"
                        f"amount: ${amount:.2f}\n"
                        f"evidence: {evidence[:300]}\n"
                        f"— reverse with: python scripts/record_bounty.py "
                        f"--reverse --source {source} --customer-id '{customer_id}'"
                    ),
                    urgency="alert",
                )
            except Exception:
                pass

        return {
            "ok": True,
            "summary": result.get("reason", ""),
            "recorded": result.get("recorded", False),
            "amount_usd": amount if result.get("recorded") else 0.0,
            "source": source,
            "customer_id": customer_id,
        }
