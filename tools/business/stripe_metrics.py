"""Read-side Stripe: MRR, active subs, recent revenue, new customers, churn.

Closes the loop on business health: is Centsibles actually growing?"""
from __future__ import annotations

import os
import time
from typing import Any

import httpx

from permissions.levels import PermissionLevel
from tools.base import BaseTool


STRIPE_API = "https://api.stripe.com/v1"


def _auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}",
            "Stripe-Version": "2024-06-20"}


class StripeMetrics(BaseTool):
    name = "stripe_metrics"
    description = (
        "Pull Centsibles' Stripe snapshot: active subscription count, MRR, "
        "recent new subs, recent charges (last N days revenue), recent "
        "cancellations. Use this to see whether Centsibles is growing. "
        "If MRR is flat after you ran an experiment, the experiment failed."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.0

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Look-back window for new/churned subs and "
                                   "charges (default 30).",
                    "default": 30,
                },
            },
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        key = os.getenv("STRIPE_SECRET_KEY", "").strip()
        if not key:
            return {
                "ok": False,
                "summary": "no STRIPE_SECRET_KEY",
                "needs_resource": "STRIPE_SECRET_KEY",
            }

        days = int(kwargs.get("days", 30))
        since = int(time.time() - days * 86400)
        h = _auth(key)

        try:
            # Active subscriptions (status=active)
            active_count = 0
            mrr_cents = 0
            currency = "usd"
            starting_after = None
            for _ in range(10):  # cap pagination
                params: dict[str, Any] = {"status": "active", "limit": 100,
                                          "expand[]": "data.items.data.price"}
                if starting_after:
                    params["starting_after"] = starting_after
                r = httpx.get(f"{STRIPE_API}/subscriptions", headers=h,
                              params=params, timeout=20.0)
                r.raise_for_status()
                data = r.json()
                for sub in data.get("data", []):
                    active_count += 1
                    for item in (sub.get("items") or {}).get("data", []):
                        price = item.get("price") or {}
                        unit = price.get("unit_amount") or 0
                        qty = item.get("quantity", 1)
                        recur = price.get("recurring") or {}
                        interval = recur.get("interval", "month")
                        interval_count = recur.get("interval_count", 1) or 1
                        currency = price.get("currency", currency)
                        # normalize to monthly
                        if interval == "month":
                            monthly = unit * qty / interval_count
                        elif interval == "year":
                            monthly = unit * qty / (12 * interval_count)
                        elif interval == "week":
                            monthly = unit * qty * (52.0 / 12.0) / interval_count
                        elif interval == "day":
                            monthly = unit * qty * (365.0 / 12.0) / interval_count
                        else:
                            monthly = 0
                        mrr_cents += monthly
                if not data.get("has_more"):
                    break
                starting_after = data["data"][-1]["id"] if data.get("data") else None

            # Recent new subs (created in window)
            new_subs = httpx.get(
                f"{STRIPE_API}/subscriptions", headers=h, timeout=15.0,
                params={"status": "all", "limit": 100,
                        "created[gte]": since},
            )
            new_subs.raise_for_status()
            new_list = new_subs.json().get("data", [])
            new_count = len(new_list)
            cancelled_count = sum(1 for s in new_list
                                  if s.get("canceled_at") and s["canceled_at"] >= since)
            # Also count cancellations of older subs in window
            cancelled = httpx.get(
                f"{STRIPE_API}/subscriptions", headers=h, timeout=15.0,
                params={"status": "canceled", "limit": 100,
                        "canceled_at[gte]": since}
                        if False else {"status": "canceled", "limit": 100},
            )
            cancelled.raise_for_status()
            cancelled_in_window = sum(
                1 for s in cancelled.json().get("data", [])
                if (s.get("canceled_at") or 0) >= since
            )

            # Recent charges (successful only)
            charges = httpx.get(
                f"{STRIPE_API}/charges", headers=h, timeout=15.0,
                params={"limit": 100, "created[gte]": since},
            )
            charges.raise_for_status()
            charge_data = charges.json().get("data", [])
            revenue_cents = sum(
                c.get("amount", 0) for c in charge_data
                if c.get("status") == "succeeded" and not c.get("refunded")
            )
            refund_cents = sum(
                c.get("amount_refunded", 0) for c in charge_data
            )

            # Balance
            bal = httpx.get(f"{STRIPE_API}/balance", headers=h, timeout=15.0)
            bal.raise_for_status()
            bdata = bal.json()
            available = sum(b.get("amount", 0) for b in bdata.get("available", []))
            pending = sum(b.get("amount", 0) for b in bdata.get("pending", []))

            mrr = round(mrr_cents / 100, 2)
            return {
                "ok": True,
                "summary": (
                    f"Centsibles: {active_count} active subs, "
                    f"MRR {currency.upper()} ${mrr:.2f}. "
                    f"Last {days}d: {new_count} new, "
                    f"{cancelled_in_window} cancelled, "
                    f"revenue ${revenue_cents/100:.2f}, "
                    f"refunds ${refund_cents/100:.2f}."
                ),
                "currency": currency,
                "active_subs": active_count,
                "mrr": mrr,
                "window_days": days,
                "new_subs_in_window": new_count,
                "cancelled_in_window": cancelled_in_window,
                "revenue_in_window": round(revenue_cents / 100, 2),
                "refunds_in_window": round(refund_cents / 100, 2),
                "balance_available": round(available / 100, 2),
                "balance_pending": round(pending / 100, 2),
            }
        except httpx.HTTPStatusError as e:
            return {
                "ok": False,
                "summary": f"stripe http {e.response.status_code}: "
                           f"{e.response.text[:300]}",
            }
        except httpx.HTTPError as e:
            return {"ok": False, "summary": f"stripe error: {e}"}
