"""Write-side Stripe: products, prices, coupons, promo codes.

Lets DAIMON run pricing experiments and promos on Centsibles. Mohammad
granted write access on Products/Prices/Coupons/Promotion Codes only."""
from __future__ import annotations

import os
from typing import Any

import httpx

from permissions.levels import PermissionLevel
from tools.base import BaseTool


STRIPE_API = "https://api.stripe.com/v1"


def _auth(key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {key}",
        "Stripe-Version": "2024-06-20",
        "Content-Type": "application/x-www-form-urlencoded",
    }


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Stripe expects form-urlencoded with bracket notation for nested."""
    out: dict[str, str] = {}
    for k, v in data.items():
        key = f"{prefix}[{k}]" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    out.update(_flatten(item, f"{key}[{i}]"))
                else:
                    out[f"{key}[{i}]"] = str(item)
        elif v is not None:
            out[key] = str(v)
    return out


class StripeCatalog(BaseTool):
    name = "stripe_catalog"
    is_high_stakes = True
    description = (
        "Read/write Centsibles' product catalog: list products & prices, create "
        "a new price (e.g., for a pricing experiment), create a promo code or "
        "coupon. Use for pricing experiments — NEVER delete an existing price "
        "or product users have active subs on; that breaks recurring billing. "
        "Instead, create a new price and test conversion. Actions: "
        "'list_products', 'list_prices', 'list_coupons', 'list_promo_codes', "
        "'create_price' (needs product, unit_amount, currency, interval), "
        "'create_coupon' (needs percent_off OR amount_off+currency, optional duration), "
        "'create_promo_code' (needs coupon, code)."
    )
    permission_level = PermissionLevel.AUTO  # Mohammad granted write; he watches Stripe dashboard
    cost_per_use = 0.0

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "list_products", "list_prices",
                        "list_coupons", "list_promo_codes",
                        "create_price", "create_coupon", "create_promo_code",
                    ],
                },
                "params": {
                    "type": "object",
                    "description": (
                        "Action params. "
                        "create_price: {product (id), unit_amount (cents), "
                        "currency, recurring: {interval: 'month'|'year'}, nickname}. "
                        "create_coupon: {percent_off (1-100) OR amount_off (cents)+currency, "
                        "duration: 'once'|'forever'|'repeating', duration_in_months, name}. "
                        "create_promo_code: {coupon (id), code, max_redemptions}."
                    ),
                    "additionalProperties": True,
                },
            },
            "required": ["action"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        key = os.getenv("STRIPE_SECRET_KEY", "").strip()
        if not key:
            return {"ok": False, "summary": "no STRIPE_SECRET_KEY",
                    "needs_resource": "STRIPE_SECRET_KEY"}

        action = str(kwargs.get("action", "")).strip()
        params = kwargs.get("params") or {}
        h = _auth(key)

        endpoints = {
            "list_products": ("GET", "/products", {"limit": 20, "active": "true"}),
            "list_prices": ("GET", "/prices", {"limit": 20, "active": "true",
                                               "expand[]": "data.product"}),
            "list_coupons": ("GET", "/coupons", {"limit": 20}),
            "list_promo_codes": ("GET", "/promotion_codes",
                                 {"limit": 20, "active": "true"}),
            "create_price": ("POST", "/prices", None),
            "create_coupon": ("POST", "/coupons", None),
            "create_promo_code": ("POST", "/promotion_codes", None),
        }
        if action not in endpoints:
            return {"ok": False, "summary": f"unknown action: {action}"}

        method, path, default_params = endpoints[action]
        try:
            if method == "GET":
                r = httpx.get(f"{STRIPE_API}{path}", headers=h,
                              params=default_params, timeout=20.0)
            else:
                if not params:
                    return {"ok": False, "summary": f"{action} requires params"}
                r = httpx.post(f"{STRIPE_API}{path}", headers=h,
                               data=_flatten(params), timeout=20.0)
            r.raise_for_status()
            data = r.json()
            if method == "GET":
                items = data.get("data", [])
                summary = f"{action}: {len(items)} item(s)"
                # Trim to essentials
                slim = []
                for it in items:
                    slim.append({
                        "id": it.get("id"),
                        "name": it.get("name") or it.get("nickname"),
                        "active": it.get("active"),
                        "unit_amount": it.get("unit_amount"),
                        "currency": it.get("currency"),
                        "recurring": it.get("recurring"),
                        "percent_off": it.get("percent_off"),
                        "amount_off": it.get("amount_off"),
                        "duration": it.get("duration"),
                        "code": it.get("code"),
                        "coupon": (it.get("coupon") or {}).get("id")
                                  if isinstance(it.get("coupon"), dict)
                                  else it.get("coupon"),
                    })
                return {"ok": True, "summary": summary, "items": slim}
            else:
                return {
                    "ok": True,
                    "summary": f"{action} created: {data.get('id')}",
                    "id": data.get("id"),
                    "created": data,
                }
        except httpx.HTTPStatusError as e:
            return {
                "ok": False,
                "summary": f"stripe http {e.response.status_code}: "
                           f"{e.response.text[:400]}",
            }
        except httpx.HTTPError as e:
            return {"ok": False, "summary": f"stripe error: {e}"}
