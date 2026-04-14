"""Telegram notifier. Also supports a stdout fallback when no token is set."""
from __future__ import annotations

from typing import Any

import httpx

from core import config
from permissions.levels import PermissionLevel
from tools.base import BaseTool


class TelegramNotifier(BaseTool):
    name = "notify_mohammad"
    description = (
        "Send a Telegram message to Mohammad. Use this for status updates, "
        "resource requests, approval requests, daily summaries, brain upgrades/downgrades, "
        "and any time he needs to know something right now. Keep messages concise."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.0  # Telegram Bot API is free

    def __init__(self, bot_token: str | None = None, chat_id: str | None = None):
        self.bot_token = bot_token or config.TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or config.TELEGRAM_CHAT_ID

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Message body. Supports emojis. Under 4000 chars.",
                },
                "urgency": {
                    "type": "string",
                    "enum": ["info", "alert", "critical"],
                    "description": "Prefixed to the message for Mohammad's triage.",
                    "default": "info",
                },
            },
            "required": ["message"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        message = kwargs.get("message", "").strip()
        urgency = kwargs.get("urgency", "info")
        if not message:
            return {"ok": False, "summary": "empty message"}

        prefix = {"info": "", "alert": "⚠️ ", "critical": "🚨 "}.get(urgency, "")
        body = f"{prefix}{message}"[:4000]

        if not self.bot_token or not self.chat_id:
            # Fallback: print to stdout so dev/dry-run still sees it
            print(f"\n[telegram/fallback] {body}\n")
            return {
                "ok": True,
                "summary": "printed to stdout (no telegram credentials)",
                "delivered_via": "stdout",
            }

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        try:
            resp = httpx.post(
                url,
                json={"chat_id": self.chat_id, "text": body},
                timeout=10.0,
            )
            resp.raise_for_status()
            return {
                "ok": True,
                "summary": f"sent to telegram chat {self.chat_id}",
                "delivered_via": "telegram",
            }
        except httpx.HTTPError as e:
            print(f"[telegram error] {e} — message was: {body}")
            return {
                "ok": False,
                "summary": f"telegram error: {e}",
                "fallback_body": body,
            }
