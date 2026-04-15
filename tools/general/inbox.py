"""Tools for DAIMON to read Mohammad's Telegram replies."""
from __future__ import annotations

from typing import Any

from core.telegram_inbox import TelegramInbox
from permissions.levels import PermissionLevel
from tools.base import BaseTool


class ReadInbox(BaseTool):
    name = "read_inbox"
    description = (
        "Read messages Mohammad has sent you via Telegram. Returns unread "
        "messages by default (oldest first so you can follow threads). Set "
        "mark_read=true to mark them as read after you've processed them. "
        "This is how you hear back from your operator — use it whenever your "
        "observations show inbox_unread > 0."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.0

    def __init__(self, inbox: TelegramInbox | None = None):
        self._inbox = inbox

    def _i(self) -> TelegramInbox:
        if self._inbox is None:
            self._inbox = TelegramInbox()
        return self._inbox

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["unread", "recent"],
                    "description": "'unread' = only new messages. 'recent' = last N "
                                   "including already-read (for context).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max messages to return, 1–50.",
                    "minimum": 1,
                    "maximum": 50,
                },
                "mark_read": {
                    "type": "boolean",
                    "description": "If true, mark returned messages as read.",
                },
            },
            "required": ["mode", "limit", "mark_read"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        mode = kwargs.get("mode", "unread")
        limit = max(1, min(50, int(kwargs.get("limit", 20))))
        mark = bool(kwargs.get("mark_read", False))
        inbox = self._i()
        if mode == "unread":
            msgs = inbox.unread(limit=limit)
        else:
            msgs = inbox.recent(limit=limit)
        if mark and msgs:
            ids = [m["id"] for m in msgs if not m.get("read")]
            if ids:
                inbox.mark_read(ids)
        return {
            "ok": True,
            "summary": f"{len(msgs)} message(s) ({mode})",
            "messages": msgs,
            "marked_read": mark,
        }
