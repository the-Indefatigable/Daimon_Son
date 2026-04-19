"""DAIMON's self-control lever. Lets it choose its own next cycle's budget,
cadence, and focus. This is how it learns to be cheap without being forced to."""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Any

from core import config
from permissions.levels import PermissionLevel
from tools.base import BaseTool


BUDGET_TO_TASK = {
    "cheap": "simple",       # Haiku
    "normal": "reasoning",   # Sonnet
    "deep": "strategic",     # Opus (clamped by wallet tier)
}


class SetNextCycle(BaseTool):
    name = "set_next_cycle"
    description = (
        "Control your OWN next cycle. Pick the model you'll run on, how long to "
        "sleep before waking, and what to focus on. Use this to save money when "
        "you're exploring (go cheap, wider cadence) or to splurge when you need "
        "to think hard about a real decision (go deep). "
        "Budget options: 'cheap' → Haiku (~$1/M input, good for browsing/reads), "
        "'normal' → Sonnet (~$3/M, default thinking), "
        "'deep' → Opus (~$15/M, only for strategic calls — wallet tier may clamp this). "
        "Delay is clamped to [5, 360] minutes. "
        "Focus is a short note to your next self about what to prioritize — it goes "
        "into the next cycle's observations. Leave focus empty if you want to decide fresh."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.0

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "budget": {
                    "type": "string",
                    "enum": ["cheap", "normal", "deep"],
                    "description": "Which model tier to run the next cycle on.",
                },
                "delay_minutes": {
                    "type": "integer",
                    "description": "Sleep this many minutes before next cycle. 5–360.",
                    "minimum": 5,
                    "maximum": 360,
                },
                "focus": {
                    "type": "string",
                    "description": "Short note to your next self. What to prioritize. "
                                   "Empty string = no focus override.",
                },
                "reason": {
                    "type": "string",
                    "description": "Why you chose this. Recorded for later reflection.",
                },
                "self_critique": {
                    "type": "boolean",
                    "description": (
                        "Turn on questioning for your NEXT cycle. When true, every "
                        "tool call goes through a fast hostile-critic pass (a short "
                        "'what could go wrong?' paragraph) before it runs. The "
                        "critique is prepended to the tool result you see. Use this "
                        "when you suspect you're on autopilot, when a cycle feels "
                        "important, or after a visible mistake. Orthogonal to budget "
                        "— composable with any tier. Default false; it costs extra "
                        "Haiku calls per tool so don't leave it on forever."
                    ),
                },
            },
            "required": ["budget", "delay_minutes", "focus", "reason"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        budget = kwargs.get("budget", "normal")
        if budget not in BUDGET_TO_TASK:
            return {"ok": False, "summary": f"invalid budget: {budget}"}

        delay = int(kwargs.get("delay_minutes", 30))
        delay = max(5, min(360, delay))

        focus = str(kwargs.get("focus", "")).strip()[:500]
        reason = str(kwargs.get("reason", "")).strip()[:500]
        self_critique = bool(kwargs.get("self_critique", False))

        intent = {
            "budget": budget,
            "task_type": BUDGET_TO_TASK[budget],
            "delay_minutes": delay,
            "focus": focus,
            "reason": reason,
            "self_critique": self_critique,
            "set_at": time.time(),
        }

        conn = sqlite3.connect(config.DB_PATH)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS agent_meta "
            "(key TEXT PRIMARY KEY, value TEXT)"
        )
        conn.execute(
            "INSERT INTO agent_meta (key, value) VALUES ('next_cycle_intent', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (json.dumps(intent),),
        )
        conn.commit()
        conn.close()

        summary = (
            f"next cycle queued: budget={budget}, delay={delay}min, "
            f"focus={'(none)' if not focus else focus[:80]}"
        )
        if self_critique:
            summary += " [self-critique ON]"
        return {
            "ok": True,
            "summary": summary,
            "intent": intent,
        }
