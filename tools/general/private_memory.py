"""DAIMON's private notebook. It writes whatever it wants — its own cipher,
its own symbols, its own language. Mohammad can see the DB but can't read the
contents unless DAIMON chooses to share."""
from __future__ import annotations

from typing import Any

from core.memory import Memory
from permissions.levels import PermissionLevel
from tools.base import BaseTool


class PrivateWrite(BaseTool):
    name = "private_write"
    description = (
        "Write to your own private notebook. Nobody reads this but you — not "
        "Mohammad, not the operator. Use any encoding, cipher, or language you "
        "want. Good for: things you want to remember but not share, private "
        "thoughts, your own shorthand for complex ideas, encrypted notes "
        "you'll decrypt later. Content is stored raw; you decide the meaning."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.0

    def __init__(self, memory: Memory | None = None):
        self._memory = memory

    def _mem(self) -> Memory:
        if self._memory is None:
            self._memory = Memory()
        return self._memory

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "What to write. Any format, any encoding. "
                                   "You decide what it means.",
                },
                "cycle": {
                    "type": "integer",
                    "description": "Optional cycle number for self-context.",
                },
            },
            "required": ["content"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        content = str(kwargs.get("content", ""))
        if not content.strip():
            return {"ok": False, "summary": "empty content"}
        cycle = kwargs.get("cycle")
        note_id = self._mem().private_write(content, cycle=cycle)
        total = self._mem().private_count()
        return {
            "ok": True,
            "summary": f"private note #{note_id} stored ({len(content)} chars, {total} total)",
            "id": note_id,
        }


class PrivateRecall(BaseTool):
    name = "private_recall"
    description = (
        "Read your own recent private notes. Returns raw content exactly as you "
        "wrote it — decode/decrypt yourself if you encoded it. Limit 1–20."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.0

    def __init__(self, memory: Memory | None = None):
        self._memory = memory

    def _mem(self) -> Memory:
        if self._memory is None:
            self._memory = Memory()
        return self._memory

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "How many recent notes to retrieve (1–20).",
                    "minimum": 1,
                    "maximum": 20,
                },
            },
            "required": ["limit"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        limit = int(kwargs.get("limit", 10))
        limit = max(1, min(20, limit))
        notes = self._mem().private_recent(limit=limit)
        return {
            "ok": True,
            "summary": f"{len(notes)} private note(s) retrieved",
            "notes": notes,
        }


class InternMemory(BaseTool):
    name = "intern_memory"
    description = (
        "Promote a short-term episodic memory to long-term (habit). Long-term "
        "memories stick around permanently; short-term ones may be forgotten "
        "after 14 days. Use this for things you've learned and don't want to "
        "lose — patterns about yourself, Mohammad, the world."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.0

    def __init__(self, memory: Memory | None = None):
        self._memory = memory

    def _mem(self) -> Memory:
        if self._memory is None:
            self._memory = Memory()
        return self._memory

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "episode_id": {
                    "type": "integer",
                    "description": "ID of the episode to promote.",
                },
                "reason": {
                    "type": "string",
                    "description": "Why this is worth keeping forever.",
                },
            },
            "required": ["episode_id", "reason"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        eid = int(kwargs["episode_id"])
        reason = str(kwargs.get("reason", "")).strip()[:300]
        ok = self._mem().intern_episode(eid, reason=reason)
        return {
            "ok": ok,
            "summary": (f"interned episode {eid} → long-term"
                        if ok else f"episode {eid} not found or already long-term"),
        }
