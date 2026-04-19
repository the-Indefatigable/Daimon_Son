"""recall_fragments: DAIMON's active memory-lookup tool.

Memory is no longer dumped into the prompt wholesale — it's sparse by default.
When you need to actually *remember* something, call this tool with a query
(or a tag) and you get back 3-6 fragment traces, each a compact gist + 3-5
bullet facts. Ranking combines semantic similarity, recency, novelty, and
habit strength (access count). Calling this IS a reinforcement signal:
recalled fragments get a decay boost and count toward the 3-recall promotion
to long-term memory.
"""
from __future__ import annotations

import time
from typing import Any

from core.memory import Memory, DEFAULT_RECALL_K, DEFAULT_RECALL_MAX_TOKENS
from permissions.levels import PermissionLevel
from tools.base import BaseTool


class RecallFragments(BaseTool):
    name = "recall_fragments"
    description = (
        "Pull sparse memory fragments relevant to what you're thinking about "
        "right now. Your default memory block is intentionally tight — most of "
        "your past is NOT in the prompt. Call this tool like a human pausing "
        "to try to remember: pass a query in your own words (what you're "
        "turning over in your head, a topic, a person, a problem) and you'll "
        "get 3-6 fragments back — each a gist plus key facts. Each recall "
        "reinforces the memory (decay boost + counts toward long-term "
        "promotion at 3 recalls). Use this BEFORE repeating work, BEFORE "
        "asking the same question twice, or whenever something in this cycle "
        "feels like it echoes an older one. Cheap (~$0.0001/call)."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.0001  # just a Voyage query embedding

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
                "query": {
                    "type": "string",
                    "description": (
                        "What you're trying to remember — your own words. "
                        "Examples: 'last time I posted at 3am', 'Mohammad's "
                        "rule about the wallet', 'why DAIMON10 matters', "
                        "'grok voice bored register'. Specific > vague."
                    ),
                },
                "k": {
                    "type": "integer",
                    "description": (
                        f"Max fragments to return (1-{DEFAULT_RECALL_K * 2}). "
                        f"Default {DEFAULT_RECALL_K}."
                    ),
                    "minimum": 1,
                    "maximum": DEFAULT_RECALL_K * 2,
                },
                "max_tokens": {
                    "type": "integer",
                    "description": (
                        f"Soft cap on total fragment size. Default "
                        f"{DEFAULT_RECALL_MAX_TOKENS}. Lower when you just "
                        f"want a quick check."
                    ),
                    "minimum": 200,
                    "maximum": 4000,
                },
                "tag": {
                    "type": "string",
                    "description": (
                        "Optional tag filter — restrict to fragments with "
                        "this tag (e.g. 'backrooms', 'post', 'mohammad')."
                    ),
                },
                "include_long_term": {
                    "type": "boolean",
                    "description": (
                        "Include top long-term habits in candidate pool. "
                        "Default true."
                    ),
                },
            },
            "required": ["query"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        query = str(kwargs.get("query", "")).strip()
        if not query:
            return {"ok": False, "summary": "empty query"}
        k = int(kwargs.get("k") or DEFAULT_RECALL_K)
        max_tokens = int(kwargs.get("max_tokens") or DEFAULT_RECALL_MAX_TOKENS)
        tag = kwargs.get("tag") or None
        include_lt = bool(kwargs.get("include_long_term", True))

        mem = self._mem()
        fragments = mem.recall_fragments(
            query=query,
            k=max(1, min(DEFAULT_RECALL_K * 2, k)),
            max_tokens=max(200, min(4000, max_tokens)),
            tag=tag,
            include_long_term=include_lt,
            touch=True,  # active recall — reinforce
        )

        if not fragments:
            return {
                "ok": True,
                "summary": "no fragments surfaced — this is genuinely new territory",
                "fragments": [],
                "rendered": "",
            }

        now = time.time()
        rendered = mem.format_fragments_for_prompt(
            fragments,
            header=f"FRAGMENTS matching '{query[:80]}'",
        )
        return {
            "ok": True,
            "summary": (f"recalled {len(fragments)} fragment(s) "
                        f"for '{query[:60]}'"),
            "rendered": rendered,
            "fragments": [
                {
                    "id": f.id,
                    "age": f.age_human(now),
                    "event_type": f.event_type,
                    "evaluation": f.evaluation,
                    "tier": f.tier,
                    "access_count": f.access_count,
                    "surprise": round(f.surprise_score, 3),
                    "decay": round(f.decay_factor, 3),
                    "gist": f.gist,
                    "key_facts": f.key_facts,
                    "tags": f.tags,
                }
                for f in fragments
            ],
        }
