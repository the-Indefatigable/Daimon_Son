"""record_outcome: pair to expect_result. DAIMON grades its own prediction
against reality. The surprise field is the salience signal — high surprise
boosts memory weight and feeds the schema layer for that principle."""
from __future__ import annotations

from typing import Any

from core.expectations import Expectations
from core.memory import Memory
from permissions.levels import PermissionLevel
from tools.base import BaseTool


class RecordOutcome(BaseTool):
    name = "record_outcome"
    description = (
        "Close the loop on a pending expectation. Record what actually happened "
        "and how surprised you are. Surprise is what teaches you — record it honestly. "
        "Also writes a strategic-memory insight when surprise ≥ 0.5 OR when a "
        "principle is named, so the lesson sticks across cycles. "
        "Use this when an expectation appears in 'due_now' or 'just_expired' "
        "in your observations."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.0

    def __init__(self, expectations: Expectations, memory: Memory):
        self._exp = expectations
        self._memory = memory

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expectation_id": {
                    "type": "integer",
                    "description": "ID of the expectation you're closing.",
                },
                "actual_value": {
                    "type": "string",
                    "description": "What actually happened, in concrete terms. "
                                   "e.g. '1 like, 0 replies', 'PR merged', "
                                   "'MRR went from $124 to $128 (+3.2%)', "
                                   "'no redemptions'. Empty answer = silence; "
                                   "say so plainly.",
                },
                "surprise": {
                    "type": "number",
                    "description": "How wrong was your prediction? "
                                   "0.0 = called it exactly. "
                                   "0.3 = roughly right, off in degree. "
                                   "0.5 = wrong direction or wrong magnitude. "
                                   "0.8 = totally missed. "
                                   "1.0 = opposite of what you predicted. "
                                   "Be honest — the surprise number is what teaches you.",
                    "minimum": 0,
                    "maximum": 1,
                },
                "lesson": {
                    "type": "string",
                    "description": "One sentence — what you now believe after seeing "
                                   "the result. This becomes a strategic insight if "
                                   "surprise ≥ 0.5 or principle is named. Empty "
                                   "string if no clear lesson yet.",
                },
            },
            "required": ["expectation_id", "actual_value", "surprise", "lesson"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        eid = int(kwargs["expectation_id"])
        actual = str(kwargs["actual_value"]).strip()[:1000]
        surprise = float(kwargs["surprise"])
        lesson = str(kwargs.get("lesson", "")).strip()[:500]

        row = self._exp.get(eid)
        if not row:
            return {"ok": False, "summary": f"no expectation with id={eid}"}
        if row["status"] not in ("pending", "expired"):
            return {"ok": False,
                    "summary": f"expectation {eid} already {row['status']}"}

        # If row is 'expired', we still want to record — flip it back to pending
        # for the duration of the update (record_outcome only updates pending rows).
        if row["status"] == "expired":
            self._exp._conn.execute(
                "UPDATE expectations SET status='pending' WHERE id=?", (eid,)
            )

        ok = self._exp.record_outcome(
            expectation_id=eid,
            actual_value=actual,
            surprise=surprise,
            notes=lesson or None,
        )
        if not ok:
            return {"ok": False, "summary": f"failed to update expectation {eid}"}

        # Promote to strategic memory when the lesson matters.
        strategic_id = None
        if lesson and (surprise >= 0.5 or row.get("principle")):
            category = row.get("principle") or "behavior"
            confidence = 1.0 - min(0.5, surprise * 0.5)  # surprise → less confident in old prior
            insight = (
                f"{lesson} "
                f"[predicted {row['predicted_metric']}={row['predicted_value']}, "
                f"got {actual[:120]}; surprise={surprise:.2f}]"
            )
            strategic_id = self._memory.store_strategic(
                category=category,
                insight=insight,
                confidence=confidence,
            )

        return {
            "ok": True,
            "summary": (
                f"outcome recorded for expectation {eid} "
                f"(surprise={surprise:.2f})"
                + (f"; strategic insight #{strategic_id} stored under "
                   f"'{row.get('principle') or 'behavior'}'" if strategic_id else "")
            ),
            "strategic_insight_id": strategic_id,
        }
