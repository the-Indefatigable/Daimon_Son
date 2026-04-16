"""expect_result: DAIMON commits a prediction at do-time so it can learn at
result-time. Predictions you didn't write down can't be wrong → can't teach."""
from __future__ import annotations

from typing import Any

from core.expectations import Expectations
from permissions.levels import PermissionLevel
from tools.base import BaseTool


class ExpectResult(BaseTool):
    name = "expect_result"
    description = (
        "Record a prediction about an action you just took (or are about to take). "
        "Use this BEFORE you check for results. It pins down what you predicted, "
        "what metric matters, and when results are reasonable to expect — so you "
        "stop anxious-refreshing every cycle and your predictions can actually be "
        "graded later. "
        "Call this for: PRs you opened, posts you shipped, price/promo changes, "
        "replies you sent — anything where 'did it work?' is a real question. "
        "Do NOT call for read-only actions or maintenance noops. "
        "After this, do NOT call check tools (pr_status / bluesky_read / metrics) "
        "for this action until your observations show it as 'due_now'. When the "
        "window opens, run the check, then call record_outcome."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.0

    def __init__(self, expectations: Expectations, get_cycle):
        self._exp = expectations
        self._get_cycle = get_cycle  # callable -> int (current cycle)

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action_kind": {
                    "type": "string",
                    "enum": [
                        "pr", "bluesky_post", "bluesky_reply",
                        "price_change", "promo_code", "blog_post",
                        "twitter_post", "other",
                    ],
                    "description": "What kind of action you took.",
                },
                "action_summary": {
                    "type": "string",
                    "description": "One sentence — what you actually did, in your "
                                   "own words. e.g. 'PR #7 changes hero headline "
                                   "from feature-list to outcome-promise'.",
                },
                "action_ref": {
                    "type": "string",
                    "description": "Identifier for the action. PR number, post URI, "
                                   "price_id, promo code — whatever lets future-you "
                                   "find it. Empty string if there isn't one.",
                },
                "predicted_metric": {
                    "type": "string",
                    "description": "What metric you'll grade this on. e.g. "
                                   "'pr_merged', 'likes_24h', 'replies_48h', "
                                   "'mrr_lift_7d', 'promo_redemptions_7d', "
                                   "'gsc_clicks_lift_14d'.",
                },
                "predicted_value": {
                    "type": "string",
                    "description": "What you predict for that metric. Be specific "
                                   "and falsifiable. e.g. 'true', '≥3', '+5%', "
                                   "'between 1 and 5'. A vague prediction = no "
                                   "prediction.",
                },
                "predicted_basis": {
                    "type": "string",
                    "description": "WHY you predict that. The hypothesis. e.g. "
                                   "'question-format posts beat statements in "
                                   "early accounts because they invite reply'. "
                                   "This is what gets graded against reality.",
                },
                "principle": {
                    "type": "string",
                    "description": "Named human-behavior principle this experiment "
                                   "tests, if any. e.g. 'scarcity', 'left_digit', "
                                   "'social_proof', 'loss_aversion', 'anchoring', "
                                   "'decoy', 'reciprocity'. Empty string if not "
                                   "tied to a principle (e.g. plain bug fix).",
                },
                "check_after_hours": {
                    "type": "number",
                    "description": "Earliest meaningful check time, in hours from "
                                   "now. PR merges: 6–24h. Bluesky engagement: "
                                   "24–48h. MRR lift: 7d=168h. Don't check before "
                                   "this — it's noise.",
                    "minimum": 0.5,
                    "maximum": 720,
                },
                "check_before_hours": {
                    "type": "number",
                    "description": "After this, attribution is too noisy → "
                                   "expectation auto-expires. Usually 2–4× "
                                   "check_after_hours. 0 = no expiry.",
                    "minimum": 0,
                    "maximum": 2160,
                },
            },
            "required": [
                "action_kind", "action_summary", "action_ref",
                "predicted_metric", "predicted_value", "predicted_basis",
                "principle", "check_after_hours", "check_before_hours",
            ],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        action_kind = str(kwargs["action_kind"]).strip()
        action_summary = str(kwargs["action_summary"]).strip()[:500]
        action_ref = str(kwargs.get("action_ref", "")).strip() or None
        predicted_metric = str(kwargs["predicted_metric"]).strip()[:100]
        predicted_value = str(kwargs["predicted_value"]).strip()[:200]
        predicted_basis = str(kwargs.get("predicted_basis", "")).strip()[:500] or None
        principle = str(kwargs.get("principle", "")).strip().lower() or None
        check_after = float(kwargs["check_after_hours"])
        check_before_raw = float(kwargs.get("check_before_hours", 0))
        check_before = check_before_raw if check_before_raw > 0 else None

        if not action_summary:
            return {"ok": False, "summary": "action_summary is required"}
        if not predicted_metric or not predicted_value:
            return {"ok": False, "summary": "predicted_metric and predicted_value required"}
        if check_before is not None and check_before <= check_after:
            return {"ok": False,
                    "summary": "check_before_hours must be > check_after_hours"}

        eid = self._exp.create(
            cycle=self._get_cycle(),
            action_kind=action_kind,
            action_ref=action_ref,
            action_summary=action_summary,
            predicted_metric=predicted_metric,
            predicted_value=predicted_value,
            predicted_basis=predicted_basis,
            principle=principle,
            check_after_hours=check_after,
            check_before_hours=check_before,
        )
        return {
            "ok": True,
            "expectation_id": eid,
            "summary": (
                f"prediction recorded (id={eid}): {predicted_metric}={predicted_value} "
                f"for {action_kind}; check after {check_after}h"
                + (f", expires after {check_before}h" if check_before else "")
            ),
        }
