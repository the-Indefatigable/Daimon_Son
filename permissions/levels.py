"""Permission tiers for tool execution."""
from __future__ import annotations

from enum import Enum


class PermissionLevel(str, Enum):
    AUTO = "auto"           # DAIMON runs it without asking
    NOTIFY = "notify"       # Runs it, tells Mohammad after
    APPROVAL = "approval"   # Waits for Mohammad's yes/no before running


# Spending thresholds (USD) that escalate permission regardless of tool default.
NOTIFY_SPEND_THRESHOLD = 50.0
APPROVAL_SPEND_THRESHOLD = 200.0


def escalate_for_spend(base_level: PermissionLevel, spend_usd: float) -> PermissionLevel:
    if spend_usd >= APPROVAL_SPEND_THRESHOLD:
        return PermissionLevel.APPROVAL
    if spend_usd >= NOTIFY_SPEND_THRESHOLD and base_level == PermissionLevel.AUTO:
        return PermissionLevel.NOTIFY
    return base_level
