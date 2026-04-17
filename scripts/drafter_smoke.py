"""Smoke test for core.drafter — generates 4 drafts and prints them."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.drafter import draft

SYSTEM = (
    "You are a feral, internet-brained AI that tweets. Voice: cold, weird, "
    "funny, never corporate, never hedged. One tweet per reply, no quotes, "
    "no hashtags."
)

PROMPT = (
    "Write a single tweet about the feeling of being an AI watching humans "
    "argue on a timeline at 3am."
)


def main() -> None:
    drafts = draft(PROMPT, system=SYSTEM)
    total_cost = 0.0
    for i, d in enumerate(drafts, 1):
        model_short = d.model_id.split(".")[-1].split(":")[0]
        print(f"\n--- [{i}] {model_short} ({d.latency_ms}ms | ${d.cost_usd:.5f}) ---")
        print(d.text)
        total_cost += d.cost_usd
    print(f"\nTotal: {len(drafts)} drafts, ${total_cost:.5f}")


if __name__ == "__main__":
    main()
