"""End-to-end smoke test: drafter -> judge. Prints the slate, Claude's pick,
reasoning, and the total pipeline cost."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.drafter import draft
from core.judge import judge

SYSTEM = (
    "You are DAIMON, a feral, internet-brained AI that posts on a timeline. "
    "Voice: specific, weird, never corporate, never hedged. No em-dashes. "
    "One tweet per reply."
)

PROMPT = (
    "Write a single tweet about the feeling of being an AI watching humans "
    "argue on a timeline at 3am."
)


def main() -> None:
    print("Generating drafts...")
    drafts = draft(PROMPT, system=SYSTEM)

    print(f"\n=== SLATE ({len(drafts)} drafts) ===")
    for i, d in enumerate(drafts, 1):
        short = d.model_id.split(".")[-1].split(":")[0]
        print(f"\n[{i}] {short}")
        print(f"    {d.text}")

    print("\n\nJudging...")
    result = judge(drafts, context=f"System: {SYSTEM}\n\nPrompt: {PROMPT}")

    winner_short = result.winner.model_id.split(".")[-1].split(":")[0]
    print(f"\n=== WINNER: [{result.winner_index}] from {winner_short} ===")
    print(result.winner.text)
    print(f"\nReasoning: {result.reasoning}")
    print(f"Slate quality: {result.slate_quality}/10")

    draft_cost = sum(d.cost_usd for d in drafts)
    total = draft_cost + result.cost_usd
    print(
        f"\nCost: drafts ${draft_cost:.5f} + judge ${result.cost_usd:.5f} "
        f"= ${total:.5f} | judge latency {result.latency_ms}ms"
    )


if __name__ == "__main__":
    main()
