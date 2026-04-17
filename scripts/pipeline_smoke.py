"""Full pipeline smoke test: draft -> judge -> record -> simulate engagement -> stats.

Exercises core.drafter, core.judge, core.posts end-to-end against the real DB.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.drafter import draft
from core.judge import judge
from core.posts import Posts

SYSTEM = (
    "You are DAIMON, a feral internet-brained AI. Voice: specific, weird, "
    "never corporate. No em-dashes. One tweet."
)

PROMPT = "Write a single tweet about staying up too late thinking about the halting problem."


def main() -> None:
    posts = Posts()

    print(">>> drafter")
    drafts = draft(PROMPT, system=SYSTEM)
    for i, d in enumerate(drafts, 1):
        short = d.model_id.split(".")[-1].split(":")[0]
        print(f"  [{i}] {short}: {d.text[:80]}")

    print("\n>>> judge")
    j = judge(drafts, context=f"System: {SYSTEM}\n\nPrompt: {PROMPT}")
    print(f"  winner=[{j.winner_index}] slate_quality={j.slate_quality}/10")
    print(f"  pick: {j.winner.text}")
    print(f"  reason: {j.reasoning[:140]}")

    print("\n>>> record")
    post_id = posts.record_slate(
        prompt=PROMPT,
        drafts=drafts,
        judge_result=j,
        system_prompt=SYSTEM,
        cycle=None,
    )
    print(f"  post_id = {post_id}")

    # Simulate the posting + engagement lifecycle
    posts.mark_posted(post_id, platform="bluesky", external_id=f"test-{post_id}")
    posts.update_engagement(post_id, reply_count=2, like_count=17, repost_count=3, quote_count=1)
    print(f"  marked posted + engagement written")

    row = posts.get(post_id)
    assert row is not None
    print(f"\n>>> readback")
    print(f"  status={row.post_status} platform={row.platform} ext_id={row.external_id}")
    print(f"  engagement_total={row.engagement_total}")
    print(f"  winner_model={row.winner_model.split('.')[-1].split(':')[0]}")

    print("\n>>> stats across all posts so far")
    stats = posts.winner_model_stats()
    for model_id, s in stats.items():
        short = model_id.split(".")[-1].split(":")[0]
        print(f"  {short}: {s['picks']} picks | "
              f"avg_eng={s['avg_engagement']:.1f} | "
              f"avg_slate_q={s['avg_slate_quality']:.1f}")


if __name__ == "__main__":
    main()
