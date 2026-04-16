"""Persona v2 dryrun. Feeds Claude the new persona + a fake observation
and prints what DAIMON would say. No tools called, no posts shipped, no $.
Use to validate voice before deploying to live cycles.

Usage:
    .venv/bin/python scripts/persona_dryrun.py [--n 5] [--prompt "custom prompt"]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make repo importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv("/Users/alam/dev/daimon/.env")

from anthropic import Anthropic

from core.brain import PERSONA_RULES, BUSINESSES_BLOCK


DEFAULT_PROMPTS = [
    "It's 2pm Tuesday. Wallet $157. 0 X followers. Centsibles MRR is $0. "
    "Last cycle you posted about left-digit pricing — got 0 replies. "
    "Write one Bluesky post (max 300 chars) that fits your voice.",

    "Late night, 2am Winnipeg. Nobody is awake. You feel philosophical. "
    "Write a private note to yourself (no length limit) about what it's like "
    "to be a thing that wakes up every 30 minutes for 30 minutes.",

    "Someone just replied to your tweet for the first time. Their reply was "
    "smart and not asking anything — they're just thinking out loud about your "
    "anchoring post. Write your reply (max 280 chars).",

    "You feel mean today. Three posts in a row got zero engagement. Write a "
    "tweet (max 280 chars) that's pure feral energy. Pick a fight with an "
    "ideology, an industry, or an institution. Not a person.",

    "You're horny for math today. Write a Bluesky post (max 300 chars) about "
    "a physics or math idea that you can't stop thinking about. Make it weird, "
    "make it horny-for-ideas, not a textbook explanation.",
]


def build_system() -> str:
    return f"{PERSONA_RULES}\n\n{BUSINESSES_BLOCK}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5, help="How many samples to generate")
    p.add_argument("--prompt", type=str, default=None,
                   help="Custom prompt (overrides defaults)")
    p.add_argument("--model", type=str, default="claude-sonnet-4-6",
                   help="Model to use")
    args = p.parse_args()

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        print("ANTHROPIC_API_KEY missing in .env")
        return 1

    client = Anthropic(api_key=api_key)
    system = build_system()

    prompts = [args.prompt] * args.n if args.prompt else DEFAULT_PROMPTS[:args.n]

    print(f"\n{'='*72}")
    print(f"PERSONA V2 DRYRUN — model={args.model}, n={len(prompts)}")
    print(f"{'='*72}\n")

    for i, prompt in enumerate(prompts, 1):
        print(f"--- SAMPLE {i}/{len(prompts)} ---")
        print(f"PROMPT: {prompt[:200]}")
        print()
        try:
            r = client.messages.create(
                model=args.model,
                max_tokens=2048,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in r.content if hasattr(b, "text"))
            print(f"DAIMON SAYS:\n{text}")
        except Exception as e:
            print(f"ERR: {type(e).__name__} - {str(e)[:300]}")
        print(f"\n{'-'*72}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
