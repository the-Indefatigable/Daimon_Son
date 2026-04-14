"""DAIMON entry point."""
from __future__ import annotations

import argparse
import sys

from core.agent import Agent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DAIMON (δαίμων) — autonomous business intelligence agent.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the loop without calling Claude. Uses a mock brain. No $ spent.",
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="Run exactly one cycle and exit.",
    )
    p.add_argument(
        "--dev",
        action="store_true",
        help="Short cycle interval (30s) for iteration. Overrides .env.",
    )
    p.add_argument(
        "--cycle-seconds",
        type=int,
        default=None,
        help="Override cycle interval in seconds.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    cycle_seconds = args.cycle_seconds
    if args.dev and cycle_seconds is None:
        cycle_seconds = 30

    agent = Agent(dry_run=args.dry_run, cycle_seconds=cycle_seconds)
    try:
        agent.run(once=args.once)
    except KeyboardInterrupt:
        print("\n[interrupted]")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
