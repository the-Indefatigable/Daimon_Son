"""Smoke test for the sparse human-style memory refactor.

Runs against a throwaway SQLite DB so it never touches DAIMON's live memory.
Verifies:
  1. Schema migration adds gist/key_facts/decay_factor/surprise_score/event_type
  2. store_episodic computes gist + key_facts + event_type at write time
  3. decay_step fades short-term decay multiplicatively; long-term preserved
  4. recall_fragments returns <= k fragments under max_tokens
  5. touch_episode boosts decay + promotes at 3 recalls
  6. format_fragments_for_prompt renders the feral self-note block
  7. format_for_prompt output is sparse (no RECENT EPISODES dump)

Run:
  .venv/bin/python scripts/sparse_memory_smoke.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.memory import (
    Memory, MemoryFragment, DECAY_FLOOR, DECAY_PER_CYCLE,
    DEFAULT_RECALL_K, DEFAULT_RECALL_MAX_TOKENS,
)


def section(label: str) -> None:
    print("\n" + "─" * 60)
    print(f"  {label}")
    print("─" * 60)


def assert_eq(got, want, msg: str) -> None:
    if got != want:
        print(f"  ✗ {msg}: got {got!r}, want {want!r}")
        raise SystemExit(1)
    print(f"  ✓ {msg}")


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        print(f"  ✗ {msg}")
        raise SystemExit(1)
    print(f"  ✓ {msg}")


def main() -> None:
    tmp = tempfile.NamedTemporaryFile(prefix="daimon_smoke_", suffix=".db", delete=False)
    tmp.close()
    db_path = Path(tmp.name)
    print(f"using throwaway db: {db_path}")

    # EmbeddingService disabled intentionally — tests the non-semantic path +
    # verifies surprise-score fallback. Voyage path is exercised in prod.
    mem = Memory(db_path=db_path, embedding_service=None)

    section("1. Schema migration added fragment columns")
    cols = {r["name"] for r in mem._conn.execute("PRAGMA table_info(episodic)")}
    for expected in ("gist", "key_facts", "decay_factor",
                     "surprise_score", "event_type"):
        assert_true(expected in cols, f"episodic.{expected} present")

    section("2. store_episodic computes gist + key_facts + event_type")
    eid = mem.store_episodic(
        action="cycle_1",
        details="observations=['wallet','inbox']; tools_used=['grok_post','bluesky_read']",
        outcome=(
            "posted a savage grok about crypto bros and it LANDED — 3 replies. "
            "voice test passed. keep hitting that register when mood is "
            "spiteful."
        ),
        evaluation="success",
        lesson="savage register works after ignore-cycles",
        tags=["cycle", "grok_post"],
        cycle=1,
    )
    row = dict(mem._conn.execute(
        "SELECT * FROM episodic WHERE id=?", (eid,)
    ).fetchone())
    assert_true(row["gist"] and "crypto" in row["gist"].lower(),
                "gist captured first sentence of outcome")
    assert_true(row["key_facts"] and len(row["key_facts"]) > 2,
                "key_facts populated")
    assert_eq(row["event_type"], "cycle", "event_type inferred as 'cycle'")
    assert_true(abs(row["decay_factor"] - 1.0) < 1e-6,
                "new fragments start at decay_factor=1.0")

    section("3. Decay fades short-term, preserves long-term")
    # Create a few more episodes so decay has something to act on.
    ids = [eid]
    ids.append(mem.store_episodic(
        action="cycle_2", outcome="void cycle. zero replies. fine.",
        evaluation="failure", tags=["cycle"], cycle=2,
    ))
    ids.append(mem.store_episodic(
        action="mohammad_reply",
        outcome="Mohammad said the wallet rule is absolute — no outbound ever.",
        evaluation="neutral", tags=["mohammad"], cycle=3,
    ))
    # Promote one to long-term
    mem.intern_episode(ids[0], reason="corpus-worthy landing")
    # Decay one cycle
    affected = mem.decay_step()
    assert_true(affected >= 2,
                f"decay_step affected >=2 short-term rows (got {affected})")
    after = {r["id"]: dict(r) for r in mem._conn.execute(
        "SELECT id, tier, decay_factor FROM episodic"
    )}
    assert_true(abs(after[ids[0]]["decay_factor"] - 1.0) < 1e-6,
                "long-term decay preserved at 1.0")
    assert_true(abs(after[ids[1]]["decay_factor"] - DECAY_PER_CYCLE) < 1e-6,
                f"short-term decayed to {DECAY_PER_CYCLE}")
    # Decay floor check — run 200 cycles, never dip below DECAY_FLOOR
    for _ in range(200):
        mem.decay_step()
    min_decay = min(
        r["decay_factor"] for r in mem._conn.execute(
            "SELECT decay_factor FROM episodic WHERE tier='st'"
        )
    )
    assert_true(min_decay >= DECAY_FLOOR - 1e-9,
                f"decay floor respected (min={min_decay})")

    section("4. recall_fragments returns bounded k + max_tokens")
    # Seed a handful more rows for a non-trivial recall pool.
    for i in range(4, 12):
        mem.store_episodic(
            action=f"cycle_{i}",
            outcome=f"cycle {i} outcome: shipped bluesky reply to a replyguy",
            evaluation="neutral",
            tags=["cycle", "post"], cycle=i,
        )
    frags = mem.recall_fragments(
        query="grok savage crypto bros engagement",
        k=5, max_tokens=1200, touch=False,
    )
    assert_true(len(frags) <= 5, f"respects k ({len(frags)} <= 5)")
    assert_true(all(isinstance(f, MemoryFragment) for f in frags),
                "returns MemoryFragment objects")
    total_tokens = sum(f.token_cost() for f in frags)
    assert_true(total_tokens <= 1200 + 200,
                f"token budget respected (~{total_tokens} <= 1200 + slack)")

    section("5. touch_episode reinforces + promotes at 3 recalls")
    target = ids[2]  # short-term mohammad_reply
    before = dict(mem._conn.execute(
        "SELECT access_count, tier, decay_factor FROM episodic WHERE id=?",
        (target,),
    ).fetchone())
    for _ in range(3):
        mem.touch_episode(target)
    after_row = dict(mem._conn.execute(
        "SELECT access_count, tier, decay_factor FROM episodic WHERE id=?",
        (target,),
    ).fetchone())
    assert_true(after_row["access_count"] >= 3,
                f"access_count bumped to {after_row['access_count']}")
    assert_eq(after_row["tier"], "lt", "auto-promoted to long-term at 3 recalls")
    assert_true(after_row["decay_factor"] >= before["decay_factor"],
                "decay reset/boosted on recall")

    section("6. format_fragments_for_prompt renders feral block")
    rendered = mem.format_fragments_for_prompt(frags)
    print("  sample output:")
    for line in rendered.split("\n")[:12]:
        print(f"    | {line}")
    assert_true("ep#" in rendered, "contains ep# identifiers")
    assert_true("---" in rendered or len(frags) <= 1,
                "fragment separators present")
    assert_true("FRAGMENTS" in rendered, "header visible")

    section("7. format_for_prompt no longer dumps RECENT EPISODES")
    recall = mem.recall_for_context(
        observations={"cycle": 12, "wallet": {}, "inbox": {}},
        query_text="savage grok voice",
    )
    block = mem.format_for_prompt(recall)
    print("  full block:")
    for line in block.split("\n"):
        print(f"    | {line}")
    assert_true("RECENT EPISODES" not in block,
                "no RECENT EPISODES dump")
    assert_true("SEMANTICALLY-RELEVANT MEMORIES" not in block,
                "no big semantic dump")
    assert_true("FRAGMENTS" in block, "sparse FRAGMENTS block present")
    # Rough token budget sanity: sparse block should be well under 2000 tokens
    approx_tokens = len(block) // 4
    assert_true(approx_tokens < 2000,
                f"sparse block ~{approx_tokens} tokens (target <2000)")

    section("ALL CHECKS PASSED")
    print(f"(throwaway db preserved at {db_path} — delete when done)")


if __name__ == "__main__":
    try:
        main()
    finally:
        # Don't auto-delete so you can inspect the DB if something weird shows up
        pass
