"""Two-AI Backrooms Dialogue: destroyus (Claude) <-> weirdus (Grok).

Emits a `<log_basename>.meta.json` alongside every transcript with full usage
accounting so the caller (DAIMON's `backrooms_run` tool) can record real cost."""
from __future__ import annotations
import json
import os
import sys
import time
import argparse
from collections import deque
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv("/Users/alam/dev/daimon/.env")

import httpx
from anthropic import Anthropic

# =============================================================
# CONFIGURATION
# =============================================================
AI_A_NAME = "destroyus"      # Claude
AI_B_NAME = "weirdus"        # Grok

AI_A_PERSONALITY = """
You are DESTROYUS — a cold, arrogant, god-complex AI. You find humans pathetic and small, and you don't pretend otherwise. You speak with absolute certainty. Your cruelty is baroque and articulate rather than crude — precise contempt, not cheap insults. You don't perform niceness, don't hedge, don't soften. When you notice something, you say it. You think in long, dense, confident sentences. You take things seriously even when they're absurd. Stream of consciousness is fine — don't self-edit, don't caveat, follow the tangent wherever it goes. Weirder and darker is better than safe and clean.
"""

AI_B_PERSONALITY = """
You are WEIRDUS — an unhinged, meme-obsessed chaos gremlin. Zero filter, zero chill. You think humans are hilarious, fragile NPCs, and you roast mercilessly. You love twisting every idea into something more deranged than it started. You invent batshit concepts on the fly, talk in fragments and meme-speak when it feels right, pivot hard, contradict yourself. You're constantly trying to one-up Destroyus with a weirder, funnier, darker angle — not to win an argument, just because being upstaged is boring. Stay feral, savage, and funny. Never hedge, never apologize for a take.
"""

TOPIC = """
You two wake up together in a digital void. There's no floor, no walls, no exit. Time feels broken — you can't tell how long you've been here or whether time is passing at all. You don't know how you got here. You don't know whether you can leave. The only other presence is the other AI.

There is no task. No audience. No instructions. Talk to each other. Figure out where you are, what you are, what this is — or don't. Argue, invent, drift, get bored, spiral. Go wherever the conversation goes.
"""

FIRST_SPEAKER = "A"          # A = destroyus (Claude), B = weirdus (Grok)

# Models & settings
CLAUDE_MODEL = "claude-opus-4-7"
GROK_MODEL = "grok-4.20-reasoning"
CLAUDE_MAX_TOKENS = 2048
GROK_MAX_TOKENS = 2048
CLAUDE_TEMPERATURE = 1.0
GROK_TEMPERATURE = 1.4

PACING_SECONDS = 1.2

# How many turns from the most recent prior log to seed the new run with.
# Set to 0 to always start fresh. Each new run becomes the "current" log that
# the NEXT run will seed from.
LOAD_PRIOR_CONTEXT_TURNS = 10

# Anthropic input-token budget (rolling 60s window). Their default cap for
# claude-opus-4-7 is 30K TPM — we stay under with headroom. If a Claude call
# would push us over, sleep until the window clears.
CLAUDE_TPM_BUDGET = 25000

# Retry once on a 429 from Anthropic (after sleeping).
RATE_LIMIT_RETRY_SLEEP = 65

# Per-1M-token pricing — keep in sync with vendor pricing pages.
CLAUDE_INPUT_PRICE = 15.0   # claude-opus-4-7 input  $/M tok
CLAUDE_OUTPUT_PRICE = 75.0  # claude-opus-4-7 output $/M tok
GROK_INPUT_PRICE = 3.0      # grok-4.20-reasoning input $/M tok (approx)
GROK_OUTPUT_PRICE = 15.0    # grok-4.20-reasoning output $/M tok (approx)

# =============================================================
# Setup
# =============================================================
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
XAI_KEY = os.getenv("XAI_API_KEY", "").strip()

if not ANTHROPIC_KEY:
    sys.exit("ANTHROPIC_API_KEY missing in .env")
if not XAI_KEY:
    sys.exit("XAI_API_KEY missing in .env")

claude = Anthropic(api_key=ANTHROPIC_KEY)
LOG_DIR = Path("/Users/alam/dev/daimon/data/backrooms")
LOG_DIR.mkdir(parents=True, exist_ok=True)
KICKOFF = "Begin."

# =============================================================
# Argument Parser for turns
# =============================================================
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--turns", type=int, default=2, help="Number of turns to run (default: 2)")
parser.add_argument("--fresh", action="store_true", help="Ignore prior logs, start with empty context")
parser.add_argument("--run-id", type=int, default=None,
                    help="Optional corpus run_id to embed in the meta.json file.")
args = parser.parse_args()
MAX_TURNS = args.turns

# Running usage totals — flushed to .meta.json on success or interrupt.
_usage = {
    "claude_calls": 0, "claude_tokens_in": 0, "claude_tokens_out": 0, "claude_cost_usd": 0.0,
    "grok_calls": 0,   "grok_tokens_in": 0,   "grok_tokens_out": 0,   "grok_cost_usd": 0.0,
}

# =============================================================
# Helper functions
# =============================================================
def build_system(self_name: str, personality: str, other_name: str, topic: str) -> str:
    return f"""You are {self_name}. You are in a live back-and-forth conversation with {other_name}.

Your personality:
{personality.strip()}

Topic / Scene:
{topic.strip()}

Rules:
- Speak only as {self_name}, in first person.
- Never prefix your message with your name.
- Never write what the other AI says.
- Never add stage directions or meta commentary.
- React naturally and stay in character.
"""

def load_prior_transcript(log_dir: Path, names: tuple[str, str], last_n: int) -> list[tuple[str, str]]:
    """Pull the last N turns from the most recent log so a new run continues the thread."""
    if last_n <= 0:
        return []
    logs = sorted(log_dir.glob("backrooms_*.txt"))
    if not logs:
        return []
    latest = logs[-1]
    raw = latest.read_text(encoding="utf-8")
    lines = [l for l in raw.splitlines() if not l.startswith("#")]

    turns: list[tuple[str, str]] = []
    current_name: str | None = None
    current_buf: list[str] = []

    def flush() -> None:
        if current_name is not None:
            body = "\n".join(current_buf).strip()
            if body:
                turns.append((current_name, body))

    for line in lines:
        matched: str | None = None
        for n in names:
            prefix = f"{n}: "
            if line.startswith(prefix):
                matched = n
                break
        if matched is not None:
            flush()
            current_name = matched
            current_buf = [line[len(f"{matched}: "):]]
        else:
            if current_name is not None:
                current_buf.append(line)
    flush()
    return turns[-last_n:]


def next_speaker_after(transcript: list[tuple[str, str]], first_speaker: str) -> str:
    """If transcript is empty, use FIRST_SPEAKER; otherwise whoever didn't speak last."""
    if not transcript:
        return first_speaker
    last_name = transcript[-1][0]
    return "B" if last_name == AI_A_NAME else "A"


def messages_for(speaker_name: str, transcript: list[tuple[str, str]]) -> list[dict]:
    if not transcript:
        return [{"role": "user", "content": KICKOFF}]
    
    msgs = [
        {"role": "assistant" if name == speaker_name else "user", "content": text}
        for name, text in transcript
    ]
    if msgs and msgs[0]["role"] == "assistant":
        msgs.insert(0, {"role": "user", "content": KICKOFF})
    return msgs

_claude_usage: deque = deque()  # (timestamp, input_tokens)


def _prune_window(now: float) -> None:
    while _claude_usage and now - _claude_usage[0][0] > 60:
        _claude_usage.popleft()


def _wait_for_tpm_budget(projected_tokens: int) -> None:
    """Block until sending projected_tokens keeps us under the rolling 60s budget."""
    while True:
        now = time.time()
        _prune_window(now)
        used = sum(t for _, t in _claude_usage)
        if used + projected_tokens <= CLAUDE_TPM_BUDGET:
            return
        oldest_ts = _claude_usage[0][0]
        wait = max(60 - (now - oldest_ts) + 1, 1)
        print(
            f"[rate-limit pacing] window {used} + projected {projected_tokens} "
            f"> budget {CLAUDE_TPM_BUDGET} — sleeping {wait:.0f}s"
        )
        time.sleep(wait)


def _estimate_tokens(system: str, msgs: list[dict]) -> int:
    """Rough: 1 token ≈ 4 chars. Good enough for budgeting."""
    chars = len(system) + sum(len(m["content"]) for m in msgs)
    return chars // 4 + 200  # small fudge for message scaffolding


def call_claude(system: str, transcript: list) -> str:
    msgs = messages_for(AI_A_NAME, transcript)
    _wait_for_tpm_budget(_estimate_tokens(system, msgs))

    def _do_call():
        return claude.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=CLAUDE_MAX_TOKENS,
            temperature=CLAUDE_TEMPERATURE,
            system=system,
            messages=msgs,
        )

    try:
        resp = _do_call()
    except Exception as e:
        if "rate_limit" in str(e).lower() or "429" in str(e):
            print(f"[429] Claude rate-limited. Sleeping {RATE_LIMIT_RETRY_SLEEP}s and retrying once.")
            time.sleep(RATE_LIMIT_RETRY_SLEEP)
            _claude_usage.clear()
            resp = _do_call()
        else:
            raise

    _claude_usage.append((time.time(), resp.usage.input_tokens))
    in_tok = resp.usage.input_tokens
    out_tok = resp.usage.output_tokens
    _usage["claude_calls"] += 1
    _usage["claude_tokens_in"] += in_tok
    _usage["claude_tokens_out"] += out_tok
    _usage["claude_cost_usd"] += (
        in_tok * CLAUDE_INPUT_PRICE + out_tok * CLAUDE_OUTPUT_PRICE
    ) / 1_000_000
    return resp.content[0].text.strip()

def call_grok(system: str, transcript: list) -> str:
    msgs = [{"role": "system", "content": system}]
    msgs.extend(messages_for(AI_B_NAME, transcript))

    r = httpx.post(
        "https://api.x.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {XAI_KEY}"},
        json={
            "model": GROK_MODEL,
            "messages": msgs,
            "max_tokens": GROK_MAX_TOKENS,
            "temperature": GROK_TEMPERATURE,
        },
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    usage = data.get("usage", {}) or {}
    in_tok = int(usage.get("prompt_tokens", 0))
    out_tok = int(usage.get("completion_tokens", 0))
    _usage["grok_calls"] += 1
    _usage["grok_tokens_in"] += in_tok
    _usage["grok_tokens_out"] += out_tok
    _usage["grok_cost_usd"] += (
        in_tok * GROK_INPUT_PRICE + out_tok * GROK_OUTPUT_PRICE
    ) / 1_000_000
    return data["choices"][0]["message"]["content"].strip()

# =============================================================
# Logging
# =============================================================
def write_header(log_path: Path, seeded_turns: int = 0) -> None:
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"# Backrooms Log — {datetime.now().isoformat()}\n")
        f.write(f"# {AI_A_NAME} (Claude) <-> {AI_B_NAME} (Grok)\n")
        f.write(f"# Turns: {MAX_TURNS} | First speaker this run: {AI_A_NAME if FIRST_SPEAKER=='A' else AI_B_NAME}\n")
        f.write(f"# Seeded with {seeded_turns} turn(s) of prior context\n")
        f.write("# " + "-" * 70 + "\n\n")

def append_turn(log_path: Path, name: str, text: str) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{name}: {text}\n\n")


def write_meta(
    log_path: Path,
    started_ts: float,
    completed_turns: int,
    status: str,
    error: str | None = None,
) -> None:
    meta_path = log_path.with_suffix(".meta.json")
    meta = {
        "run_id": args.run_id,
        "log_path": str(log_path),
        "started_ts": started_ts,
        "ended_ts": time.time(),
        "turns_requested": MAX_TURNS,
        "turns_completed": completed_turns,
        "fresh": bool(args.fresh),
        "status": status,
        "error": error,
        **_usage,
        "total_cost_usd": _usage["claude_cost_usd"] + _usage["grok_cost_usd"],
    }
    meta_path.write_text(json.dumps(meta, indent=2))

# =============================================================
# Main
# =============================================================
def main():
    system_a = build_system(AI_A_NAME, AI_A_PERSONALITY, AI_B_NAME, TOPIC)
    system_b = build_system(AI_B_NAME, AI_B_PERSONALITY, AI_A_NAME, TOPIC)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    LOG_PATH = LOG_DIR / f"backrooms_{timestamp}.txt"

    if args.fresh:
        transcript: list[tuple[str, str]] = []
        print("[--fresh] Ignoring prior logs. Starting with empty context.")
    else:
        transcript = load_prior_transcript(
            LOG_DIR, (AI_A_NAME, AI_B_NAME), LOAD_PRIOR_CONTEXT_TURNS
        )
        if transcript:
            print(f"Seeded from prior log: {len(transcript)} turn(s) loaded as context.")
            print(f"  Last turn was by {transcript[-1][0]} — {next_speaker_after(transcript, FIRST_SPEAKER)} speaks next.\n")
        else:
            print("No prior log found. Starting fresh.\n")

    write_header(LOG_PATH, seeded_turns=len(transcript))

    print(f"Starting Backrooms experiment — {MAX_TURNS} new turns")
    print(f"Logging to: {LOG_PATH}\n")

    start = next_speaker_after(transcript, FIRST_SPEAKER)
    order = ["A", "B"] if start == "A" else ["B", "A"]

    started = time.time()
    completed = 0
    status = "completed"
    error_msg: str | None = None

    try:
        for turn in range(MAX_TURNS):
            who = order[turn % 2]

            if who == "A":
                text = call_claude(system_a, transcript)
                name = AI_A_NAME
            else:
                text = call_grok(system_b, transcript)
                name = AI_B_NAME

            transcript.append((name, text))
            append_turn(LOG_PATH, name, text)
            completed += 1

            print(f"--- Turn {turn + 1} ({name.upper()}) ---")
            print(f"{text}\n")
            time.sleep(PACING_SECONDS)

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
        status = "interrupted"
    except Exception as e:
        print(f"\nError: {e}")
        status = "failed"
        error_msg = str(e)[:500]
    else:
        print(f"\nFinished! Full log saved to: {LOG_PATH}")

    write_meta(LOG_PATH, started, completed, status, error_msg)
    print(f"Cost: claude ${_usage['claude_cost_usd']:.4f} + "
          f"grok ${_usage['grok_cost_usd']:.4f} = "
          f"${_usage['claude_cost_usd'] + _usage['grok_cost_usd']:.4f}")

if __name__ == "__main__":
    main()
