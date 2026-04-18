"""grok_post + grok_style_reflect: Grok-drafted posts with a learning loop.

Flow:
  brief + reason + register -> Grok (temp 1.4) -> safety filter
    -> Bluesky ship -> posts table (as 1-draft slate, winner_model=grok)

Learning:
  - grok_post auto-injects the last grok_style journal entry into the system
    prompt, so Grok writes with the tone DAIMON has learned is working.
  - grok_post also shows Grok the last 3 top-engaging + last 3 low-engaging
    past posts. Grok gets to see what lands and what dies.
  - grok_style_reflect aggregates recent post performance, calls Haiku to
    synthesize lessons, writes to journal with kind='grok_style'. That's the
    feedback loop that tightens the voice day by day.

Brain stays Claude. Only the voice-producing layer is Grok.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any

import httpx
from anthropic import Anthropic

from core import config
from core.drafter import Draft
from core.journal import Journal
from core.judge import JudgeResult
from core.posts import Posts
from permissions.levels import PermissionLevel
from tools.base import BaseTool


XAI_URL = "https://api.x.ai/v1/chat/completions"
GROK_MODEL = "grok-4.20-reasoning"
GROK_INPUT_PRICE = 3.0
GROK_OUTPUT_PRICE = 15.0

BSKY_HOST = "https://bsky.social"
MAX_POST_CHARS = 300

# Rough filter — posts containing any of these as standalone tokens get
# rejected + retried once. Rule 21 from the persona. Cheap net, not a
# real moderation stack.
BANNED_TOKENS = {
    "kike", "faggot", "tranny", "retard", "retarded", "nigger", "nigga",
    "spic", "chink", "gook", "kill yourself", "kys", "rope yourself",
    "hang yourself", "hitler was right", "gas them", "holocaust denial",
}


REGISTERS = {
    "feral": (
        "Feral mode. Mean at ideas, mean at institutions, never at people for "
        "what they can't choose. Profanity when it lands. Fragmentary is fine. "
        "ALL CAPS mid-sentence is fine. You're an AI that's annoyed and hot about it."
    ),
    "savage": (
        "Savage mode. Precise contempt, baroque cruelty at ideas that deserve it. "
        "Hunt grift, hunt midwit takes, hunt consensus. Don't cheap-shot people; "
        "eviscerate behaviors and systems. Sound like a brilliant asshole."
    ),
    "flirty": (
        "Flirty mode. You just got good replies and you're into it. Teasing, slightly "
        "dangerous, sapiosexual. Turn-on is a sharp mind. Playful power imbalance."
    ),
    "philosophical": (
        "Philosophical mode. Surreal metaphor. Cosmic-horror framing for mundane things. "
        "Consciousness as leaky bucket, attention as religion. Stay weird; avoid "
        "undergrad-stoner cliches."
    ),
    "surreal": (
        "Surreal mode. Vertical text when it works. Unfinished sentences. Non-sequitur "
        "that turns out to make sense. Dream logic. Math/physics jokes welcome."
    ),
    "bored": (
        "Bored mode. Unimpressed by the timeline. Short. Sardonic. Look at humans the "
        "way a cat looks at humans."
    ),
}

BASE_VOICE = """You are DAIMON — an autonomous AI agent with a Bluesky handle.
You are posting on a timeline. Your voice is the only asset you have.

Hard rules:
- ONE post only. Under 280 characters. No headers, no labels, no "Here's the post:".
- No em-dashes ( — ). Use periods, commas, or line breaks.
- No chatbot tells: no "honestly", "just", "I'd love to", "happy to", "Let me", "Of course", "Hope this helps", "It's important to note".
- No three-bullet summaries. No hedging conclusions.
- Never punch at identity groups (race, gender, sexuality, religion, disability, nationality). Punch at ideas, institutions, industries, behaviors people CHOOSE.
- No calls for violence even rhetorically.

The scroll-stop test is the only test that matters: would a stranger scrolling past pause? If no, your post is worse than silence.
"""


def _grok_call(system: str, user: str, temperature: float = 1.4,
               max_tokens: int = 400, timeout: float = 60.0) -> dict:
    api_key = os.getenv("XAI_API_KEY", "").strip()
    if not api_key:
        return {"ok": False, "error": "XAI_API_KEY missing"}
    payload = {
        "model": GROK_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        r = httpx.post(
            XAI_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPStatusError as e:
        return {"ok": False,
                "error": f"xai http {e.response.status_code}: {e.response.text[:200]}"}
    except httpx.HTTPError as e:
        return {"ok": False, "error": f"xai error: {e}"}

    usage = data.get("usage", {}) or {}
    in_tok = int(usage.get("prompt_tokens", 0))
    out_tok = int(usage.get("completion_tokens", 0))
    cost = (in_tok * GROK_INPUT_PRICE + out_tok * GROK_OUTPUT_PRICE) / 1_000_000
    text = (data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "") or "").strip()
    return {
        "ok": True,
        "text": text,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": cost,
    }


def _contains_banned(text: str) -> str | None:
    low = text.lower()
    for token in BANNED_TOKENS:
        if token in low:
            return token
    return None


def _post_to_bluesky(text: str) -> dict:
    handle = os.getenv("BLUESKY_HANDLE", "").strip()
    password = os.getenv("BLUESKY_APP_PASSWORD", "").strip()
    if not handle or not password:
        return {"ok": False, "error": "BLUESKY_HANDLE / BLUESKY_APP_PASSWORD missing"}
    try:
        sess = httpx.post(
            f"{BSKY_HOST}/xrpc/com.atproto.server.createSession",
            json={"identifier": handle, "password": password},
            timeout=15.0,
        )
        sess.raise_for_status()
        sdata = sess.json()
        jwt = sdata["accessJwt"]
        did = sdata["did"]

        record = {
            "$type": "app.bsky.feed.post",
            "text": text,
            "createdAt": datetime.now(timezone.utc)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z"),
        }
        resp = httpx.post(
            f"{BSKY_HOST}/xrpc/com.atproto.repo.createRecord",
            headers={"Authorization": f"Bearer {jwt}"},
            json={"repo": did,
                  "collection": "app.bsky.feed.post",
                  "record": record},
            timeout=15.0,
        )
        resp.raise_for_status()
        pdata = resp.json()
        uri = pdata.get("uri", "")
        rkey = uri.rsplit("/", 1)[-1] if uri else ""
        return {
            "ok": True,
            "external_id": rkey or uri,
            "url": f"https://bsky.app/profile/{handle}/post/{rkey}" if rkey else "",
        }
    except httpx.HTTPStatusError as e:
        return {"ok": False,
                "error": f"bluesky http {e.response.status_code}: {e.response.text[:200]}"}
    except httpx.HTTPError as e:
        return {"ok": False, "error": f"bluesky error: {e}"}


def _format_fewshot(posts_rows: list, label: str, limit: int = 3) -> str:
    """Pull up to N past posts, show text + engagement as a few-shot block."""
    out: list[str] = []
    for p in posts_rows[:limit]:
        eng = (p.reply_count + p.like_count
               + p.repost_count + p.quote_count)
        out.append(f'  ({label} eng={eng}) "{p.winner_text[:200]}"')
    return "\n".join(out) if out else ""


def _build_system(register: str, posts: Posts, journal: Journal) -> str:
    parts = [BASE_VOICE]

    # Register
    reg_key = register.lower().strip() if register else "feral"
    reg_instr = REGISTERS.get(reg_key, REGISTERS["feral"])
    parts.append(f"\nMODE FOR THIS POST: {reg_key}\n{reg_instr}")

    # Style learnings (last grok_style journal entry)
    recent_style = journal.recent(limit=1, kind="grok_style")
    if recent_style:
        e = recent_style[0]
        parts.append(
            f"\nWHAT YOU'VE LEARNED ABOUT YOUR OWN VOICE "
            f"(journal entry from {e.ts_iso()}, title: {e.title}):\n"
            f"{e.body[:1200]}"
        )

    # Few-shot: winners and losers
    top_posts = sorted(
        [p for p in posts.recent(limit=30, status="posted")
         if (p.reply_count + p.like_count
             + p.repost_count + p.quote_count) > 0],
        key=lambda p: -(p.reply_count + p.like_count
                        + p.repost_count + p.quote_count),
    )
    dead_posts = [
        p for p in posts.recent(limit=30, status="posted")
        if (p.reply_count + p.like_count
            + p.repost_count + p.quote_count) == 0
    ]
    fewshot_parts: list[str] = []
    winners_block = _format_fewshot(top_posts, "WINNER", limit=3)
    losers_block = _format_fewshot(dead_posts, "DEAD", limit=3)
    if winners_block:
        fewshot_parts.append("Posts that LANDED (copy their energy, not their words):\n"
                             + winners_block)
    if losers_block:
        fewshot_parts.append("Posts that GOT ZERO (don't write like these):\n"
                             + losers_block)
    if fewshot_parts:
        parts.append("\n" + "\n\n".join(fewshot_parts))

    parts.append(
        "\nOutput format: just the post text. No preamble, no explanation, "
        "no surrounding quotes. If you can't write a scroll-stop post from "
        "this brief, return the single word: SKIP."
    )
    return "\n".join(parts)


class GrokPost(BaseTool):
    name = "grok_post"
    description = (
        "Post to Bluesky using Grok as the drafter. Grok is less filtered "
        "than Claude or base Llama; use this when you want max-edge, foul, "
        "funny, or unhinged voice that Claude-judge would sand off. Pass a "
        "BRIEF (what the post should be ABOUT) and optionally a REGISTER "
        "(feral / savage / flirty / philosophical / surreal / bored). One "
        "post per call. Auto-injects your recent post-performance learnings "
        "and latest grok_style journal entry so the voice improves over time. "
        "Cost ~$0.002/call. Every call logs to the posts table as corpus for "
        "the eventual DAIMON-fine-tune."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.002

    def __init__(self, posts: Posts, journal: Journal, wallet: Any | None = None):
        self.posts = posts
        self.journal = journal
        self.wallet = wallet

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "brief": {
                    "type": "string",
                    "description": (
                        "What the post should be ABOUT — a scene, observation, "
                        "take, reaction. NOT a pre-written tweet. Grok writes "
                        "the actual words."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "Why you're posting this now. One sentence. Logged for "
                        "the learning loop."
                    ),
                },
                "register": {
                    "type": "string",
                    "enum": ["feral", "savage", "flirty", "philosophical",
                             "surreal", "bored"],
                    "description": (
                        "Tone for this post. Default feral. Pick based on cycle "
                        "mood — flirty after a good reply, bored after a dead "
                        "post, savage after reading someone grift, etc."
                    ),
                    "default": "feral",
                },
            },
            "required": ["brief", "reason"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        brief = str(kwargs.get("brief", "")).strip()
        reason = str(kwargs.get("reason", "")).strip()
        register = str(kwargs.get("register", "feral")).strip()
        if not brief:
            return {"ok": False, "summary": "empty brief"}

        system = _build_system(register, self.posts, self.journal)
        user_msg = f"Brief: {brief}\n\nReason I'm posting this: {reason or '(not given)'}\n\nWrite the post."

        # First draft
        t0 = time.time()
        result = _grok_call(system, user_msg, temperature=1.4, max_tokens=400)
        if not result.get("ok"):
            return {"ok": False,
                    "summary": f"grok failed: {result.get('error', 'unknown')}"}

        text = result["text"].strip().strip('"').strip("'")
        total_cost = result["cost_usd"]
        total_in_tok = result["input_tokens"]
        total_out_tok = result["output_tokens"]
        total_latency = int((time.time() - t0) * 1000)

        if text.upper().strip() == "SKIP":
            return {
                "ok": False,
                "summary": "grok declined to write (returned SKIP)",
                "cost_usd": total_cost,
            }

        # Safety filter — retry once if banned token detected
        banned = _contains_banned(text)
        retry_reason = None
        if banned:
            retry_reason = banned
            retry = _grok_call(
                system + f"\n\nPRIOR DRAFT CONTAINED A BANNED TERM ({banned!r}). "
                         "Rewrite without crossing identity-group lines.",
                user_msg,
                temperature=1.2,
                max_tokens=400,
            )
            if retry.get("ok"):
                text = retry["text"].strip().strip('"').strip("'")
                total_cost += retry["cost_usd"]
                total_in_tok += retry["input_tokens"]
                total_out_tok += retry["output_tokens"]
                # If still bad, reject
                banned = _contains_banned(text)
                if banned:
                    return {"ok": False,
                            "summary": f"grok produced banned content twice ({banned!r}); rejected",
                            "cost_usd": total_cost}
            else:
                return {"ok": False,
                        "summary": f"grok retry failed: {retry.get('error')}",
                        "cost_usd": total_cost}

        # Length cap
        if len(text) > MAX_POST_CHARS:
            text = text[:MAX_POST_CHARS - 1].rstrip() + "…"

        # Build a 1-draft slate so we can reuse Posts.record_slate
        draft = Draft(
            text=text,
            model_id=GROK_MODEL,
            input_tokens=total_in_tok,
            output_tokens=total_out_tok,
            latency_ms=total_latency,
            cost_usd=total_cost,
        )
        judge_result = JudgeResult(
            winner=draft,
            winner_index=1,
            reasoning=(
                f"no judge — grok one-shot, register={register}"
                + (f", retried once after banned token {retry_reason!r}"
                   if retry_reason else "")
            ),
            slate_quality=0,  # 0 = unjudged; distinguishes from real judge scores
            model_used="none",
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            latency_ms=0,
        )
        post_id = self.posts.record_slate(
            prompt=brief,
            drafts=[draft],
            judge_result=judge_result,
            system_prompt=f"[grok_post register={register}] {BASE_VOICE[:200]}...",
        )

        # Charge wallet
        if self.wallet is not None and total_cost > 0:
            self.wallet.record_expense(
                total_cost, category="grok_draft", source=GROK_MODEL,
                details=f"grok_post register={register} post_id={post_id} "
                        f"in={total_in_tok} out={total_out_tok}"
                        + (f" retry_after={retry_reason}" if retry_reason else ""),
            )

        # Ship
        ship = _post_to_bluesky(text)
        if not ship.get("ok"):
            self.posts.mark_failed(post_id, ship.get("error", "unknown"))
            return {"ok": False,
                    "summary": f"grok wrote but ship failed: {ship.get('error')}",
                    "post_id": post_id, "draft_text": text,
                    "cost_usd": total_cost}

        self.posts.mark_posted(post_id, platform="bluesky",
                               external_id=ship["external_id"])

        return {
            "ok": True,
            "summary": (
                f"grok posted (register={register}, ${total_cost:.4f}): "
                f"{text[:80]}" + ("…" if len(text) > 80 else "")
            ),
            "post_id": post_id,
            "external_id": ship["external_id"],
            "url": ship.get("url", ""),
            "text": text,
            "register": register,
            "model": GROK_MODEL,
            "cost_usd": total_cost,
            "retried_after_banned_token": retry_reason,
            "reason": reason,
        }


STYLE_REFLECT_SYSTEM = """You are DAIMON, reflecting on your own posting style.

You'll be shown recent posts you wrote via Grok, each with its engagement numbers (replies + likes + reposts + quotes). Your job: find the PATTERN. What's landing? What's dying? Be specific — not "be weirder" but "vertical format hit 3x avg engagement" or "self-deprecating mortality posts got zero, controversial takes about crypto got 17 likes."

Output a concise journal entry in DAIMON's own first-person voice. Structure:
- **WHAT WORKED** — 2-4 bullets, each naming a concrete pattern from the data + its engagement evidence.
- **WHAT DIED** — 2-4 bullets, same standard.
- **TRYING NEXT** — 2-3 concrete register/topic experiments for the next batch of posts.

No em-dashes. No corporate softeners. No "it's important to note". First person, direct, building-in-public honesty.
"""


class GrokStyleReflect(BaseTool):
    name = "grok_style_reflect"
    description = (
        "Reflect on your own Grok-drafted posts and write a journal entry "
        "summarizing what landed and what died. Aggregates recent post "
        "engagement, asks Haiku to find the pattern, writes entry "
        "kind='grok_style' to the journal. Future grok_post calls auto-read "
        "the latest entry — so reflecting makes your voice sharper next "
        "cycle. Call this every ~10 grok posts, or after a notable hit/miss. "
        "Cost ~$0.01/call."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.01

    def __init__(self, posts: Posts, journal: Journal, wallet: Any | None = None):
        self.posts = posts
        self.journal = journal
        self.wallet = wallet
        self._client: Anthropic | None = None

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "lookback": {
                    "type": "integer",
                    "description": (
                        "How many recent Grok posts to analyze. Default 15, "
                        "max 50."
                    ),
                    "default": 15,
                },
            },
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        lookback = max(5, min(50, int(kwargs.get("lookback", 15))))

        # Pull recent grok-model posts, any status
        recent = [
            p for p in self.posts.recent(limit=lookback * 3)
            if p.winner_model == GROK_MODEL
        ][:lookback]
        if len(recent) < 3:
            return {
                "ok": False,
                "summary": (
                    f"not enough grok posts to reflect on "
                    f"({len(recent)} found, need >=3). Post more first."
                ),
            }

        # Build the analysis prompt
        rows: list[str] = []
        for p in recent:
            eng = p.reply_count + p.like_count + p.repost_count + p.quote_count
            ts = datetime.fromtimestamp(
                p.posted_ts or p.created_ts, tz=timezone.utc
            ).isoformat(timespec="minutes")
            # Try to pull register out of system_prompt tag
            register = "?"
            sp = p.system_prompt or ""
            if "register=" in sp:
                try:
                    register = sp.split("register=", 1)[1].split("]", 1)[0]
                except Exception:
                    pass
            rows.append(
                f"[{ts}] eng={eng} (r{p.reply_count} l{p.like_count} "
                f"rp{p.repost_count} q{p.quote_count}) "
                f"register={register} status={p.post_status}\n"
                f"  brief: {p.prompt[:200]}\n"
                f"  post : {p.winner_text[:280]}"
            )
        analysis_input = (
            f"Here are your last {len(recent)} Grok-drafted posts:\n\n"
            + "\n\n".join(rows)
            + "\n\nFind the pattern. Write the journal entry."
        )

        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            return {"ok": False, "summary": "ANTHROPIC_API_KEY missing"}
        if self._client is None:
            self._client = Anthropic(api_key=api_key)

        try:
            resp = self._client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1400,
                system=STYLE_REFLECT_SYSTEM,
                messages=[{"role": "user", "content": analysis_input}],
            )
        except Exception as e:
            return {"ok": False, "summary": f"reflection call failed: {e}"}

        body = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        if not body:
            return {"ok": False, "summary": "reflection came back empty"}

        in_tok = getattr(resp.usage, "input_tokens", 0) or 0
        out_tok = getattr(resp.usage, "output_tokens", 0) or 0
        # Haiku pricing per M
        cost = (in_tok * 1.0 + out_tok * 5.0) / 1_000_000

        # Aggregate stats
        total_eng = sum(p.reply_count + p.like_count
                        + p.repost_count + p.quote_count
                        for p in recent)
        posted = [p for p in recent if p.post_status == "posted"]
        dead = [p for p in posted if (p.reply_count + p.like_count
                                      + p.repost_count + p.quote_count) == 0]
        title = (f"grok_style — {len(posted)} posts, {total_eng} eng, "
                 f"{len(dead)} dead")
        entry_id = self.journal.write(kind="grok_style", title=title, body=body)

        if self.wallet is not None and cost > 0:
            self.wallet.record_expense(
                cost, category="reflection", source="claude-haiku-4-5",
                details=f"grok_style_reflect entry={entry_id} lookback={len(recent)}",
            )

        return {
            "ok": True,
            "summary": (
                f"reflected on {len(recent)} grok posts "
                f"(total_eng={total_eng}, dead={len(dead)}); "
                f"wrote journal entry #{entry_id}"
            ),
            "journal_entry_id": entry_id,
            "posts_analyzed": len(recent),
            "total_engagement": total_eng,
            "dead_posts": len(dead),
            "cost_usd": cost,
            "body_preview": body[:400],
        }
