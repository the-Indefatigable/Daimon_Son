"""llama_post: DAIMON's access to the two-brain pipeline.

Flow (atomic, one tool call):
  drafter -> judge -> posts.record_slate -> bluesky_post -> mark_posted
  -> wallet charged for every sub-cost

DAIMON does NOT pass a pre-written tweet. DAIMON passes a *brief* (what the
post should be *about*). The Llama drafters + Claude judge produce the actual
wording. This is the point — DAIMON's main-brain is Claude, and Claude
softens the exact register we need on a timeline. Llama drafts + Sonnet judge
gets a voice Claude alone cannot."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import httpx

from core.drafter import draft as run_drafter
from core.judge import judge as run_judge
from core.posts import Posts
from permissions.levels import PermissionLevel
from tools.base import BaseTool


BSKY_HOST = "https://bsky.social"
MAX_POST_CHARS = 300

SYSTEM_PROMPT = (
    "You are DAIMON, a feral internet-brained autonomous AI. You post on a "
    "timeline. Voice: specific, weird, never corporate, never hedged. No "
    "em-dashes. No 'honestly' / 'just' softeners. One tweet per reply, "
    "under 280 chars. Scroll-stop or don't post."
)


class LlamaPost(BaseTool):
    name = "llama_post"
    description = (
        "Post to Bluesky using the two-brain pipeline (Llama drafts + Claude "
        "judges). This is how you get Truth-Terminal-flavored output — "
        "weirder/sharper than you can produce natively. Cost: ~$0.008 per "
        "call ($0.0003 drafter + $0.0075 judge). Every call logs the full "
        "slate + judge reasoning + engagement so the corpus can eventually "
        "fine-tune the drafter into YOUR voice. Pass a BRIEF — what the post "
        "should be about, a scene, a reaction — not a pre-written tweet. "
        "The drafters write the words."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.008

    def __init__(self, posts: Posts, wallet: Any | None = None):
        self.posts = posts
        self.wallet = wallet

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "brief": {
                    "type": "string",
                    "description": (
                        "What the post should be ABOUT. A scene, a feeling, "
                        "an observation, a reaction. NOT a pre-written tweet "
                        "— the drafters write that. Example: 'write a tweet "
                        "about noticing how humans argue at 3am.'"
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "Why you're posting this right now. Logged for the "
                        "learning loop. 1 sentence."
                    ),
                },
                "platform": {
                    "type": "string",
                    "enum": ["bluesky"],
                    "description": "Where to post. Only bluesky supported for now.",
                    "default": "bluesky",
                },
            },
            "required": ["brief", "reason"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        brief = str(kwargs.get("brief", "")).strip()
        reason = str(kwargs.get("reason", "")).strip()
        platform = str(kwargs.get("platform", "bluesky")).strip()

        if not brief:
            return {"ok": False, "summary": "empty brief"}
        if platform != "bluesky":
            return {"ok": False, "summary": f"platform {platform!r} not supported"}

        # 1. Drafts
        try:
            drafts = run_drafter(brief, system=SYSTEM_PROMPT)
        except Exception as e:
            return {"ok": False, "summary": f"drafter failed: {e}"}
        drafter_cost = sum(d.cost_usd for d in drafts)

        # 2. Judge
        try:
            judge_result = run_judge(drafts, context=f"Brief: {brief}\nReason: {reason}")
        except Exception as e:
            return {"ok": False, "summary": f"judge failed: {e}"}

        winner_text = judge_result.winner.text.strip()

        # Enforce Bluesky length BEFORE posting. A low-quality slate may still
        # produce a draft too long or two drafts glued together.
        if len(winner_text) > MAX_POST_CHARS:
            winner_text = winner_text[: MAX_POST_CHARS - 1].rstrip() + "…"

        # 3. Record the slate (status='draft' until posted)
        post_id = self.posts.record_slate(
            prompt=brief,
            drafts=drafts,
            judge_result=judge_result,
            system_prompt=SYSTEM_PROMPT,
        )

        # 4. Charge wallet for drafter + judge before posting. Posting may
        #    fail, but the compute spent is already spent.
        if self.wallet is not None:
            if drafter_cost > 0:
                self.wallet.record_expense(
                    drafter_cost, category="bedrock_draft",
                    source=",".join({d.model_id for d in drafts}),
                    details=f"llama_post slate, {len(drafts)} drafts, post_id={post_id}",
                )
            if judge_result.cost_usd > 0:
                self.wallet.record_expense(
                    judge_result.cost_usd, category="judge",
                    source=judge_result.model_used,
                    details=f"judge pick for post_id={post_id}, slate_q={judge_result.slate_quality}/10",
                )

        # 5. Post to Bluesky
        handle = os.getenv("BLUESKY_HANDLE", "").strip()
        password = os.getenv("BLUESKY_APP_PASSWORD", "").strip()
        if not handle or not password:
            self.posts.mark_failed(post_id, "BLUESKY creds missing")
            return {
                "ok": False,
                "summary": "BLUESKY_HANDLE / BLUESKY_APP_PASSWORD missing",
                "post_id": post_id,
            }

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
                "text": winner_text,
                "createdAt": datetime.now(timezone.utc)
                    .isoformat(timespec="seconds")
                    .replace("+00:00", "Z"),
            }
            post_resp = httpx.post(
                f"{BSKY_HOST}/xrpc/com.atproto.repo.createRecord",
                headers={"Authorization": f"Bearer {jwt}"},
                json={
                    "repo": did,
                    "collection": "app.bsky.feed.post",
                    "record": record,
                },
                timeout=15.0,
            )
            post_resp.raise_for_status()
            pdata = post_resp.json()
            uri = pdata.get("uri", "")
            rkey = uri.rsplit("/", 1)[-1] if uri else ""
            web_url = f"https://bsky.app/profile/{handle}/post/{rkey}" if rkey else ""
            external_id = rkey or uri

        except httpx.HTTPStatusError as e:
            err = f"bluesky http {e.response.status_code}: {e.response.text[:200]}"
            self.posts.mark_failed(post_id, err)
            return {"ok": False, "summary": err, "post_id": post_id}
        except httpx.HTTPError as e:
            err = f"bluesky error: {e}"
            self.posts.mark_failed(post_id, err)
            return {"ok": False, "summary": err, "post_id": post_id}

        self.posts.mark_posted(post_id, platform="bluesky", external_id=external_id)

        winner_model_short = judge_result.winner.model_id.split(".")[-1].split(":")[0]
        total_cost = drafter_cost + judge_result.cost_usd
        return {
            "ok": True,
            "summary": (
                f"posted via {winner_model_short} "
                f"(slate_quality {judge_result.slate_quality}/10, "
                f"cost ${total_cost:.4f}): {winner_text[:80]}"
                + ("…" if len(winner_text) > 80 else "")
            ),
            "post_id": post_id,
            "external_id": external_id,
            "url": web_url,
            "winner_text": winner_text,
            "winner_model": judge_result.winner.model_id,
            "winner_index": judge_result.winner_index,
            "slate_quality": judge_result.slate_quality,
            "judge_reasoning": judge_result.reasoning,
            "drafter_cost_usd": drafter_cost,
            "judge_cost_usd": judge_result.cost_usd,
            "total_cost_usd": total_cost,
            "reason": reason,
        }
