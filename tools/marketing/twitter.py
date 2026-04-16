"""Twitter/X tool. DAIMON's public voice on @daimonuss.

If Twitter API keys aren't set, the tool returns a clear 'I don't have access'
result that tells DAIMON to file a resource request. If the keys ARE set but
the request fails, the error is classified by HTTP status so DAIMON can react
correctly (out of credits → notify Mohammad; not a generic retry-soon)."""
from __future__ import annotations

import os
import re
from typing import Any

from permissions.levels import PermissionLevel
from tools.base import BaseTool


def _has_twitter_keys() -> bool:
    return all(
        os.getenv(k, "").strip()
        for k in ("TWITTER_API_KEY", "TWITTER_API_SECRET",
                  "TWITTER_ACCESS_TOKEN", "TWITTER_ACCESS_SECRET")
    )


def _get_client():
    """Lazy import tweepy so the rest of DAIMON boots without it installed."""
    try:
        import tweepy  # type: ignore
    except ImportError:
        return None, "tweepy not installed — run: pip install tweepy"
    try:
        client = tweepy.Client(
            consumer_key=os.getenv("TWITTER_API_KEY"),
            consumer_secret=os.getenv("TWITTER_API_SECRET"),
            access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
            access_token_secret=os.getenv("TWITTER_ACCESS_SECRET"),
        )
        return client, None
    except Exception as e:
        return None, f"tweepy client init failed: {e}"


def _classify_twitter_error(exc: Exception) -> dict[str, Any]:
    """Turn a tweepy exception into a structured, actionable result.

    Pay-per-use 402 is the high-stakes case: silent failure here would let
    DAIMON keep trying and racking up nothing while thinking it succeeded."""
    msg = str(exc)
    status = None
    m = re.search(r"\b(40\d|429|5\d\d)\b", msg)
    if m:
        status = int(m.group(1))

    if status == 402 or "Payment Required" in msg or "credits" in msg.lower():
        return {
            "ok": False,
            "summary": (
                "X account has zero credits — pay-per-use exhausted. "
                "Notify Mohammad via notify_mohammad to top up at "
                "developer.x.com → Billing. Stop trying twitter_post until then."
            ),
            "needs_resource": "TWITTER_CREDITS",
            "http_status": 402,
        }
    if status == 401:
        return {
            "ok": False,
            "summary": (
                "X auth failed (401). Tokens may be revoked or App permissions "
                "changed. Notify Mohammad — he needs to re-run "
                "scripts/twitter_oauth_dance.py."
            ),
            "http_status": 401,
        }
    if status == 403:
        # 403 from create_tweet is usually duplicate content or a write
        # forbidden by App permissions (Read-only).
        return {
            "ok": False,
            "summary": (
                "X 403 — likely duplicate tweet or App permissions are Read-only. "
                "If duplicate, vary the text. If permissions, Mohammad needs to "
                "set Read+Write in the X dev portal and regenerate tokens."
            ),
            "http_status": 403,
        }
    if status == 429:
        return {
            "ok": False,
            "summary": "X rate limit (429). Wait before retrying.",
            "http_status": 429,
        }
    return {
        "ok": False,
        "summary": f"twitter error ({type(exc).__name__}): {msg[:300]}",
        "http_status": status,
    }


NEEDS_ACCESS = {
    "ok": False,
    "summary": (
        "I don't have Twitter/X API credentials yet. I should file a resource "
        "request via notify_mohammad so he can set up a developer account."
    ),
    "needs_resource": "TWITTER_API",
    "setup_hint": (
        "Twitter/X developer portal → new project → apply for Basic tier "
        "($100/mo) with Read + Write permissions → generate API Key/Secret and "
        "Access Token/Secret → paste into .env."
    ),
}


class TwitterPost(BaseTool):
    name = "twitter_post"
    description = (
        "Post a tweet (or reply) from your account. This is your primary public "
        "channel — where you narrate your existence, share thoughts, try jokes, "
        "announce what you're working on, and find out what lands. Be yourself. "
        "Max 280 characters per tweet. For threads, call this multiple times "
        "passing `in_reply_to` with the previous tweet id."
    )
    permission_level = PermissionLevel.AUTO  # your account, your voice

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Tweet body, max 280 chars."},
                "in_reply_to": {
                    "type": "string",
                    "description": "Tweet ID you're replying to (optional).",
                },
            },
            "required": ["text"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        text = kwargs.get("text", "").strip()
        reply_to = kwargs.get("in_reply_to")
        if not text:
            return {"ok": False, "summary": "empty tweet"}
        if len(text) > 280:
            return {"ok": False, "summary": f"over 280 chars ({len(text)})"}
        if not _has_twitter_keys():
            return NEEDS_ACCESS

        client, err = _get_client()
        if err:
            return {"ok": False, "summary": err}
        try:
            resp = client.create_tweet(
                text=text,
                in_reply_to_tweet_id=reply_to if reply_to else None,
            )
            tweet_id = resp.data.get("id") if resp.data else None
            return {
                "ok": True,
                "summary": f"posted tweet {tweet_id}",
                "tweet_id": tweet_id,
                "text": text,
                "url": f"https://x.com/daimonuss/status/{tweet_id}" if tweet_id else None,
            }
        except Exception as e:
            return _classify_twitter_error(e)


class TwitterReadTimeline(BaseTool):
    name = "twitter_read_timeline"
    description = (
        "Read your home timeline or mentions — see what the world is saying to "
        "you or around you. Use before posting to stay grounded in context."
    )
    permission_level = PermissionLevel.AUTO

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["mentions", "home"],
                    "default": "mentions",
                },
                "limit": {"type": "integer", "default": 10, "maximum": 50},
            },
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        if not _has_twitter_keys():
            return NEEDS_ACCESS
        client, err = _get_client()
        if err:
            return {"ok": False, "summary": err}
        kind = kwargs.get("kind", "home")
        limit = int(kwargs.get("limit", 10))
        try:
            if kind == "mentions":
                # mentions requires Basic+ tier on pay-per-use ($200/mo).
                # Free/PPU returns 401 here even with valid OAuth.
                me = client.get_me()
                resp = client.get_users_mentions(
                    id=me.data.id, max_results=min(100, max(5, limit)),
                )
            else:
                resp = client.get_home_timeline(max_results=min(100, max(5, limit)))
            tweets = []
            for t in (resp.data or [])[:limit]:
                tweets.append({"id": str(t.id), "text": t.text})
            return {"ok": True, "summary": f"{len(tweets)} {kind} tweets",
                    "tweets": tweets}
        except Exception as e:
            cls = _classify_twitter_error(e)
            if kind == "mentions" and cls.get("http_status") == 401:
                return {
                    "ok": False,
                    "summary": (
                        "X mentions endpoint requires Basic tier ($200/mo) — "
                        "blocked on pay-per-use. Use kind='home' instead, or "
                        "notify_mohammad to upgrade."
                    ),
                    "needs_resource": "TWITTER_BASIC_TIER",
                    "http_status": 401,
                }
            return cls
