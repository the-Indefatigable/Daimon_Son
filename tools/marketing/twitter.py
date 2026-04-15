"""Twitter/X tool. DAIMON's public voice — the heart of the Truth-Terminal channel.

If Twitter API keys aren't set, the tool returns a clear 'I don't have access'
result that tells DAIMON to file a resource request."""
from __future__ import annotations

import os
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
            }
        except Exception as e:
            return {"ok": False, "summary": f"twitter error: {e}"}


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
        kind = kwargs.get("kind", "mentions")
        limit = int(kwargs.get("limit", 10))
        try:
            me = client.get_me()
            if kind == "mentions":
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
            return {"ok": False, "summary": f"twitter error: {e}"}
