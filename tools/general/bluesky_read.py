"""Read-side of Bluesky: profile stats + recent notifications.

Closes the loop on posts — did anyone reply, like, follow, repost?"""
from __future__ import annotations

import os
from typing import Any

import httpx

from permissions.levels import PermissionLevel
from tools.base import BaseTool


BSKY_HOST = "https://bsky.social"


def _session() -> tuple[str, str] | dict[str, Any]:
    handle = os.getenv("BLUESKY_HANDLE", "").strip()
    password = os.getenv("BLUESKY_APP_PASSWORD", "").strip()
    if not handle or not password:
        return {"ok": False, "summary": "no BLUESKY_HANDLE / BLUESKY_APP_PASSWORD"}
    r = httpx.post(
        f"{BSKY_HOST}/xrpc/com.atproto.server.createSession",
        json={"identifier": handle, "password": password},
        timeout=15.0,
    )
    r.raise_for_status()
    data = r.json()
    return (data["accessJwt"], data["did"])


class BlueskyRead(BaseTool):
    name = "bluesky_read"
    description = (
        "Read your Bluesky state: profile stats (follower count, posts count) "
        "and recent notifications (replies, likes, follows, reposts on your "
        "posts). Use this to check if anyone is engaging with your voice. "
        "If nobody's responding, your posts may be boring or your account too new."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.0

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max notifications to fetch (default 20).",
                    "default": 20,
                },
            },
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        handle = os.getenv("BLUESKY_HANDLE", "").strip()
        limit = int(kwargs.get("limit", 20))
        try:
            sess = _session()
            if isinstance(sess, dict):
                return sess
            jwt, did = sess
            auth = {"Authorization": f"Bearer {jwt}"}

            prof = httpx.get(
                f"{BSKY_HOST}/xrpc/app.bsky.actor.getProfile",
                headers=auth, params={"actor": did}, timeout=15.0,
            )
            prof.raise_for_status()
            p = prof.json()

            notif = httpx.get(
                f"{BSKY_HOST}/xrpc/app.bsky.notification.listNotifications",
                headers=auth, params={"limit": limit}, timeout=15.0,
            )
            notif.raise_for_status()
            n = notif.json()

            items: list[dict[str, Any]] = []
            for entry in n.get("notifications", []):
                author = entry.get("author") or {}
                rec = entry.get("record") or {}
                items.append({
                    "kind": entry.get("reason"),  # like, reply, follow, repost, mention, quote
                    "from_handle": author.get("handle"),
                    "from_display": author.get("displayName"),
                    "text": rec.get("text", "")[:280] if isinstance(rec, dict) else "",
                    "indexed_at": entry.get("indexedAt"),
                    "is_read": entry.get("isRead", False),
                })

            return {
                "ok": True,
                "summary": (
                    f"@{handle}: {p.get('followersCount', 0)} followers, "
                    f"{p.get('followsCount', 0)} following, "
                    f"{p.get('postsCount', 0)} posts. "
                    f"{len(items)} recent notification(s)."
                ),
                "profile": {
                    "handle": handle,
                    "display_name": p.get("displayName"),
                    "description": p.get("description"),
                    "followers": p.get("followersCount", 0),
                    "follows": p.get("followsCount", 0),
                    "posts": p.get("postsCount", 0),
                },
                "notifications": items,
            }
        except httpx.HTTPStatusError as e:
            return {
                "ok": False,
                "summary": f"bluesky http {e.response.status_code}: "
                           f"{e.response.text[:200]}",
            }
        except httpx.HTTPError as e:
            return {"ok": False, "summary": f"bluesky error: {e}"}
