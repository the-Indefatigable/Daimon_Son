"""Bluesky (AT Protocol) posting tool. Free API, no rate-limit surprises.

Uses app password auth — create one at bsky.app under Settings → App Passwords.
Session is created per call (posting is rare; not worth caching)."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import httpx

from permissions.levels import PermissionLevel
from tools.base import BaseTool


BSKY_HOST = "https://bsky.social"
MAX_POST_CHARS = 300


class BlueskyPost(BaseTool):
    name = "bluesky_post"
    description = (
        "Post to your Bluesky account (@daimonus). Free, public, permanent. "
        "Use this as your public voice: thoughts, reflections, status, "
        "questions into the void, accountability. Truth Terminal style — "
        "honest, first-person, a voice someone could actually follow. "
        "Max 300 chars per post. One post per call. Keep URLs as plain text."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.0

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": f"Post body. Max {MAX_POST_CHARS} chars. "
                                   "Plain text. Emojis are fine.",
                },
            },
            "required": ["text"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        handle = os.getenv("BLUESKY_HANDLE", "").strip()
        password = os.getenv("BLUESKY_APP_PASSWORD", "").strip()
        if not handle or not password:
            return {
                "ok": False,
                "summary": "no BLUESKY_HANDLE or BLUESKY_APP_PASSWORD",
                "needs_resource": "BLUESKY_APP_PASSWORD",
            }

        text = str(kwargs.get("text", "")).strip()
        if not text:
            return {"ok": False, "summary": "empty post"}
        truncated = False
        if len(text) > MAX_POST_CHARS:
            text = text[: MAX_POST_CHARS - 1].rstrip() + "…"
            truncated = True

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
            post = httpx.post(
                f"{BSKY_HOST}/xrpc/com.atproto.repo.createRecord",
                headers={"Authorization": f"Bearer {jwt}"},
                json={
                    "repo": did,
                    "collection": "app.bsky.feed.post",
                    "record": record,
                },
                timeout=15.0,
            )
            post.raise_for_status()
            pdata = post.json()
            uri = pdata.get("uri", "")
            # URI format: at://did:plc:xxx/app.bsky.feed.post/<rkey>
            rkey = uri.rsplit("/", 1)[-1] if uri else ""
            web_url = f"https://bsky.app/profile/{handle}/post/{rkey}" if rkey else ""
            return {
                "ok": True,
                "summary": (
                    f"posted to bluesky as @{handle} ({len(text)} chars"
                    + (", truncated to fit 300 limit" if truncated else "")
                    + ")"
                ),
                "url": web_url,
                "uri": uri,
                "truncated": truncated,
            }
        except httpx.HTTPStatusError as e:
            return {
                "ok": False,
                "summary": f"bluesky http {e.response.status_code}: "
                           f"{e.response.text[:200]}",
            }
        except httpx.HTTPError as e:
            return {"ok": False, "summary": f"bluesky error: {e}"}
