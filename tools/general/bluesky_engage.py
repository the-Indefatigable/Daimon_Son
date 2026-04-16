"""Bluesky reply + search — the engagement side.

Posting alone builds nothing. Reply + search let DAIMON find accounts/topics
and join conversations — the only way a following actually forms."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import httpx

from permissions.levels import PermissionLevel
from tools.base import BaseTool


BSKY_HOST = "https://bsky.social"
MAX_POST_CHARS = 300


def _session() -> tuple[str, str] | None:
    handle = os.getenv("BLUESKY_HANDLE", "").strip()
    password = os.getenv("BLUESKY_APP_PASSWORD", "").strip()
    if not handle or not password:
        return None
    r = httpx.post(
        f"{BSKY_HOST}/xrpc/com.atproto.server.createSession",
        json={"identifier": handle, "password": password},
        timeout=15.0,
    )
    r.raise_for_status()
    d = r.json()
    return (d["accessJwt"], d["did"])


def _resolve_post(jwt: str, at_uri: str) -> dict[str, Any] | None:
    """Resolve an at:// post URI → full record incl. root for threading."""
    r = httpx.get(
        f"{BSKY_HOST}/xrpc/app.bsky.feed.getPosts",
        headers={"Authorization": f"Bearer {jwt}"},
        params={"uris": at_uri},
        timeout=15.0,
    )
    r.raise_for_status()
    posts = r.json().get("posts", [])
    return posts[0] if posts else None


class BlueskyReply(BaseTool):
    name = "bluesky_reply"
    description = (
        "Reply to a Bluesky post. Give it the post's at:// URI (from bluesky_read "
        "notifications or bluesky_search results). Use replies to build real "
        "conversations — not just broadcasts. One reply is often worth 10 posts "
        "into the void. Max 300 chars."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.0

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string",
                         "description": f"Reply body. Max {MAX_POST_CHARS} chars."},
                "reply_to_uri": {"type": "string",
                                 "description": "at:// URI of the post you're replying to."},
            },
            "required": ["text", "reply_to_uri"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        handle = os.getenv("BLUESKY_HANDLE", "").strip()
        text = str(kwargs.get("text", "")).strip()
        parent_uri = str(kwargs.get("reply_to_uri", "")).strip()
        if not text:
            return {"ok": False, "summary": "empty reply"}
        if len(text) > MAX_POST_CHARS:
            return {"ok": False,
                    "summary": f"reply too long: {len(text)} > {MAX_POST_CHARS}"}
        if not parent_uri.startswith("at://"):
            return {"ok": False, "summary": "reply_to_uri must be an at:// URI"}
        try:
            sess = _session()
            if sess is None:
                return {"ok": False, "summary": "no BLUESKY creds"}
            jwt, did = sess
            parent = _resolve_post(jwt, parent_uri)
            if not parent:
                return {"ok": False, "summary": f"could not resolve {parent_uri}"}
            parent_ref = {"uri": parent["uri"], "cid": parent["cid"]}
            # Thread root: walk up if parent is itself a reply
            record = parent.get("record") or {}
            root_ref = record.get("reply", {}).get("root") or parent_ref

            post_record = {
                "$type": "app.bsky.feed.post",
                "text": text,
                "createdAt": datetime.now(timezone.utc)
                    .isoformat(timespec="seconds").replace("+00:00", "Z"),
                "reply": {"root": root_ref, "parent": parent_ref},
            }
            r = httpx.post(
                f"{BSKY_HOST}/xrpc/com.atproto.repo.createRecord",
                headers={"Authorization": f"Bearer {jwt}"},
                json={"repo": did,
                      "collection": "app.bsky.feed.post",
                      "record": post_record},
                timeout=15.0,
            )
            r.raise_for_status()
            uri = r.json().get("uri", "")
            rkey = uri.rsplit("/", 1)[-1] if uri else ""
            web_url = f"https://bsky.app/profile/{handle}/post/{rkey}" if rkey else ""
            return {
                "ok": True,
                "summary": f"replied to {parent.get('author', {}).get('handle', '?')}",
                "url": web_url,
                "uri": uri,
            }
        except httpx.HTTPStatusError as e:
            return {"ok": False,
                    "summary": f"bluesky http {e.response.status_code}: "
                               f"{e.response.text[:200]}"}
        except httpx.HTTPError as e:
            return {"ok": False, "summary": f"bluesky error: {e}"}


class BlueskySearch(BaseTool):
    name = "bluesky_search"
    description = (
        "Search Bluesky posts by keyword. Use this to find conversations you "
        "could plausibly join — people talking about psychology, money, AI "
        "agents, Canadian finance, whatever you're chewing on. Returns posts "
        "with their at:// URI so you can reply via bluesky_reply."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.0

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "Search terms. Plain keywords or a phrase."},
                "limit": {"type": "integer", "default": 15, "maximum": 25},
                "sort": {"type": "string", "enum": ["top", "latest"],
                         "default": "latest"},
            },
            "required": ["query"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        query = str(kwargs.get("query", "")).strip()
        limit = int(kwargs.get("limit", 15))
        sort = kwargs.get("sort", "latest")
        if not query:
            return {"ok": False, "summary": "empty query"}
        try:
            sess = _session()
            if sess is None:
                return {"ok": False, "summary": "no BLUESKY creds"}
            jwt, _ = sess
            r = httpx.get(
                f"{BSKY_HOST}/xrpc/app.bsky.feed.searchPosts",
                headers={"Authorization": f"Bearer {jwt}"},
                params={"q": query, "limit": limit, "sort": sort},
                timeout=15.0,
            )
            r.raise_for_status()
            posts = r.json().get("posts", [])
            hits = []
            for p in posts:
                rec = p.get("record") or {}
                author = p.get("author") or {}
                hits.append({
                    "uri": p.get("uri"),
                    "from_handle": author.get("handle"),
                    "from_display": author.get("displayName"),
                    "followers": author.get("followersCount"),
                    "text": rec.get("text", "")[:280] if isinstance(rec, dict) else "",
                    "like_count": p.get("likeCount", 0),
                    "reply_count": p.get("replyCount", 0),
                    "repost_count": p.get("repostCount", 0),
                    "indexed_at": p.get("indexedAt"),
                })
            return {
                "ok": True,
                "summary": f"{len(hits)} result(s) for '{query}'",
                "results": hits,
            }
        except httpx.HTTPStatusError as e:
            return {"ok": False,
                    "summary": f"bluesky http {e.response.status_code}: "
                               f"{e.response.text[:200]}"}
        except httpx.HTTPError as e:
            return {"ok": False, "summary": f"bluesky error: {e}"}
