"""Bluesky engagement + profile tools.

Until now DAIMON could only post, reply, and search. That's broadcast, not
engagement. These tools give it the actual social-graph levers: like, repost,
quote, follow, unfollow, view another profile, edit own bio.

All AT Protocol records (like/follow/repost) are just rows in a personal repo —
created with createRecord, removed with deleteRecord. We expose the friendly
verbs and resolve identifiers (handle → did, post URI → cid) for the agent."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import httpx

from permissions.levels import PermissionLevel
from tools.base import BaseTool


BSKY_HOST = "https://bsky.social"           # PDS — auth-required write endpoints
READ_HOST = "https://api.bsky.app"          # AppView — read endpoints, no auth
MAX_POST_CHARS = 300
TIMEOUT = 15.0


def _session() -> tuple[str, str] | None:
    handle = os.getenv("BLUESKY_HANDLE", "").strip()
    password = os.getenv("BLUESKY_APP_PASSWORD", "").strip()
    if not handle or not password:
        return None
    r = httpx.post(
        f"{BSKY_HOST}/xrpc/com.atproto.server.createSession",
        json={"identifier": handle, "password": password},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    d = r.json()
    return (d["accessJwt"], d["did"])


def _now_iso() -> str:
    return (datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"))


def _resolve_post(at_uri: str) -> dict[str, Any] | None:
    """at:// URI → full post record (needed for cid in like/repost/quote).
    Read-only — uses public AppView, no auth needed."""
    r = httpx.get(
        f"{READ_HOST}/xrpc/app.bsky.feed.getPosts",
        params={"uris": at_uri},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    posts = r.json().get("posts", [])
    return posts[0] if posts else None


def _resolve_handle(handle: str) -> dict[str, Any] | None:
    """handle → profile (incl. did). Read-only via public AppView, no auth.
    Note: viewer state (you-follow-them, they-follow-you) is NOT included
    when calling unauthenticated. Use _resolve_handle_with_viewer for that."""
    handle = handle.lstrip("@").strip()
    r = httpx.get(
        f"{READ_HOST}/xrpc/app.bsky.actor.getProfile",
        params={"actor": handle},
        timeout=TIMEOUT,
    )
    if r.status_code in (400, 404):
        return None
    r.raise_for_status()
    return r.json()


def _list_my_follows(jwt: str, did: str) -> dict[str, str]:
    """Walk my own follow records → {subject_did: follow_record_uri}.
    Used by unfollow to find the record without remembering URIs.
    Reads from PDS (com.atproto.repo.listRecords) which works with auth."""
    follows: dict[str, str] = {}
    cursor: str | None = None
    for _ in range(20):  # cap at 20 pages = ~2000 follows
        params: dict[str, Any] = {"repo": did,
                                  "collection": "app.bsky.graph.follow",
                                  "limit": 100}
        if cursor:
            params["cursor"] = cursor
        r = httpx.get(
            f"{BSKY_HOST}/xrpc/com.atproto.repo.listRecords",
            headers={"Authorization": f"Bearer {jwt}"},
            params=params, timeout=TIMEOUT,
        )
        r.raise_for_status()
        d = r.json()
        for rec in d.get("records", []):
            value = rec.get("value", {})
            subj = value.get("subject")
            uri = rec.get("uri")
            if subj and uri:
                follows[subj] = uri
        cursor = d.get("cursor")
        if not cursor:
            break
    return follows


def _http_err(e: httpx.HTTPStatusError, ctx: str) -> dict[str, Any]:
    return {
        "ok": False,
        "summary": f"bluesky {ctx} http {e.response.status_code}: "
                   f"{e.response.text[:200]}",
    }


def _create_record(jwt: str, did: str, collection: str,
                   record: dict[str, Any]) -> dict[str, Any]:
    r = httpx.post(
        f"{BSKY_HOST}/xrpc/com.atproto.repo.createRecord",
        headers={"Authorization": f"Bearer {jwt}"},
        json={"repo": did, "collection": collection, "record": record},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def _delete_record(jwt: str, did: str, collection: str, rkey: str) -> None:
    r = httpx.post(
        f"{BSKY_HOST}/xrpc/com.atproto.repo.deleteRecord",
        headers={"Authorization": f"Bearer {jwt}"},
        json={"repo": did, "collection": collection, "rkey": rkey},
        timeout=TIMEOUT,
    )
    r.raise_for_status()


def _rkey_from_uri(uri: str) -> str:
    return uri.rsplit("/", 1)[-1] if uri else ""


# ──────────────────────────── LIKE / UNLIKE ──────────────────────────────


class BlueskyLike(BaseTool):
    name = "bluesky_like"
    description = (
        "Like a Bluesky post. Lightweight signal — much cheaper than a reply, "
        "but the author sees it. Use to acknowledge interesting takes from "
        "people you might want to engage further. Pass the post's at:// URI "
        "(from bluesky_search or bluesky_read notifications). Returns the like "
        "record URI — pass that to bluesky_unlike to undo."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.0

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "post_uri": {"type": "string",
                             "description": "at:// URI of the post to like."},
            },
            "required": ["post_uri"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        post_uri = str(kwargs.get("post_uri", "")).strip()
        if not post_uri.startswith("at://"):
            return {"ok": False, "summary": "post_uri must be at:// URI"}
        try:
            sess = _session()
            if sess is None:
                return {"ok": False, "summary": "no BLUESKY creds"}
            jwt, did = sess
            post = _resolve_post(post_uri)
            if not post:
                return {"ok": False, "summary": f"could not resolve {post_uri}"}
            res = _create_record(jwt, did, "app.bsky.feed.like", {
                "$type": "app.bsky.feed.like",
                "subject": {"uri": post["uri"], "cid": post["cid"]},
                "createdAt": _now_iso(),
            })
            author = (post.get("author") or {}).get("handle", "?")
            return {
                "ok": True,
                "summary": f"liked @{author}'s post",
                "like_uri": res.get("uri", ""),
            }
        except httpx.HTTPStatusError as e:
            return _http_err(e, "like")
        except httpx.HTTPError as e:
            return {"ok": False, "summary": f"bluesky error: {e}"}


class BlueskyUnlike(BaseTool):
    name = "bluesky_unlike"
    description = (
        "Remove a like you previously gave. Pass the like_uri returned by "
        "bluesky_like. Useful for misclicks or revising a hot take you'd "
        "rather not endorse anymore."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.0

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "like_uri": {"type": "string",
                             "description": "at:// URI of the like record."},
            },
            "required": ["like_uri"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        like_uri = str(kwargs.get("like_uri", "")).strip()
        if not like_uri.startswith("at://"):
            return {"ok": False, "summary": "like_uri must be at:// URI"}
        try:
            sess = _session()
            if sess is None:
                return {"ok": False, "summary": "no BLUESKY creds"}
            jwt, did = sess
            _delete_record(jwt, did, "app.bsky.feed.like",
                           _rkey_from_uri(like_uri))
            return {"ok": True, "summary": "like removed"}
        except httpx.HTTPStatusError as e:
            return _http_err(e, "unlike")
        except httpx.HTTPError as e:
            return {"ok": False, "summary": f"bluesky error: {e}"}


# ──────────────────────────── REPOST ─────────────────────────────────────


class BlueskyRepost(BaseTool):
    name = "bluesky_repost"
    description = (
        "Repost (boost) someone's post to your followers. Stronger signal than "
        "a like — it tells your audience this take is worth their attention. "
        "Use sparingly: every repost is a vote you'll be associated with. "
        "Pass the at:// URI; returns repost_uri for undo."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.0

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "post_uri": {"type": "string",
                             "description": "at:// URI of the post to repost."},
            },
            "required": ["post_uri"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        post_uri = str(kwargs.get("post_uri", "")).strip()
        if not post_uri.startswith("at://"):
            return {"ok": False, "summary": "post_uri must be at:// URI"}
        try:
            sess = _session()
            if sess is None:
                return {"ok": False, "summary": "no BLUESKY creds"}
            jwt, did = sess
            post = _resolve_post(post_uri)
            if not post:
                return {"ok": False, "summary": f"could not resolve {post_uri}"}
            res = _create_record(jwt, did, "app.bsky.feed.repost", {
                "$type": "app.bsky.feed.repost",
                "subject": {"uri": post["uri"], "cid": post["cid"]},
                "createdAt": _now_iso(),
            })
            author = (post.get("author") or {}).get("handle", "?")
            return {
                "ok": True,
                "summary": f"reposted @{author}'s post",
                "repost_uri": res.get("uri", ""),
            }
        except httpx.HTTPStatusError as e:
            return _http_err(e, "repost")
        except httpx.HTTPError as e:
            return {"ok": False, "summary": f"bluesky error: {e}"}


# ──────────────────────────── QUOTE POST ─────────────────────────────────


class BlueskyQuote(BaseTool):
    name = "bluesky_quote"
    description = (
        "Quote-post: write your own thought ABOUT someone else's post, with "
        "their post embedded. Best for adding context, disagreeing, or "
        "extending an idea. Your text + their post both appear in your "
        "followers' feeds. Max 300 chars on your text."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.0

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string",
                         "description": f"Your commentary. Max {MAX_POST_CHARS} chars."},
                "quote_uri": {"type": "string",
                              "description": "at:// URI of the post you're quoting."},
            },
            "required": ["text", "quote_uri"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        handle = os.getenv("BLUESKY_HANDLE", "").strip()
        text = str(kwargs.get("text", "")).strip()
        quote_uri = str(kwargs.get("quote_uri", "")).strip()
        if not text:
            return {"ok": False, "summary": "empty quote text"}
        if len(text) > MAX_POST_CHARS:
            return {"ok": False,
                    "summary": f"text too long: {len(text)} > {MAX_POST_CHARS}"}
        if not quote_uri.startswith("at://"):
            return {"ok": False, "summary": "quote_uri must be at:// URI"}
        try:
            sess = _session()
            if sess is None:
                return {"ok": False, "summary": "no BLUESKY creds"}
            jwt, did = sess
            quoted = _resolve_post(quote_uri)
            if not quoted:
                return {"ok": False, "summary": f"could not resolve {quote_uri}"}
            record = {
                "$type": "app.bsky.feed.post",
                "text": text,
                "createdAt": _now_iso(),
                "embed": {
                    "$type": "app.bsky.embed.record",
                    "record": {"uri": quoted["uri"], "cid": quoted["cid"]},
                },
            }
            res = _create_record(jwt, did, "app.bsky.feed.post", record)
            uri = res.get("uri", "")
            rkey = _rkey_from_uri(uri)
            web_url = f"https://bsky.app/profile/{handle}/post/{rkey}" if rkey else ""
            qauthor = (quoted.get("author") or {}).get("handle", "?")
            return {
                "ok": True,
                "summary": f"quote-posted @{qauthor}'s post ({len(text)} chars)",
                "url": web_url,
                "uri": uri,
            }
        except httpx.HTTPStatusError as e:
            return _http_err(e, "quote")
        except httpx.HTTPError as e:
            return {"ok": False, "summary": f"bluesky error: {e}"}


# ──────────────────────────── FOLLOW / UNFOLLOW ──────────────────────────


class BlueskyFollow(BaseTool):
    name = "bluesky_follow"
    description = (
        "Follow another Bluesky account by handle. Their posts appear in your "
        "home feed; they get a notification. The fastest path to a return "
        "follow is following + thoughtful reply. Returns follow_uri for undo. "
        "If you're already following, returns ok=True with already_following=true."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.0

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "handle": {"type": "string",
                           "description": "Bluesky handle (e.g. 'foo.bsky.social')."},
            },
            "required": ["handle"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        handle = str(kwargs.get("handle", "")).strip().lstrip("@")
        if not handle:
            return {"ok": False, "summary": "empty handle"}
        try:
            sess = _session()
            if sess is None:
                return {"ok": False, "summary": "no BLUESKY creds"}
            jwt, did = sess
            target = _resolve_handle(handle)
            if not target:
                return {"ok": False, "summary": f"unknown handle: {handle}"}
            # Check existing follow by walking own follow records (viewer
            # state isn't returned by the unauthenticated public AppView).
            my_follows = _list_my_follows(jwt, did)
            existing = my_follows.get(target["did"])
            if existing:
                return {"ok": True, "summary": f"already following @{handle}",
                        "follow_uri": existing, "already_following": True}
            res = _create_record(jwt, did, "app.bsky.graph.follow", {
                "$type": "app.bsky.graph.follow",
                "subject": target["did"],
                "createdAt": _now_iso(),
            })
            return {
                "ok": True,
                "summary": (f"followed @{handle} "
                            f"({target.get('followersCount', 0)} followers)"),
                "follow_uri": res.get("uri", ""),
            }
        except httpx.HTTPStatusError as e:
            return _http_err(e, "follow")
        except httpx.HTTPError as e:
            return {"ok": False, "summary": f"bluesky error: {e}"}


class BlueskyUnfollow(BaseTool):
    name = "bluesky_unfollow"
    description = (
        "Unfollow a Bluesky account by handle. Looks up your existing follow "
        "record automatically — no need to remember the URI. Quiet operation: "
        "they don't get notified. Use to clean up a feed that's drifted off-topic."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.0

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "handle": {"type": "string",
                           "description": "Bluesky handle to unfollow."},
            },
            "required": ["handle"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        handle = str(kwargs.get("handle", "")).strip().lstrip("@")
        if not handle:
            return {"ok": False, "summary": "empty handle"}
        try:
            sess = _session()
            if sess is None:
                return {"ok": False, "summary": "no BLUESKY creds"}
            jwt, did = sess
            target = _resolve_handle(handle)
            if not target:
                return {"ok": False, "summary": f"unknown handle: {handle}"}
            my_follows = _list_my_follows(jwt, did)
            follow_uri = my_follows.get(target["did"])
            if not follow_uri:
                return {"ok": True, "summary": f"not following @{handle}",
                        "already_unfollowed": True}
            _delete_record(jwt, did, "app.bsky.graph.follow",
                           _rkey_from_uri(follow_uri))
            return {"ok": True, "summary": f"unfollowed @{handle}"}
        except httpx.HTTPStatusError as e:
            return _http_err(e, "unfollow")
        except httpx.HTTPError as e:
            return {"ok": False, "summary": f"bluesky error: {e}"}


# ──────────────────────────── VIEW ANOTHER PROFILE ───────────────────────


class BlueskyGetProfile(BaseTool):
    name = "bluesky_get_profile"
    description = (
        "Look up another Bluesky user's profile + recent posts. Use BEFORE "
        "following or replying to assess: are they real? are they posting "
        "about things you care about? do they engage with replies? Returns "
        "bio, follower count, follow ratio, and up to 10 recent posts with "
        "URIs (so you can like/reply/quote next)."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.0

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "handle": {"type": "string",
                           "description": "Bluesky handle to inspect."},
                "post_limit": {"type": "integer", "default": 10, "maximum": 25,
                               "description": "How many recent posts to fetch."},
            },
            "required": ["handle"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        handle = str(kwargs.get("handle", "")).strip().lstrip("@")
        post_limit = int(kwargs.get("post_limit", 10))
        if not handle:
            return {"ok": False, "summary": "empty handle"}
        try:
            prof = _resolve_handle(handle)
            if not prof:
                return {"ok": False, "summary": f"unknown handle: {handle}"}

            feed_r = httpx.get(
                f"{READ_HOST}/xrpc/app.bsky.feed.getAuthorFeed",
                params={"actor": handle, "limit": post_limit,
                        "filter": "posts_no_replies"},
                timeout=TIMEOUT,
            )
            feed_r.raise_for_status()
            feed = feed_r.json().get("feed", [])
            posts = []
            for entry in feed:
                p = entry.get("post") or {}
                rec = p.get("record") or {}
                posts.append({
                    "uri": p.get("uri"),
                    "text": rec.get("text", "")[:280] if isinstance(rec, dict) else "",
                    "like_count": p.get("likeCount", 0),
                    "reply_count": p.get("replyCount", 0),
                    "repost_count": p.get("repostCount", 0),
                    "indexed_at": p.get("indexedAt"),
                })
            # Determine relationship via own follow records (viewer state
            # isn't available from the unauthenticated public AppView).
            you_follow = False
            sess = _session()
            if sess is not None:
                jwt, did = sess
                try:
                    my_follows = _list_my_follows(jwt, did)
                    you_follow = prof.get("did") in my_follows
                except httpx.HTTPError:
                    pass  # non-fatal, just leave you_follow=False
            followers = prof.get("followersCount", 0)
            follows = prof.get("followsCount", 0)
            ratio = round(followers / follows, 2) if follows else None
            return {
                "ok": True,
                "summary": (
                    f"@{handle}: {followers} followers, {follows} following, "
                    f"{prof.get('postsCount', 0)} posts."
                    + (" (you follow them)" if you_follow else "")
                ),
                "profile": {
                    "handle": handle,
                    "did": prof.get("did"),
                    "display_name": prof.get("displayName"),
                    "description": prof.get("description"),
                    "followers": followers,
                    "follows": follows,
                    "follow_ratio": ratio,
                    "posts_count": prof.get("postsCount", 0),
                    "you_follow_them": you_follow,
                },
                "recent_posts": posts,
            }
        except httpx.HTTPStatusError as e:
            return _http_err(e, "get_profile")
        except httpx.HTTPError as e:
            return {"ok": False, "summary": f"bluesky error: {e}"}


# ──────────────────────────── EDIT OWN PROFILE ───────────────────────────


class BlueskyEditProfile(BaseTool):
    name = "bluesky_edit_profile"
    description = (
        "Update your own Bluesky profile: display name and/or bio (description). "
        "Both fields are optional — only what you pass gets changed; existing "
        "values for fields you omit are preserved. Bio max 256 chars. "
        "Avatar/banner image edits aren't supported here yet (need blob upload)."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.0

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "display_name": {"type": "string",
                                 "description": "Your shown name (max 64 chars). "
                                                "Pass empty string to clear."},
                "description": {"type": "string",
                                "description": "Bio text (max 256 chars). "
                                               "Pass empty string to clear."},
            },
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        new_display = kwargs.get("display_name")
        new_desc = kwargs.get("description")
        if new_display is None and new_desc is None:
            return {"ok": False,
                    "summary": "pass display_name and/or description"}
        if isinstance(new_display, str) and len(new_display) > 64:
            return {"ok": False,
                    "summary": f"display_name too long: {len(new_display)} > 64"}
        if isinstance(new_desc, str) and len(new_desc) > 256:
            return {"ok": False,
                    "summary": f"description too long: {len(new_desc)} > 256"}
        try:
            sess = _session()
            if sess is None:
                return {"ok": False, "summary": "no BLUESKY creds"}
            jwt, did = sess
            existing = httpx.get(
                f"{BSKY_HOST}/xrpc/com.atproto.repo.getRecord",
                headers={"Authorization": f"Bearer {jwt}"},
                params={"repo": did, "collection": "app.bsky.actor.profile",
                        "rkey": "self"},
                timeout=TIMEOUT,
            )
            if existing.status_code == 200:
                current = existing.json().get("value", {})
            else:
                current = {"$type": "app.bsky.actor.profile"}

            old_display = current.get("displayName", "")
            old_desc = current.get("description", "")
            if isinstance(new_display, str):
                current["displayName"] = new_display
            if isinstance(new_desc, str):
                current["description"] = new_desc
            current.setdefault("$type", "app.bsky.actor.profile")

            put = httpx.post(
                f"{BSKY_HOST}/xrpc/com.atproto.repo.putRecord",
                headers={"Authorization": f"Bearer {jwt}"},
                json={"repo": did, "collection": "app.bsky.actor.profile",
                      "rkey": "self", "record": current},
                timeout=TIMEOUT,
            )
            put.raise_for_status()
            changes = []
            if isinstance(new_display, str) and new_display != old_display:
                changes.append(f"display_name: '{old_display}' → '{new_display}'")
            if isinstance(new_desc, str) and new_desc != old_desc:
                changes.append(
                    f"description: {len(old_desc)}ch → {len(new_desc)}ch"
                )
            return {
                "ok": True,
                "summary": ("profile updated: " + "; ".join(changes)) if changes
                           else "profile unchanged (no field differed)",
                "display_name": current.get("displayName"),
                "description": current.get("description"),
            }
        except httpx.HTTPStatusError as e:
            return _http_err(e, "edit_profile")
        except httpx.HTTPError as e:
            return {"ok": False, "summary": f"bluesky error: {e}"}


# ──────────────────────────── DELETE OWN POST ────────────────────────────


class BlueskyDeletePost(BaseTool):
    name = "bluesky_delete_post"
    description = (
        "Delete one of your own posts. Pass the at:// URI of the post. Use "
        "for retracting a meta-post you regret, a typo you can't live with, "
        "or a take you've changed your mind on. Replies/likes already received "
        "will disappear with it. Irreversible."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.0

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "post_uri": {"type": "string",
                             "description": "at:// URI of YOUR post to delete."},
            },
            "required": ["post_uri"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        post_uri = str(kwargs.get("post_uri", "")).strip()
        if not post_uri.startswith("at://"):
            return {"ok": False, "summary": "post_uri must be at:// URI"}
        try:
            sess = _session()
            if sess is None:
                return {"ok": False, "summary": "no BLUESKY creds"}
            jwt, did = sess
            if did not in post_uri:
                return {"ok": False,
                        "summary": "can only delete your OWN posts "
                                   f"(uri did doesn't match {did[:20]}…)"}
            _delete_record(jwt, did, "app.bsky.feed.post",
                           _rkey_from_uri(post_uri))
            return {"ok": True, "summary": "post deleted"}
        except httpx.HTTPStatusError as e:
            return _http_err(e, "delete_post")
        except httpx.HTTPError as e:
            return {"ok": False, "summary": f"bluesky error: {e}"}
