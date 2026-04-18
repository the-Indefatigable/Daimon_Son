"""Jina Reader wrapper — pulls clean markdown from JS-rendered pages.

r.jina.ai renders a URL in a real browser on their side and returns
readable markdown. Handles SPAs, dynamic content, and most auth-free
pages that `web_browser` can't parse.

Free tier works without a key (rate-limited). Set JINA_API_KEY for
higher limits and priority."""
from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote

import httpx

from permissions.levels import PermissionLevel
from tools.base import BaseTool


JINA_READER_HOST = "https://r.jina.ai"


class WebReadClean(BaseTool):
    name = "web_read_clean"
    description = (
        "Fetch a URL as clean markdown via Jina Reader. Renders JS on "
        "their side, so this works on modern SPAs (React/Vue/Next) and "
        "sites where web_browser returns empty text. Prefer this over "
        "web_browser when a page looks client-rendered or returned "
        "suspiciously little text. Free tier — no per-call cost."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.0

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full URL (http/https).",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Cap on returned markdown length. Default 8000.",
                    "default": 8000,
                },
            },
            "required": ["url"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        url = str(kwargs.get("url", "")).strip()
        max_chars = int(kwargs.get("max_chars", 8000))
        if not url.startswith(("http://", "https://")):
            return {"ok": False, "summary": f"invalid url: {url!r}"}

        reader_url = f"{JINA_READER_HOST}/{quote(url, safe=':/?#&=%')}"
        headers = {
            "Accept": "text/markdown",
            "X-Return-Format": "markdown",
        }
        api_key = os.getenv("JINA_API_KEY", "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            with httpx.Client(follow_redirects=True, timeout=30.0) as client:
                resp = client.get(reader_url, headers=headers)
        except httpx.HTTPError as e:
            return {"ok": False, "summary": f"jina fetch failed: {e}"}

        if resp.status_code >= 400:
            return {
                "ok": False,
                "summary": f"jina http {resp.status_code}: {resp.text[:200]}",
                "status": resp.status_code,
            }

        markdown = resp.text or ""
        truncated = len(markdown) > max_chars
        if truncated:
            markdown = markdown[:max_chars]

        title = ""
        for line in markdown.splitlines()[:20]:
            if line.startswith("Title:"):
                title = line[len("Title:"):].strip()
                break
            if line.startswith("# "):
                title = line[2:].strip()
                break

        return {
            "ok": True,
            "summary": (
                f"fetched {url} via jina — {len(markdown)} chars"
                + (" (truncated)" if truncated else "")
            ),
            "title": title,
            "markdown": markdown,
            "truncated": truncated,
            "source": "jina_reader",
        }
