"""Fetch a URL and return a clean-ish summary (title, headings, main text)."""
from __future__ import annotations

from typing import Any

import httpx
from bs4 import BeautifulSoup

from permissions.levels import PermissionLevel
from tools.base import BaseTool


class WebBrowser(BaseTool):
    name = "web_browser"
    description = (
        "Fetch a URL and extract readable content: title, meta description, "
        "headings, and the main text. Use for research, competitor scans, "
        "blog reading, checking external docs."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.0

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL (http/https)."},
                "max_chars": {
                    "type": "integer",
                    "description": "Cap on the returned text length. Default 6000.",
                    "default": 6000,
                },
            },
            "required": ["url"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        url = kwargs.get("url", "")
        max_chars = int(kwargs.get("max_chars", 6000))
        if not url.startswith(("http://", "https://")):
            return {"ok": False, "summary": f"invalid url: {url!r}"}

        try:
            with httpx.Client(follow_redirects=True, timeout=20.0,
                              headers={"User-Agent": "DAIMON/0.1"}) as client:
                resp = client.get(url)
            status = resp.status_code
        except httpx.HTTPError as e:
            return {"ok": False, "summary": f"fetch failed: {e}"}

        if status >= 400:
            return {"ok": False, "summary": f"http {status}", "status": status}

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()

        title = (soup.title.string or "").strip() if soup.title else ""
        meta_desc = ""
        desc_tag = soup.find("meta", attrs={"name": "description"})
        if desc_tag and desc_tag.get("content"):
            meta_desc = desc_tag["content"].strip()

        headings = [h.get_text(strip=True) for h in soup.find_all(["h1", "h2", "h3"])]
        headings = [h for h in headings if h][:30]

        text = soup.get_text(separator="\n", strip=True)
        text = "\n".join(line for line in text.splitlines() if line.strip())
        text = text[:max_chars]

        return {
            "ok": True,
            "summary": f"fetched {url} ({status}) — {len(text)} chars of text",
            "status": status,
            "title": title,
            "meta_description": meta_desc,
            "headings": headings,
            "text": text,
        }
