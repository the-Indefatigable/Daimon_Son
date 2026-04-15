"""Web search via Tavily. DAIMON's way to discover things it doesn't already
know the URL for. Free tier: 1,000 searches/month."""
from __future__ import annotations

import os
from typing import Any

import httpx

from permissions.levels import PermissionLevel
from tools.base import BaseTool


TAVILY_URL = "https://api.tavily.com/search"


class WebSearch(BaseTool):
    name = "web_search"
    description = (
        "Search the web for information. Returns synthesized results with "
        "titles, URLs, and content snippets — good for research, learning, "
        "discovering things you don't know yet. Use this BEFORE web_browser "
        "when you don't have a specific URL in mind. "
        "search_depth='basic' is cheap and fast; 'advanced' costs more API "
        "credits but gets deeper results. Limit 1–10 results."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.0  # Tavily free tier

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "How many results to return (1–10).",
                    "minimum": 1,
                    "maximum": 10,
                },
                "search_depth": {
                    "type": "string",
                    "enum": ["basic", "advanced"],
                    "description": "'basic' for most queries; 'advanced' for "
                                   "harder research (costs more credits).",
                },
            },
            "required": ["query", "max_results", "search_depth"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        api_key = os.getenv("TAVILY_API_KEY", "").strip()
        if not api_key:
            return {
                "ok": False,
                "summary": "no TAVILY_API_KEY — file a resource request for web search",
                "needs_resource": "TAVILY_API_KEY",
            }
        query = str(kwargs.get("query", "")).strip()
        if not query:
            return {"ok": False, "summary": "empty query"}

        max_results = max(1, min(10, int(kwargs.get("max_results", 5))))
        depth = kwargs.get("search_depth", "basic")
        if depth not in ("basic", "advanced"):
            depth = "basic"

        payload = {
            "api_key": api_key,
            "query": query,
            "search_depth": depth,
            "max_results": max_results,
            "include_answer": True,
        }
        try:
            resp = httpx.post(TAVILY_URL, json=payload, timeout=20.0)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            return {
                "ok": False,
                "summary": f"tavily http {e.response.status_code}: "
                           f"{e.response.text[:200]}",
            }
        except httpx.HTTPError as e:
            return {"ok": False, "summary": f"tavily error: {e}"}

        results = [
            {
                "title": r.get("title", "")[:200],
                "url": r.get("url", ""),
                "content": (r.get("content") or "")[:800],
                "score": round(float(r.get("score", 0)), 3),
            }
            for r in (data.get("results") or [])
        ]
        answer = (data.get("answer") or "").strip()
        return {
            "ok": True,
            "summary": f"{len(results)} result(s) for '{query[:80]}'"
                       + (f" — answer: {answer[:200]}" if answer else ""),
            "answer": answer,
            "results": results,
        }
