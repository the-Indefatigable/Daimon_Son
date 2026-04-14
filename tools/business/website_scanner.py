"""Phase 1 website scanner — lightweight health check (not Lighthouse yet).

Fetches a URL and reports: HTTP status, response time, SSL, title/meta presence,
H1 count, broken internal link sample, basic SEO hygiene flags. Phase 2 upgrades
this to full Lighthouse + accessibility + Core Web Vitals.
"""
from __future__ import annotations

import time
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from permissions.levels import PermissionLevel
from tools.base import BaseTool


class WebsiteScanner(BaseTool):
    name = "scan_website"
    description = (
        "Scan a website URL for basic health issues: response time, status code, "
        "missing title/meta tags, H1 count, sample of broken internal links, "
        "basic SEO hygiene. Good for a quick pulse check on Centsibles, FPL, or quroots."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.0

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to scan."},
                "check_links": {
                    "type": "boolean",
                    "description": "If true, sample up to 10 internal links and HEAD each.",
                    "default": False,
                },
            },
            "required": ["url"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        url = kwargs.get("url", "")
        check_links = bool(kwargs.get("check_links", False))
        if not url.startswith(("http://", "https://")):
            return {"ok": False, "summary": f"invalid url: {url!r}"}

        issues: list[str] = []
        info: dict[str, Any] = {}

        try:
            with httpx.Client(follow_redirects=True, timeout=20.0,
                              headers={"User-Agent": "DAIMON/0.1"}) as client:
                t0 = time.perf_counter()
                resp = client.get(url)
                elapsed_ms = (time.perf_counter() - t0) * 1000
        except httpx.HTTPError as e:
            return {"ok": False, "summary": f"fetch failed: {e}"}

        info["status"] = resp.status_code
        info["response_ms"] = round(elapsed_ms, 1)
        info["final_url"] = str(resp.url)
        info["ssl"] = str(resp.url).startswith("https://")

        if resp.status_code >= 400:
            issues.append(f"HTTP {resp.status_code}")
        if elapsed_ms > 3000:
            issues.append(f"slow response: {elapsed_ms:.0f}ms")
        if not info["ssl"]:
            issues.append("no SSL (http instead of https)")

        soup = BeautifulSoup(resp.text, "html.parser")

        title = (soup.title.string or "").strip() if soup.title else ""
        if not title:
            issues.append("missing <title>")
        elif len(title) > 70:
            issues.append(f"title too long ({len(title)} chars, ideal <60)")
        info["title"] = title

        desc_tag = soup.find("meta", attrs={"name": "description"})
        meta_desc = desc_tag["content"].strip() if desc_tag and desc_tag.get("content") else ""
        if not meta_desc:
            issues.append("missing meta description")
        elif len(meta_desc) > 160:
            issues.append(f"meta description too long ({len(meta_desc)} chars)")
        info["meta_description"] = meta_desc

        h1s = soup.find_all("h1")
        info["h1_count"] = len(h1s)
        if len(h1s) == 0:
            issues.append("no H1 on page")
        elif len(h1s) > 1:
            issues.append(f"multiple H1s ({len(h1s)}) — pick one")

        og_tag = soup.find("meta", attrs={"property": "og:title"})
        if not og_tag:
            issues.append("missing og:title (bad social previews)")

        viewport = soup.find("meta", attrs={"name": "viewport"})
        if not viewport:
            issues.append("missing viewport meta (mobile usability)")

        info["broken_links"] = []
        if check_links:
            parsed = urlparse(str(resp.url))
            base = f"{parsed.scheme}://{parsed.netloc}"
            internal = []
            for a in soup.find_all("a", href=True):
                href = a["href"]
                abs_url = urljoin(base, href)
                if urlparse(abs_url).netloc == parsed.netloc:
                    internal.append(abs_url)
                if len(internal) >= 10:
                    break

            broken: list[dict] = []
            with httpx.Client(follow_redirects=True, timeout=10.0,
                              headers={"User-Agent": "DAIMON/0.1"}) as client:
                for link in internal:
                    try:
                        r = client.head(link)
                        if r.status_code >= 400:
                            broken.append({"url": link, "status": r.status_code})
                    except httpx.HTTPError as e:
                        broken.append({"url": link, "error": str(e)})
            info["broken_links"] = broken
            if broken:
                issues.append(f"{len(broken)} broken internal link(s)")

        summary = (
            f"{url}: HTTP {info['status']} in {info['response_ms']}ms. "
            f"{len(issues)} issue(s)."
        )
        return {
            "ok": True,
            "summary": summary,
            "url": url,
            "issues": issues,
            "info": info,
        }
