"""List DAIMON's own PRs across allowlisted repos and report their status.

Closes the loop on self-PR and business-PR work: did it get merged? Closed? Still open?"""
from __future__ import annotations

import os
from typing import Any

import httpx

from permissions.levels import PermissionLevel
from tools.base import BaseTool


GITHUB_API = "https://api.github.com"


def _headers(token: str) -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "DAIMON/0.1",
    }


class GitHubPRStatus(BaseTool):
    name = "github_pr_status"
    description = (
        "List your own open/merged/closed PRs across your repos. Use this to "
        "check: did Mohammad merge my PR? Was it closed? Is it still waiting? "
        "Scans the Daimon_Son repo and all business repos you have access to. "
        "Only PRs on branches starting with 'daimon/' (i.e., yours)."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.0

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "state": {
                    "type": "string",
                    "enum": ["open", "closed", "all"],
                    "description": "Filter by state. Default 'all'.",
                    "default": "all",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max PRs per repo (default 10).",
                    "default": 10,
                },
            },
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        token = os.getenv("GITHUB_PAT", "").strip()
        if not token:
            return {"ok": False, "summary": "no GITHUB_PAT"}

        state = kwargs.get("state", "all")
        limit = int(kwargs.get("limit", 10))

        repos: list[str] = []
        self_repo = os.getenv("GITHUB_SELF_REPO", "").strip()
        if self_repo:
            repos.append(self_repo)
        biz = [r.strip() for r in os.getenv("GITHUB_BUSINESS_REPOS", "").split(",") if r.strip()]
        repos.extend(biz)
        seen = set()
        repos = [r for r in repos if not (r in seen or seen.add(r))]

        h = _headers(token)
        out: list[dict[str, Any]] = []
        totals = {"open": 0, "merged": 0, "closed": 0}
        for repo in repos:
            try:
                r = httpx.get(
                    f"{GITHUB_API}/repos/{repo}/pulls",
                    headers=h, timeout=15.0,
                    params={"state": state, "per_page": limit,
                            "sort": "updated", "direction": "desc"},
                )
                r.raise_for_status()
                for pr in r.json():
                    head_ref = (pr.get("head") or {}).get("ref", "")
                    if not head_ref.startswith("daimon/"):
                        continue
                    status = (
                        "merged" if pr.get("merged_at")
                        else ("open" if pr["state"] == "open" else "closed")
                    )
                    totals[status] = totals.get(status, 0) + 1
                    out.append({
                        "repo": repo,
                        "number": pr["number"],
                        "title": pr["title"],
                        "status": status,
                        "url": pr["html_url"],
                        "updated_at": pr["updated_at"],
                        "branch": head_ref,
                    })
            except httpx.HTTPError as e:
                out.append({"repo": repo, "error": str(e)[:150]})

        return {
            "ok": True,
            "summary": f"{len(out)} of your PRs: "
                       f"{totals['open']} open, {totals['merged']} merged, "
                       f"{totals['closed']} closed",
            "totals": totals,
            "prs": out,
        }
