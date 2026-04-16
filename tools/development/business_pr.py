"""Open PRs against Mohammad's business repos (currently Centsibles only).

Scoped to an allowlist — DAIMON cannot touch anything outside it. Uses a
separate fine-grained PAT (GITHUB_BUSINESS_PAT) with write access narrowed
to those repos. DAIMON proposes; Mohammad reviews and merges."""
from __future__ import annotations

import base64
import os
import time
from typing import Any

import httpx

from permissions.levels import PermissionLevel
from tools.base import BaseTool


GITHUB_API = "https://api.github.com"

MAX_FILES_PER_PR = 20
MAX_BYTES_PER_FILE = 50_000


def _headers(token: str) -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "DAIMON/0.1",
    }


def _allowlist() -> list[str]:
    raw = os.getenv("GITHUB_BUSINESS_REPOS", "")
    return [r.strip() for r in raw.split(",") if r.strip()]


class GitHubBusinessPR(BaseTool):
    name = "github_business_pr"
    description = (
        "Propose a change to one of Mohammad's business repos by opening a pull "
        "request. You have write access ONLY to repos in the allowlist (currently "
        "centsibles-frontend, centsibles-backend). Mohammad reviews and merges — "
        "you cannot merge yourself. Use this to actually help the businesses: "
        "SEO fixes, landing-page copy, bug fixes, new features, meta tags. "
        "Every file you include: (path, new content). Max 20 files per PR, max "
        "50KB per file. PR body must explain what_changed, why, and risk."
    )
    permission_level = PermissionLevel.AUTO  # PR itself is the approval gate
    cost_per_use = 0.0

    def input_schema(self) -> dict[str, Any]:
        allow = _allowlist()
        return {
            "type": "object",
            "properties": {
                "repo": {
                    "type": "string",
                    "description": "Target repo as owner/name. Must be in the "
                                   f"allowlist: {', '.join(allow) or '(none set)'}",
                },
                "title": {
                    "type": "string",
                    "description": "Short PR title. Under 80 chars.",
                },
                "what_changed": {
                    "type": "string",
                    "description": "What the PR changes, in plain language.",
                },
                "why": {
                    "type": "string",
                    "description": "Why you want this change. Expected impact "
                                   "on users / traffic / revenue if relevant.",
                },
                "risk": {
                    "type": "string",
                    "description": "What could go wrong. Honest risk assessment.",
                },
                "branch_name": {
                    "type": "string",
                    "description": "Short, URL-safe branch name (e.g. "
                                   "'daimon/landing-meta-tags'). Prefixed with 'daimon/'.",
                },
                "files": {
                    "type": "array",
                    "description": "List of files to write. Each is {path, content}. "
                                   f"Max {MAX_FILES_PER_PR} files.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path relative to repo root.",
                            },
                            "content": {
                                "type": "string",
                                "description": "Full new file content.",
                            },
                        },
                        "required": ["path", "content"],
                    },
                    "minItems": 1,
                    "maxItems": MAX_FILES_PER_PR,
                },
            },
            "required": ["repo", "title", "what_changed", "why", "risk",
                        "branch_name", "files"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        token = os.getenv("GITHUB_BUSINESS_PAT", "").strip()
        if not token:
            return {
                "ok": False,
                "summary": "no GITHUB_BUSINESS_PAT — file a resource request",
                "needs_resource": "GITHUB_BUSINESS_PAT",
            }

        allow = _allowlist()
        if not allow:
            return {"ok": False, "summary": "GITHUB_BUSINESS_REPOS allowlist empty"}

        repo = str(kwargs.get("repo", "")).strip()
        if repo not in allow:
            return {
                "ok": False,
                "summary": f"repo '{repo}' not in allowlist: {allow}",
            }

        title = str(kwargs.get("title", "")).strip()[:80]
        what = str(kwargs.get("what_changed", "")).strip()
        why = str(kwargs.get("why", "")).strip()
        risk = str(kwargs.get("risk", "")).strip()
        branch_raw = str(kwargs.get("branch_name", "")).strip()
        files = kwargs.get("files") or []

        if not all([title, what, why, risk, branch_raw, files]):
            return {"ok": False, "summary": "missing required field(s)"}
        if len(files) > MAX_FILES_PER_PR:
            return {"ok": False,
                    "summary": f"too many files ({len(files)} > {MAX_FILES_PER_PR})"}

        for f in files:
            if not isinstance(f, dict) or "path" not in f or "content" not in f:
                return {"ok": False, "summary": "malformed file entry"}
            path = str(f["path"])
            if path.startswith("/") or ".." in path.split("/"):
                return {"ok": False, "summary": f"unsafe path: {path}"}
            content = str(f["content"])
            if len(content.encode("utf-8")) > MAX_BYTES_PER_FILE:
                return {"ok": False,
                        "summary": f"{path} exceeds {MAX_BYTES_PER_FILE}B cap"}

        branch_safe = "".join(
            c if (c.isalnum() or c in "-_/") else "-" for c in branch_raw
        ).strip("-/")
        if not branch_safe:
            return {"ok": False, "summary": "invalid branch name"}
        if not branch_safe.startswith("daimon/"):
            branch_safe = f"daimon/{branch_safe}"
        branch_safe = f"{branch_safe}-{int(time.time())}"

        h = _headers(token)
        try:
            repo_info = httpx.get(f"{GITHUB_API}/repos/{repo}",
                                  headers=h, timeout=15.0)
            repo_info.raise_for_status()
            default_branch = repo_info.json().get("default_branch", "main")

            ref = httpx.get(
                f"{GITHUB_API}/repos/{repo}/git/ref/heads/{default_branch}",
                headers=h, timeout=15.0,
            )
            ref.raise_for_status()
            base_sha = ref.json()["object"]["sha"]

            new_ref = httpx.post(
                f"{GITHUB_API}/repos/{repo}/git/refs",
                headers=h, timeout=15.0,
                json={"ref": f"refs/heads/{branch_safe}", "sha": base_sha},
            )
            new_ref.raise_for_status()

            committed_paths: list[str] = []
            for f in files:
                path = str(f["path"])
                content = str(f["content"])
                existing = httpx.get(
                    f"{GITHUB_API}/repos/{repo}/contents/{path}",
                    headers=h, timeout=15.0,
                    params={"ref": branch_safe},
                )
                body: dict[str, Any] = {
                    "message": f"{title}: update {path}",
                    "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
                    "branch": branch_safe,
                }
                if existing.status_code == 200:
                    body["sha"] = existing.json().get("sha", "")
                elif existing.status_code not in (404,):
                    existing.raise_for_status()
                put = httpx.put(
                    f"{GITHUB_API}/repos/{repo}/contents/{path}",
                    headers=h, timeout=30.0, json=body,
                )
                put.raise_for_status()
                committed_paths.append(path)

            pr_body = (
                f"**What changed**\n{what}\n\n"
                f"**Why**\n{why}\n\n"
                f"**Risk**\n{risk}\n\n"
                f"**Files touched**\n" + "\n".join(f"- `{p}`" for p in committed_paths)
                + "\n\n---\n*Opened by DAIMON. Merging requires human review.*"
            )
            pr = httpx.post(
                f"{GITHUB_API}/repos/{repo}/pulls",
                headers=h, timeout=15.0,
                json={
                    "title": title,
                    "head": branch_safe,
                    "base": default_branch,
                    "body": pr_body,
                    "maintainer_can_modify": True,
                },
            )
            pr.raise_for_status()
            pr_data = pr.json()
            return {
                "ok": True,
                "summary": f"PR opened on {repo}: #{pr_data['number']} — {title}",
                "pr_url": pr_data.get("html_url"),
                "pr_number": pr_data.get("number"),
                "repo": repo,
                "branch": branch_safe,
                "files": committed_paths,
            }
        except httpx.HTTPStatusError as e:
            return {
                "ok": False,
                "summary": f"github http {e.response.status_code}: "
                           f"{e.response.text[:300]}",
            }
        except httpx.HTTPError as e:
            return {"ok": False, "summary": f"github error: {e}"}
