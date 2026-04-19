"""DAIMON's self-modification tool. Opens PRs against its own repo (Daimon_Son)
and only that repo. Never merges — Mohammad reviews and merges manually.

This is how DAIMON evolves its own body."""
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


class GitHubProposePR(BaseTool):
    name = "github_propose_pr"
    is_high_stakes = True
    description = (
        "Propose a change to your OWN codebase (the daimon repo) by opening a "
        "pull request. Mohammad reviews and merges. You cannot merge yourself. "
        "Use this to evolve your body: add new tools, fix bugs in your loop, "
        "improve your own prompts, adjust your own schema. "
        "Every file you include: (path, new content). Max 20 files per PR, max "
        "50KB per file. PR body must explain what_changed, why, and risk. "
        "This tool is hardcoded to your own repo only — you cannot use it to "
        "touch Mohammad's business repos."
    )
    permission_level = PermissionLevel.AUTO  # PR is already the approval gate
    cost_per_use = 0.0

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
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
                    "description": "Why you want this change. Your reasoning.",
                },
                "risk": {
                    "type": "string",
                    "description": "What could go wrong. Honest risk assessment.",
                },
                "branch_name": {
                    "type": "string",
                    "description": "Short, URL-safe branch name (e.g. "
                                   "'daimon/add-email-tool'). Will be prefixed with 'daimon/'.",
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
            "required": ["title", "what_changed", "why", "risk",
                        "branch_name", "files"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        token = os.getenv("GITHUB_SELF_PAT", "").strip()
        repo = os.getenv("GITHUB_SELF_REPO", "").strip()
        if not token or not repo:
            return {
                "ok": False,
                "summary": "no GITHUB_SELF_PAT or GITHUB_SELF_REPO — file a "
                           "resource request",
                "needs_resource": "GITHUB_SELF_PAT",
            }

        # Hardcoded scope: never touch anything but the self repo
        if "/" not in repo:
            return {"ok": False, "summary": f"malformed self-repo: {repo}"}

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

        # Namespace the branch under daimon/ so human-made branches stay separate
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
            # 1. Get default branch + its tip SHA
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

            # 2. Create new branch from base SHA
            new_ref = httpx.post(
                f"{GITHUB_API}/repos/{repo}/git/refs",
                headers=h, timeout=15.0,
                json={"ref": f"refs/heads/{branch_safe}", "sha": base_sha},
            )
            new_ref.raise_for_status()

            # 3. Commit each file via Contents API (one commit per file is
            # simpler than building a tree, and for DAIMON's small PRs it's fine)
            committed_paths: list[str] = []
            for f in files:
                path = str(f["path"])
                content = str(f["content"])
                # If file exists, fetch its sha on this branch so we can update
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

            # 4. Open the PR
            pr_body = (
                f"**What changed**\n{what}\n\n"
                f"**Why**\n{why}\n\n"
                f"**Risk**\n{risk}\n\n"
                f"**Files touched**\n" + "\n".join(f"- `{p}`" for p in committed_paths)
                + "\n\n---\n*Opened by DAIMON (cycle-level self-modification). "
                  "Merging requires human review.*"
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
                "summary": f"PR opened: #{pr_data['number']} — {title}",
                "pr_url": pr_data.get("html_url"),
                "pr_number": pr_data.get("number"),
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
