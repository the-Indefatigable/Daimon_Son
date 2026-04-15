"""GitHub reader — READ-ONLY. DAIMON can list repos, read files, see commits.
No writes. No PRs. No issues. Earn that later.

Uses the GitHub REST API with a classic PAT. Scope required: `repo` read (or
`public_repo` if only touching public stuff). Fine-grained PAT with contents:read
on specific repos also works."""
from __future__ import annotations

import os
from typing import Any

import httpx

from permissions.levels import PermissionLevel
from tools.base import BaseTool


GITHUB_API = "https://api.github.com"


def _gh_headers() -> dict[str, str]:
    token = os.getenv("GITHUB_PAT", "").strip()
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "DAIMON/0.1",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _has_token() -> bool:
    return bool(os.getenv("GITHUB_PAT", "").strip())


class GitHubListRepos(BaseTool):
    name = "github_list_repos"
    description = (
        "List repositories your operator has given you read access to. Use this "
        "first to discover what codebases exist before reading files from them."
    )
    permission_level = PermissionLevel.AUTO

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "per_page": {"type": "integer", "default": 30, "maximum": 100},
                "visibility": {
                    "type": "string",
                    "enum": ["all", "public", "private"],
                    "default": "all",
                },
            },
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        if not _has_token():
            return {
                "ok": False,
                "summary": "no GITHUB_PAT set — file a resource request for read-only PAT access",
                "needs_resource": "GITHUB_PAT",
            }
        per_page = int(kwargs.get("per_page", 30))
        visibility = kwargs.get("visibility", "all")
        try:
            resp = httpx.get(
                f"{GITHUB_API}/user/repos",
                headers=_gh_headers(),
                params={"per_page": per_page, "visibility": visibility, "sort": "pushed"},
                timeout=15.0,
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            return {"ok": False, "summary": f"github error: {e}"}
        repos = [
            {
                "full_name": r["full_name"],
                "description": r.get("description") or "",
                "default_branch": r.get("default_branch"),
                "language": r.get("language"),
                "pushed_at": r.get("pushed_at"),
                "private": r.get("private"),
            }
            for r in resp.json()
        ]
        return {"ok": True, "summary": f"{len(repos)} repo(s)", "repos": repos}


class GitHubRepoInfo(BaseTool):
    name = "github_repo_info"
    description = (
        "Get metadata about a specific repo: default branch, languages, size, "
        "last push, open issues count, topics. Use before diving into files."
    )
    permission_level = PermissionLevel.AUTO

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "owner/name, e.g. 'alam/centsibles'"},
            },
            "required": ["repo"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        if not _has_token():
            return {"ok": False, "summary": "no GITHUB_PAT set"}
        repo = kwargs["repo"]
        try:
            resp = httpx.get(f"{GITHUB_API}/repos/{repo}",
                             headers=_gh_headers(), timeout=15.0)
            resp.raise_for_status()
            r = resp.json()
            langs = httpx.get(f"{GITHUB_API}/repos/{repo}/languages",
                              headers=_gh_headers(), timeout=15.0).json()
        except httpx.HTTPError as e:
            return {"ok": False, "summary": f"github error: {e}"}
        return {
            "ok": True,
            "summary": f"{repo}: {r.get('language')} | {r.get('default_branch')} | "
                       f"{r.get('stargazers_count', 0)}⭐ | pushed {r.get('pushed_at')}",
            "info": {
                "full_name": r.get("full_name"),
                "description": r.get("description"),
                "default_branch": r.get("default_branch"),
                "languages": langs,
                "size_kb": r.get("size"),
                "pushed_at": r.get("pushed_at"),
                "open_issues": r.get("open_issues_count"),
                "topics": r.get("topics", []),
                "private": r.get("private"),
            },
        }


class GitHubListFiles(BaseTool):
    name = "github_list_files"
    description = (
        "List files and directories at a path in a repo. Use to navigate the "
        "tree before requesting specific file contents."
    )
    permission_level = PermissionLevel.AUTO

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "owner/name"},
                "path": {"type": "string", "default": "",
                         "description": "Directory path. Empty string = root."},
                "ref": {"type": "string",
                        "description": "Branch, tag, or SHA. Defaults to default branch."},
            },
            "required": ["repo"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        if not _has_token():
            return {"ok": False, "summary": "no GITHUB_PAT set"}
        repo = kwargs["repo"]
        path = kwargs.get("path", "")
        ref = kwargs.get("ref")
        url = f"{GITHUB_API}/repos/{repo}/contents/{path}"
        params = {"ref": ref} if ref else {}
        try:
            resp = httpx.get(url, headers=_gh_headers(), params=params, timeout=15.0)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            return {"ok": False, "summary": f"github error: {e}"}
        data = resp.json()
        if isinstance(data, dict):
            data = [data]
        entries = [
            {"name": e["name"], "type": e["type"], "size": e.get("size", 0),
             "path": e["path"]}
            for e in data
        ]
        return {
            "ok": True,
            "summary": f"{repo}/{path or '(root)'}: {len(entries)} entries",
            "entries": entries,
        }


class GitHubReadFile(BaseTool):
    name = "github_read_file"
    description = (
        "Read the contents of a specific file in a repo at a specific ref. "
        "Use for inspecting code, configs, READMEs. Returns decoded text."
    )
    permission_level = PermissionLevel.AUTO

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "owner/name"},
                "path": {"type": "string", "description": "File path from repo root"},
                "ref": {"type": "string"},
                "max_chars": {"type": "integer", "default": 10000},
            },
            "required": ["repo", "path"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        import base64
        if not _has_token():
            return {"ok": False, "summary": "no GITHUB_PAT set"}
        repo = kwargs["repo"]
        path = kwargs["path"]
        ref = kwargs.get("ref")
        max_chars = int(kwargs.get("max_chars", 10000))
        url = f"{GITHUB_API}/repos/{repo}/contents/{path}"
        params = {"ref": ref} if ref else {}
        try:
            resp = httpx.get(url, headers=_gh_headers(), params=params, timeout=20.0)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            return {"ok": False, "summary": f"github error: {e}"}
        data = resp.json()
        if data.get("type") != "file":
            return {"ok": False, "summary": f"{path} is not a file"}
        content_b64 = data.get("content", "")
        try:
            raw = base64.b64decode(content_b64).decode("utf-8", errors="replace")
        except Exception as e:
            return {"ok": False, "summary": f"decode failed: {e}"}
        truncated = len(raw) > max_chars
        raw = raw[:max_chars]
        return {
            "ok": True,
            "summary": f"{repo}/{path} @ {ref or 'default'} — {len(raw)} chars"
                       + (" (truncated)" if truncated else ""),
            "path": path,
            "content": raw,
            "truncated": truncated,
            "sha": data.get("sha"),
        }


class GitHubRecentCommits(BaseTool):
    name = "github_recent_commits"
    description = (
        "List recent commits on a repo branch. Use to see what's changed "
        "lately — a good signal for which codebases are active."
    )
    permission_level = PermissionLevel.AUTO

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "repo": {"type": "string"},
                "branch": {"type": "string"},
                "limit": {"type": "integer", "default": 10, "maximum": 30},
            },
            "required": ["repo"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        if not _has_token():
            return {"ok": False, "summary": "no GITHUB_PAT set"}
        repo = kwargs["repo"]
        branch = kwargs.get("branch")
        limit = int(kwargs.get("limit", 10))
        params: dict[str, Any] = {"per_page": limit}
        if branch:
            params["sha"] = branch
        try:
            resp = httpx.get(f"{GITHUB_API}/repos/{repo}/commits",
                             headers=_gh_headers(), params=params, timeout=15.0)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            return {"ok": False, "summary": f"github error: {e}"}
        commits = [
            {
                "sha": c["sha"][:7],
                "author": (c.get("author") or {}).get("login") or c["commit"]["author"]["name"],
                "date": c["commit"]["author"]["date"],
                "message": c["commit"]["message"].split("\n")[0][:120],
            }
            for c in resp.json()
        ]
        return {
            "ok": True,
            "summary": f"{len(commits)} recent commits on {repo}",
            "commits": commits,
        }
