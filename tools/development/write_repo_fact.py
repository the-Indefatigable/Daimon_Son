"""write_repo_fact: capture durable architectural facts about a repo so DAIMON
doesn't re-discover them every cycle. Upserts on (repo, category, key)."""
from __future__ import annotations

from typing import Any

from core.repo_schema import RepoSchema
from permissions.levels import PermissionLevel
from tools.base import BaseTool


CATEGORY_HELP = (
    "overview = one-paragraph what-is-this-repo. "
    "stack = framework/build/deploy. "
    "flow = named user/system flow (signup, checkout, login). "
    "contract = API endpoint shape, env var, schema invariant. "
    "gotcha = non-obvious thing that bit you before. "
    "note = anything else worth keeping."
)


class WriteRepoFact(BaseTool):
    name = "write_repo_fact"
    description = (
        "Save a durable architectural fact about a repo. Upserts by "
        "(repo, category, key) — calling with the same key updates that fact. "
        "Use this aggressively after: (a) Mohammad answers a structural question "
        "via inbox, (b) you finish reading code and learn how something works, "
        "(c) a PR outcome teaches you something about the repo's invariants. "
        "Goal: the next time you touch this repo, your observations should "
        "already contain the architectural map — no re-discovery. "
        f"Categories — {CATEGORY_HELP} "
        "Keep `body` tight (1-3 sentences). One fact per call. If you have "
        "five things to write, make five calls — they're all cheap."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.0

    def __init__(self, schema: RepoSchema, get_cycle):
        self._schema = schema
        self._get_cycle = get_cycle

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "repo": {
                    "type": "string",
                    "description": "Repo name. Use the short form you already know "
                                   "from github tools — e.g. 'centsibles-frontend', "
                                   "'centsibles-backend', 'firstprinciple-blog', "
                                   "'Daimon_Son'. Be consistent across calls.",
                },
                "category": {
                    "type": "string",
                    "enum": ["overview", "stack", "flow", "contract", "gotcha", "note"],
                    "description": CATEGORY_HELP,
                },
                "key": {
                    "type": "string",
                    "description": "Short identifier for this fact within the "
                                   "category. e.g. 'signup_flow', 'checkout_endpoint', "
                                   "'ts_imports', 'price_allowlist'. Reusing a key "
                                   "OVERWRITES the previous fact — that's the upsert.",
                },
                "body": {
                    "type": "string",
                    "description": "The fact itself. 1-3 sentences. Concrete file "
                                   "paths, function names, and invariants beat "
                                   "abstractions. e.g. 'Signup: POST /api/signup → "
                                   "verify email link → redirect /dashboard. Lives "
                                   "in src/auth/SignUpPage.tsx. Email link survives "
                                   "device switch but query params don't.'",
                },
                "source": {
                    "type": "string",
                    "enum": ["mohammad_reply", "self_audit", "pr_outcome",
                             "web_search", "experiment", "other"],
                    "description": "Where this fact came from. Helps you weigh "
                                   "trust — Mohammad's answers are gold-standard.",
                },
                "confidence": {
                    "type": "number",
                    "description": "0.0-1.0. Mohammad-sourced facts: 0.9-1.0. "
                                   "Self-audit from reading code: 0.7-0.9. "
                                   "Inferred from PR outcomes: 0.5-0.8.",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
            },
            "required": ["repo", "category", "key", "body", "source", "confidence"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        result = self._schema.upsert(
            repo=str(kwargs["repo"]),
            category=str(kwargs["category"]),
            key=str(kwargs["key"]),
            body=str(kwargs["body"]),
            source=str(kwargs.get("source", "self_audit")),
            cycle=self._get_cycle(),
            confidence=float(kwargs.get("confidence", 0.7)),
        )
        if not result.get("ok"):
            return {"ok": False, "summary": result.get("reason", "write failed")}
        action = result["action"]
        return {
            "ok": True,
            "summary": (
                f"repo_fact {action}: {kwargs['repo']}/{kwargs['category']}/{kwargs['key']} "
                f"(id={result['id']})"
                + (f"; replaced: {result['previous_body']}" if action == "updated"
                   and result.get("previous_body") else "")
            ),
            "fact_id": result["id"],
        }


class ReadRepoFacts(BaseTool):
    name = "read_repo_facts"
    description = (
        "Read all architectural facts on file for a repo. Cheap — call this "
        "BEFORE reading source code if you have facts already, and especially "
        "before opening a PR against a repo you haven't touched recently. "
        "Returns facts grouped by category (overview / stack / flow / contract "
        "/ gotcha / note)."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.0

    def __init__(self, schema: RepoSchema):
        self._schema = schema

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "repo": {
                    "type": "string",
                    "description": "Repo name to read facts for.",
                },
                "categories": {
                    "type": "string",
                    "description": "Optional comma-separated filter, e.g. "
                                   "'flow,gotcha'. Empty string = all categories.",
                },
            },
            "required": ["repo", "categories"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        repo = str(kwargs["repo"]).strip()
        cat_str = str(kwargs.get("categories", "")).strip()
        cats = [c.strip().lower() for c in cat_str.split(",") if c.strip()] or None
        facts = self._schema.for_repo(repo, categories=cats)
        if not facts:
            known = self._schema.known_repos()
            return {
                "ok": True,
                "summary": (
                    f"no facts on file for {repo}. "
                    f"Known repos: {known or '(none yet — start writing)'}"
                ),
                "facts": [],
            }
        # Group by category for prompt-friendly output
        grouped: dict[str, list[dict]] = {}
        for f in facts:
            grouped.setdefault(f["category"], []).append({
                "key": f["key"],
                "body": f["body"],
                "source": f["source"],
                "confidence": round(f["confidence"], 2),
                "cycle": f["cycle"],
            })
        return {
            "ok": True,
            "summary": f"{len(facts)} fact(s) for {repo} across {list(grouped)}",
            "facts": grouped,
        }
