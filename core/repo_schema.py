"""RepoSchema: structured architectural memory per repo.

Solves "DAIMON walks into the codebase fresh every Tuesday" — instead of
re-discovering the signup flow each time, it accumulates atomic facts
keyed by (repo, category, key). Lives forever, upserts on key.

Each fact is small and atomic:
  repo='centsibles-frontend', category='flow', key='signup',
  body='POST /signup → email verify → redirect /dashboard. SignUpPage.tsx.'

Categories (loose convention, not enforced):
  overview  — one-paragraph what-is-this-repo
  stack     — frameworks, build tools, deploy target
  flow      — named user/system flow (signup, checkout, login)
  contract  — API endpoint shape, env var, schema
  gotcha    — non-obvious thing that bit us before
  note      — anything else worth keeping
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

from . import config


class RepoSchema:
    def __init__(self, db_path: Path = config.DB_PATH):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS repo_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                repo TEXT NOT NULL,
                category TEXT NOT NULL,
                key TEXT NOT NULL,
                body TEXT NOT NULL,
                source TEXT,                  -- 'mohammad_reply' | 'self_audit' | 'pr_outcome' | etc.
                cycle INTEGER,
                created_ts REAL NOT NULL,
                updated_ts REAL NOT NULL,
                confidence REAL DEFAULT 0.7,
                UNIQUE(repo, category, key)
            );
            CREATE INDEX IF NOT EXISTS idx_rf_repo ON repo_facts(repo);
            CREATE INDEX IF NOT EXISTS idx_rf_repo_cat ON repo_facts(repo, category);
            """
        )

    # ---------- write ----------
    def upsert(
        self,
        *,
        repo: str,
        category: str,
        key: str,
        body: str,
        source: str | None = None,
        cycle: int | None = None,
        confidence: float = 0.7,
    ) -> dict[str, Any]:
        repo = repo.strip()
        category = category.strip().lower()
        key = key.strip()
        body = body.strip()
        if not (repo and category and key and body):
            return {"ok": False, "reason": "repo, category, key, body all required"}
        confidence = max(0.0, min(1.0, float(confidence)))
        now = time.time()
        existing = self._conn.execute(
            "SELECT id, body FROM repo_facts WHERE repo=? AND category=? AND key=?",
            (repo, category, key),
        ).fetchone()
        if existing:
            self._conn.execute(
                "UPDATE repo_facts SET body=?, source=?, cycle=?, updated_ts=?, "
                "confidence=? WHERE id=?",
                (body, source, cycle, now, confidence, existing["id"]),
            )
            return {"ok": True, "action": "updated", "id": existing["id"],
                    "previous_body": existing["body"][:200]}
        cur = self._conn.execute(
            "INSERT INTO repo_facts "
            "(repo, category, key, body, source, cycle, created_ts, updated_ts, confidence) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (repo, category, key, body, source, cycle, now, now, confidence),
        )
        return {"ok": True, "action": "created", "id": cur.lastrowid}

    def delete(self, fact_id: int) -> bool:
        cur = self._conn.execute("DELETE FROM repo_facts WHERE id=?", (fact_id,))
        return bool(cur.rowcount)

    # ---------- read ----------
    def for_repo(self, repo: str, categories: list[str] | None = None) -> list[dict]:
        if categories:
            placeholders = ",".join("?" for _ in categories)
            rows = self._conn.execute(
                f"SELECT * FROM repo_facts WHERE repo=? AND category IN ({placeholders}) "
                "ORDER BY category, key",
                (repo, *categories),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM repo_facts WHERE repo=? ORDER BY category, key",
                (repo,),
            ).fetchall()
        return [dict(r) for r in rows]

    def known_repos(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT repo FROM repo_facts ORDER BY repo"
        ).fetchall()
        return [r["repo"] for r in rows]

    def overview_index(self) -> dict[str, str]:
        """One-line per repo: the 'overview' fact if present, else fact count."""
        index: dict[str, str] = {}
        for repo in self.known_repos():
            ov = self._conn.execute(
                "SELECT body FROM repo_facts WHERE repo=? AND category='overview' "
                "ORDER BY updated_ts DESC LIMIT 1", (repo,)
            ).fetchone()
            if ov:
                index[repo] = ov["body"][:200]
            else:
                cnt = self._conn.execute(
                    "SELECT COUNT(*) AS c FROM repo_facts WHERE repo=?", (repo,)
                ).fetchone()
                index[repo] = f"({cnt['c']} facts on file, no overview yet)"
        return index

    # ---------- prompt formatting ----------
    def snapshot_for_observations(self, focus_text: str | None = None) -> dict[str, Any]:
        """Always include the cheap repo index. If focus mentions a known repo,
        also include its full fact set (so DAIMON walks in informed)."""
        index = self.overview_index()
        active_repos: list[str] = []
        if focus_text:
            ft = focus_text.lower()
            for repo in index:
                # match on repo name OR its short form ("centsibles" matches "centsibles-frontend")
                short = repo.split("/")[-1]
                short_root = short.split("-")[0]
                if short.lower() in ft or short_root.lower() in ft:
                    active_repos.append(repo)
        deep: dict[str, list[dict]] = {}
        for repo in active_repos:
            facts = self.for_repo(repo)
            deep[repo] = [
                {"category": f["category"], "key": f["key"],
                 "body": f["body"][:400],
                 "confidence": round(f["confidence"], 2)}
                for f in facts
            ]
        result: dict[str, Any] = {
            "known_repos": index,
            "active_repo_facts": deep,
        }
        if not index:
            result["guidance"] = (
                "No repo facts on file yet. The next time you read a repo or get "
                "an architectural answer from Mohammad, write the durable bits to "
                "repo_facts via write_repo_fact. Otherwise you'll re-ask next week."
            )
        elif not deep and focus_text:
            result["guidance"] = (
                f"You have facts on {list(index.keys())} but none matched your "
                "focus. If your work touches one of these repos, call "
                "read_repo_facts(repo=...) before reading source — it's cheaper."
            )
        else:
            result["guidance"] = (
                "Use write_repo_fact to capture durable architectural facts so "
                "you don't re-discover them next week. Especially after Mohammad "
                "answers a structural question."
            )
        return result

    def close(self) -> None:
        self._conn.close()
