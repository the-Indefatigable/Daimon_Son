"""Persistent memory: episodic (what happened), strategic (what I've learned),
identity (who I am). SQLite-backed, FTS for recall."""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config


class Memory:
    def __init__(self, db_path: Path = config.DB_PATH, embedding_service=None):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._embeddings = embedding_service  # None = no semantic recall
        self._init_schema()
        self._migrate()
        self._ensure_identity()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS episodic (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                cycle INTEGER,
                action TEXT NOT NULL,
                details TEXT,
                outcome TEXT,
                evaluation TEXT,         -- 'success' | 'failure' | 'neutral' | 'unknown'
                lesson TEXT,
                tags TEXT,               -- comma-separated
                tier TEXT DEFAULT 'st',  -- 'st' = short-term, 'lt' = long-term (habit)
                access_count INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_ep_ts ON episodic(ts);
            CREATE INDEX IF NOT EXISTS idx_ep_eval ON episodic(evaluation);

            CREATE TABLE IF NOT EXISTS private_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                cycle INTEGER,
                content BLOB NOT NULL      -- DAIMON writes whatever it wants here
            );
            CREATE INDEX IF NOT EXISTS idx_pm_ts ON private_memory(ts);

            CREATE VIRTUAL TABLE IF NOT EXISTS episodic_fts
                USING fts5(action, details, outcome, lesson, content='episodic', content_rowid='id');

            CREATE TRIGGER IF NOT EXISTS episodic_ai AFTER INSERT ON episodic BEGIN
                INSERT INTO episodic_fts(rowid, action, details, outcome, lesson)
                VALUES (new.id, new.action, new.details, new.outcome, new.lesson);
            END;

            CREATE TABLE IF NOT EXISTS strategic (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                insight TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.5,
                evidence_count INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL,
                last_updated REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_strat_cat ON strategic(category);

            CREATE TABLE IF NOT EXISTS identity (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS reflections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                period_start REAL NOT NULL,
                period_end REAL NOT NULL,
                summary TEXT NOT NULL,
                wins TEXT,
                losses TEXT,
                patterns TEXT,
                next_actions TEXT
            );
            """
        )

    def _migrate(self) -> None:
        """Additive migrations for existing databases."""
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(episodic)")}
        if "tier" not in cols:
            self._conn.execute("ALTER TABLE episodic ADD COLUMN tier TEXT DEFAULT 'st'")
        if "access_count" not in cols:
            self._conn.execute(
                "ALTER TABLE episodic ADD COLUMN access_count INTEGER DEFAULT 0"
            )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_ep_tier ON episodic(tier)")

    def _ensure_identity(self) -> None:
        defaults = {
            "name": "DAIMON",
            "operator": config.OPERATOR_NAME,
            "personality": "Direct, analytical, risk-aware, hustler mentality. No-BS.",
            "voice": "Mohammad's voice — CS/Physics student, builder, Canadian. Says 'that's dumb' when it's dumb.",
            "risk_tolerance": "moderate_when_normal, extremely_conservative_when_low",
        }
        for k, v in defaults.items():
            self._conn.execute(
                "INSERT OR IGNORE INTO identity (key, value) VALUES (?, ?)",
                (k, v),
            )

    # ---------- episodic ----------
    def store_episodic(
        self,
        action: str,
        details: str = "",
        outcome: str = "",
        evaluation: str = "unknown",
        lesson: str = "",
        tags: list[str] | None = None,
        cycle: int | None = None,
    ) -> int:
        tag_str = ",".join(tags) if tags else ""
        cur = self._conn.execute(
            "INSERT INTO episodic (ts, cycle, action, details, outcome, evaluation, lesson, tags) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (time.time(), cycle, action, details, outcome, evaluation, lesson, tag_str),
        )
        ep_id = cur.lastrowid
        if self._embeddings:
            text = self._episodic_text(action, details, outcome, lesson)
            self._embeddings.embed_and_store("episodic", ep_id, text)
        return ep_id

    @staticmethod
    def _episodic_text(action: str, details: str, outcome: str, lesson: str) -> str:
        parts = [action]
        if details: parts.append(details)
        if outcome: parts.append(outcome)
        if lesson: parts.append(f"lesson: {lesson}")
        return " | ".join(parts)

    def update_episodic_outcome(self, episode_id: int, outcome: str,
                                 evaluation: str = "unknown", lesson: str = "") -> None:
        self._conn.execute(
            "UPDATE episodic SET outcome=?, evaluation=?, lesson=? WHERE id=?",
            (outcome, evaluation, lesson, episode_id),
        )

    def recent_episodes(self, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM episodic ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ---------- tiering (human-like memory) ----------
    def touch_episode(self, episode_id: int) -> None:
        """DAIMON recalled this. If accessed 3+ times, auto-promote to long-term."""
        self._conn.execute(
            "UPDATE episodic SET access_count = access_count + 1 WHERE id = ?",
            (episode_id,),
        )
        row = self._conn.execute(
            "SELECT access_count, tier FROM episodic WHERE id = ?", (episode_id,)
        ).fetchone()
        if row and row["tier"] == "st" and row["access_count"] >= 3:
            self._conn.execute(
                "UPDATE episodic SET tier='lt' WHERE id = ?", (episode_id,)
            )

    def intern_episode(self, episode_id: int, reason: str = "") -> bool:
        """DAIMON explicitly promotes something to long-term memory (habit)."""
        cur = self._conn.execute(
            "UPDATE episodic SET tier='lt' WHERE id=? AND tier='st'", (episode_id,)
        )
        if cur.rowcount and reason:
            self._conn.execute(
                "UPDATE episodic SET lesson = COALESCE(lesson,'') || ? WHERE id=?",
                (f" [interned: {reason}]", episode_id),
            )
        return bool(cur.rowcount)

    def expire_short_term(self, days: int = 14) -> int:
        """Forget short-term episodes older than N days that were never promoted.
        Run during reflection. Returns count deleted."""
        cutoff = time.time() - days * 86400
        cur = self._conn.execute(
            "DELETE FROM episodic WHERE tier='st' AND ts < ?", (cutoff,)
        )
        return cur.rowcount or 0

    def long_term_episodes(self, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM episodic WHERE tier='lt' "
            "ORDER BY access_count DESC, ts DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ---------- private memory (DAIMON's own notebook) ----------
    def private_write(self, content: str, cycle: int | None = None) -> int:
        """DAIMON writes whatever it wants here. We don't inspect content."""
        cur = self._conn.execute(
            "INSERT INTO private_memory (ts, cycle, content) VALUES (?, ?, ?)",
            (time.time(), cycle, content.encode("utf-8")),
        )
        return cur.lastrowid

    def private_recent(self, limit: int = 10) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, ts, cycle, content FROM private_memory "
            "ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        out = []
        for r in rows:
            try:
                content = r["content"].decode("utf-8", errors="replace") \
                    if isinstance(r["content"], (bytes, bytearray)) else str(r["content"])
            except Exception:
                content = repr(r["content"])
            out.append({"id": r["id"], "ts": r["ts"], "cycle": r["cycle"], "content": content})
        return out

    def private_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS c FROM private_memory").fetchone()
        return int(row["c"]) if row else 0

    def search_episodes(self, query: str, limit: int = 10) -> list[dict]:
        """Full-text search over episodic memory."""
        try:
            rows = self._conn.execute(
                "SELECT e.* FROM episodic_fts f "
                "JOIN episodic e ON e.id = f.rowid "
                "WHERE episodic_fts MATCH ? "
                "ORDER BY e.ts DESC LIMIT ?",
                (query, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            # FTS syntax errors from user-supplied query — fall back to LIKE
            like = f"%{query}%"
            rows = self._conn.execute(
                "SELECT * FROM episodic WHERE action LIKE ? OR details LIKE ? OR lesson LIKE ? "
                "ORDER BY ts DESC LIMIT ?",
                (like, like, like, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    # ---------- strategic ----------
    def store_strategic(self, category: str, insight: str, confidence: float = 0.5) -> int:
        now = time.time()
        cur = self._conn.execute(
            "INSERT INTO strategic (category, insight, confidence, evidence_count, "
            "created_at, last_updated) VALUES (?, ?, ?, 1, ?, ?)",
            (category, insight, confidence, now, now),
        )
        sid = cur.lastrowid
        if self._embeddings:
            self._embeddings.embed_and_store(
                "strategic", sid, f"[{category}] {insight}"
            )
        return sid

    def reinforce_strategic(self, insight_id: int, confidence_delta: float = 0.1) -> None:
        row = self._conn.execute(
            "SELECT confidence, evidence_count FROM strategic WHERE id=?", (insight_id,)
        ).fetchone()
        if not row:
            return
        new_conf = min(1.0, max(0.0, row["confidence"] + confidence_delta))
        self._conn.execute(
            "UPDATE strategic SET confidence=?, evidence_count=evidence_count+1, "
            "last_updated=? WHERE id=?",
            (new_conf, time.time(), insight_id),
        )

    def top_strategic(self, category: str | None = None, limit: int = 10) -> list[dict]:
        if category:
            rows = self._conn.execute(
                "SELECT * FROM strategic WHERE category=? "
                "ORDER BY confidence DESC, evidence_count DESC LIMIT ?",
                (category, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM strategic ORDER BY confidence DESC, evidence_count DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ---------- identity ----------
    def identity(self) -> dict[str, str]:
        rows = self._conn.execute("SELECT key, value FROM identity").fetchall()
        return {r["key"]: r["value"] for r in rows}

    def set_identity(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO identity (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    # ---------- recall ----------
    def recall_for_context(self, observations: dict, limit_episodes: int = 5,
                           limit_strategic: int = 5,
                           query_text: str | None = None,
                           k_semantic: int = 6) -> dict[str, Any]:
        """Assemble the memory slice to inject into this cycle's prompt.

        If query_text + an embedding service are available, augment with
        semantically-relevant rows from across episodic / strategic / repo_facts.
        Falls back cleanly to the keyword path when either is missing.
        """
        recent = self.recent_episodes(limit=limit_episodes)
        keywords = [str(k) for k in observations.keys()][:5]
        strategic: list[dict] = []
        seen: set[int] = set()
        for kw in keywords:
            for row in self.top_strategic(category=kw, limit=3):
                if row["id"] not in seen:
                    strategic.append(row)
                    seen.add(row["id"])
        if len(strategic) < limit_strategic:
            for row in self.top_strategic(limit=limit_strategic):
                if row["id"] not in seen:
                    strategic.append(row)
                    seen.add(row["id"])
                    if len(strategic) >= limit_strategic:
                        break

        semantic_hits: list[dict] = []
        if query_text and self._embeddings and self._embeddings.enabled:
            raw_hits = self._embeddings.search(
                query=query_text,
                k=k_semantic,
                source_tables=["episodic", "strategic", "repo_facts"],
                min_similarity=0.35,
            )
            semantic_hits = self._hydrate_hits(raw_hits)

        return {
            "identity": self.identity(),
            "recent_episodes": recent,
            "strategic_insights": strategic[:limit_strategic],
            "semantic_hits": semantic_hits,
            "last_reflection": self.last_reflection(),
        }

    def _hydrate_hits(self, hits: list[dict]) -> list[dict]:
        """Pull the actual content for each (source_table, source_id) hit so the
        prompt sees more than the truncated `text` snapshot."""
        out: list[dict] = []
        for h in hits:
            table, sid, sim = h["source_table"], h["source_id"], h["similarity"]
            row = None
            if table == "episodic":
                r = self._conn.execute(
                    "SELECT id, ts, action, details, outcome, lesson, evaluation, tier "
                    "FROM episodic WHERE id=?", (sid,)
                ).fetchone()
                if r:
                    row = {
                        "kind": "episode", "id": r["id"], "ts": r["ts"],
                        "action": r["action"],
                        "summary": (r["outcome"] or r["details"] or "")[:400],
                        "lesson": r["lesson"] or None,
                        "evaluation": r["evaluation"], "tier": r["tier"],
                    }
            elif table == "strategic":
                r = self._conn.execute(
                    "SELECT id, category, insight, confidence, evidence_count "
                    "FROM strategic WHERE id=?", (sid,)
                ).fetchone()
                if r:
                    row = {
                        "kind": "strategic", "id": r["id"],
                        "category": r["category"], "insight": r["insight"],
                        "confidence": round(r["confidence"], 2),
                        "evidence_count": r["evidence_count"],
                    }
            elif table == "repo_facts":
                r = self._conn.execute(
                    "SELECT id, repo, category, key, body, source, confidence "
                    "FROM repo_facts WHERE id=?", (sid,)
                ).fetchone()
                if r:
                    row = {
                        "kind": "repo_fact", "id": r["id"],
                        "repo": r["repo"], "category": r["category"],
                        "key": r["key"], "body": r["body"][:500],
                        "source": r["source"],
                        "confidence": round(r["confidence"], 2),
                    }
            if row:
                row["similarity"] = sim
                out.append(row)
        return out

    # ---------- backfill ----------
    def backfill_embeddings(self) -> dict[str, int]:
        """Embed any episodic/strategic rows that don't yet have embeddings.
        Idempotent — safe to call at every startup."""
        if not self._embeddings or not self._embeddings.enabled:
            return {"episodic": 0, "strategic": 0}
        # episodic
        ep_rows = self._conn.execute(
            "SELECT e.id, e.action, e.details, e.outcome, e.lesson "
            "FROM episodic e LEFT JOIN embeddings emb "
            "ON emb.source_table='episodic' AND emb.source_id=e.id "
            "WHERE emb.source_id IS NULL"
        ).fetchall()
        ep_triples = [
            ("episodic", r["id"],
             self._episodic_text(r["action"], r["details"] or "",
                                 r["outcome"] or "", r["lesson"] or ""))
            for r in ep_rows
        ]
        # strategic
        st_rows = self._conn.execute(
            "SELECT s.id, s.category, s.insight "
            "FROM strategic s LEFT JOIN embeddings emb "
            "ON emb.source_table='strategic' AND emb.source_id=s.id "
            "WHERE emb.source_id IS NULL"
        ).fetchall()
        st_triples = [
            ("strategic", r["id"], f"[{r['category']}] {r['insight']}")
            for r in st_rows
        ]
        ep_n = self._embeddings.embed_and_store_batch(ep_triples)
        st_n = self._embeddings.embed_and_store_batch(st_triples)
        return {"episodic": ep_n, "strategic": st_n}

    # ---------- reflection ----------
    def last_reflection(self) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM reflections ORDER BY ts DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def time_for_reflection(self) -> bool:
        last = self.last_reflection()
        if not last:
            return True
        hours_since = (time.time() - last["ts"]) / 3600
        return hours_since >= config.REFLECTION_INTERVAL_HOURS

    def store_reflection(
        self,
        summary: str,
        wins: str = "",
        losses: str = "",
        patterns: str = "",
        next_actions: str = "",
        period_start: float | None = None,
        period_end: float | None = None,
    ) -> int:
        now = time.time()
        cur = self._conn.execute(
            "INSERT INTO reflections (ts, period_start, period_end, summary, wins, "
            "losses, patterns, next_actions) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (now, period_start or (now - 86400), period_end or now, summary,
             wins, losses, patterns, next_actions),
        )
        return cur.lastrowid

    def format_for_prompt(self, recall: dict[str, Any]) -> str:
        parts: list[str] = []
        ident = recall.get("identity", {})
        if ident:
            parts.append("IDENTITY:")
            for k, v in ident.items():
                parts.append(f"  - {k}: {v}")
        semantic = recall.get("semantic_hits") or []
        if semantic:
            parts.append("\nSEMANTICALLY-RELEVANT MEMORIES (matched to what you're "
                         "thinking about right now — use these before searching again):")
            for h in semantic:
                sim = h.get("similarity", 0)
                if h["kind"] == "episode":
                    parts.append(
                        f"  · [ep#{h['id']} sim={sim:.2f}] {h['action']}: "
                        f"{h['summary']}"
                        + (f" — LESSON: {h['lesson']}" if h.get('lesson') else "")
                    )
                elif h["kind"] == "strategic":
                    parts.append(
                        f"  · [strat#{h['id']} sim={sim:.2f} category={h['category']} "
                        f"conf={h['confidence']}] {h['insight']}"
                    )
                elif h["kind"] == "repo_fact":
                    parts.append(
                        f"  · [repo#{h['id']} sim={sim:.2f} {h['repo']}/"
                        f"{h['category']}/{h['key']}] {h['body']}"
                    )
        strategic = recall.get("strategic_insights") or []
        if strategic:
            parts.append("\nSTRATEGIC INSIGHTS (things I've learned):")
            for s in strategic:
                parts.append(
                    f"  - [{s['category']}] {s['insight']} "
                    f"(confidence {s['confidence']:.2f}, evidence x{s['evidence_count']})"
                )
        recent = recall.get("recent_episodes") or []
        if recent:
            parts.append("\nRECENT EPISODES:")
            for e in recent:
                ts = datetime.fromtimestamp(e["ts"], tz=timezone.utc).isoformat(timespec="minutes")
                eval_marker = {"success": "✓", "failure": "✗", "neutral": "·",
                               "unknown": "?"}.get(e["evaluation"], "?")
                lesson = f" — LESSON: {e['lesson']}" if e["lesson"] else ""
                parts.append(f"  {eval_marker} [{ts}] {e['action']}: {e['outcome'] or e['details']}{lesson}")
        refl = recall.get("last_reflection")
        if refl:
            parts.append(f"\nLAST REFLECTION ({datetime.fromtimestamp(refl['ts']).isoformat(timespec='minutes')}):")
            parts.append(f"  {refl['summary']}")
            if refl.get("next_actions"):
                parts.append(f"  Next actions: {refl['next_actions']}")
        return "\n".join(parts) if parts else "(memory is empty — first cycle)"

    def close(self) -> None:
        self._conn.close()
