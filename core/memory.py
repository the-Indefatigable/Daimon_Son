"""Persistent memory: episodic (what happened), strategic (what I've learned),
identity (who I am). SQLite-backed, FTS for recall.

Human-style fragment recall
---------------------------
On top of raw episodic rows, every episode carries a compact *fragment view*:
  - gist         — 1-2 sentences in DAIMON's own voice (derived from outcome)
  - key_facts    — 3-5 bullet fragments
  - surprise     — Bayesian-ish novelty score (0..1) via embedding distance
  - decay_factor — ACT-R-style forgetting multiplier (0..1), ticks down per cycle

`recall_fragments(query, max_tokens=1200)` returns 3-6 ranked fragments instead of
a big episodic dump. Ranking combines semantic similarity with surprise (novel
stuff is more memorable), access_count (habits stay live), and decay (old stuff
fades). Output format is written in a feral self-note register so Claude mirrors
the tone DAIMON should be reproducing, not polite English.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config


# Rough token estimate: 1 token ~= 4 chars of English. Cheap and conservative
# enough for sparse-fragment budgeting.
def _approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)


GIST_MAX_CHARS = 320              # ~80 tokens target
KEY_FACT_MAX_CHARS = 140          # one-liner
KEY_FACTS_MAX = 5
DECAY_PER_CYCLE = 0.95            # ACT-R-inspired forgetting curve
DECAY_FLOOR = 0.05                # below this, treat as forgotten
SURPRISE_NOVELTY_WINDOW = 20      # compare new embedding against last N centroids
DEFAULT_RECALL_K = 6
DEFAULT_RECALL_MAX_TOKENS = 1200


@dataclass
class MemoryFragment:
    """Sparse, human-readable episodic trace. Built from the `episodic` row plus
    fragment-view columns. Never persisted standalone — the `episodic` table IS
    the storage; this is the lens."""
    id: int
    ts: float
    event_type: str
    gist: str
    key_facts: list[str]
    surprise_score: float
    decay_factor: float
    tags: list[str] = field(default_factory=list)
    tier: str = "st"
    access_count: int = 0
    evaluation: str = "unknown"

    def age_human(self, now: float | None = None) -> str:
        now = now or time.time()
        delta = max(0.0, now - self.ts)
        if delta < 3600:
            return f"{int(delta / 60)}m ago"
        if delta < 86400:
            return f"{int(delta / 3600)}h ago"
        return f"{int(delta / 86400)}d ago"

    def to_prompt_block(self, now: float | None = None) -> str:
        # Only surface an eval marker when it's a real signal — skip
        # neutral/unknown so the head line stays clean.
        marker = {"success": "HIT", "failure": "MISS"}.get(self.evaluation)
        parts = [f"ep#{self.id}", self.age_human(now)]
        if marker:
            parts.append(marker)
        parts.append(self.event_type)
        if self.tier == "lt":
            parts.append("LT")
        if self.access_count >= 2:
            parts.append(f"×{self.access_count}")
        head = " · ".join(parts)
        lines = [head]
        gist = (self.gist or "").strip()
        if gist:
            lines.append(f'  "{gist}"')
        for fact in self.key_facts[:KEY_FACTS_MAX]:
            f = fact.strip()
            if f:
                lines.append(f"  - {f}")
        return "\n".join(lines)

    def token_cost(self) -> int:
        return _approx_tokens(self.to_prompt_block())


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
        # Sparse-fragment columns — added 2026-04-18
        if "gist" not in cols:
            self._conn.execute("ALTER TABLE episodic ADD COLUMN gist TEXT")
        if "key_facts" not in cols:
            self._conn.execute("ALTER TABLE episodic ADD COLUMN key_facts TEXT")
        if "decay_factor" not in cols:
            self._conn.execute(
                "ALTER TABLE episodic ADD COLUMN decay_factor REAL DEFAULT 1.0"
            )
        if "surprise_score" not in cols:
            self._conn.execute(
                "ALTER TABLE episodic ADD COLUMN surprise_score REAL DEFAULT 0.5"
            )
        if "event_type" not in cols:
            self._conn.execute("ALTER TABLE episodic ADD COLUMN event_type TEXT")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_ep_tier ON episodic(tier)")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ep_decay ON episodic(decay_factor)"
        )

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
        gist: str | None = None,
        key_facts: list[str] | None = None,
        event_type: str | None = None,
    ) -> int:
        tag_str = ",".join(tags) if tags else ""
        etype = event_type or self._infer_event_type(action, tags or [])
        gist_text = gist if gist is not None else self._compute_gist(
            action=action, outcome=outcome, details=details, lesson=lesson
        )
        facts = key_facts if key_facts is not None else self._compute_key_facts(
            action=action, outcome=outcome, details=details, lesson=lesson
        )
        facts_json = json.dumps(facts[:KEY_FACTS_MAX])

        # Compute surprise BEFORE insert so we have the value ready; uses the
        # embedding of this row compared against the recent centroid.
        text = self._episodic_text(action, details, outcome, lesson)
        surprise = self._compute_surprise(text)

        cur = self._conn.execute(
            "INSERT INTO episodic "
            "(ts, cycle, action, details, outcome, evaluation, lesson, tags, "
            " gist, key_facts, decay_factor, surprise_score, event_type) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1.0, ?, ?)",
            (time.time(), cycle, action, details, outcome, evaluation, lesson,
             tag_str, gist_text, facts_json, surprise, etype),
        )
        ep_id = cur.lastrowid
        if self._embeddings:
            self._embeddings.embed_and_store("episodic", ep_id, text)
        return ep_id

    # ---------- fragment-view helpers ----------
    @staticmethod
    def _infer_event_type(action: str, tags: list[str]) -> str:
        a = (action or "").lower()
        tagset = {str(t).lower() for t in tags}
        if a.startswith("cycle_") or "cycle" in tagset:
            return "cycle"
        if "backrooms" in tagset or a.startswith("backrooms"):
            return "backrooms"
        if "grok_post" in tagset or "llama_post" in tagset or "bluesky_post" in tagset:
            return "post"
        if "read_inbox" in tagset or "mohammad" in tagset or a == "mohammad_reply":
            return "mohammad_reply"
        if "tier_change" in tagset or a in ("brain_upgrade", "brain_downgrade"):
            return "tier_change"
        if a == "death":
            return "death"
        return "event"

    @staticmethod
    def _compute_gist(action: str, outcome: str, details: str, lesson: str) -> str:
        """Deterministic gist extraction — first sentence of outcome (DAIMON's own
        voice) else first chunk of details. Feral register comes from the source
        text; we don't rewrite."""
        source = outcome or details or action or ""
        # Take the first sentence-ish segment.
        cut = re.split(r"(?<=[.!?])\s+", source.strip(), maxsplit=1)
        head = cut[0] if cut else source
        head = head.strip()
        if len(head) > GIST_MAX_CHARS:
            head = head[:GIST_MAX_CHARS - 1].rstrip() + "…"
        if lesson and len(head) + len(lesson) + 4 <= GIST_MAX_CHARS:
            head = f"{head} // {lesson.strip()}" if head else lesson.strip()
        return head

    @staticmethod
    def _compute_key_facts(action: str, outcome: str,
                           details: str, lesson: str) -> list[str]:
        """Atomic fragments in DAIMON's own voice when possible.

        Priority: outcome sentences after the first (that's DAIMON actually
        speaking) → lesson (the distilled takeaway) → details (mostly
        metadata — boilerplate, skipped when outcome is rich).
        """
        facts: list[str] = []

        def _push(chunk: str) -> None:
            chunk = chunk.strip(" -·•\t")
            if not chunk:
                return
            if len(chunk) > KEY_FACT_MAX_CHARS:
                chunk = chunk[:KEY_FACT_MAX_CHARS - 1].rstrip() + "…"
            if chunk and chunk not in facts:
                facts.append(chunk)

        # Outcome tail — everything after the first sentence (first went to gist)
        outcome = (outcome or "").strip()
        tail_sents: list[str] = []
        if outcome:
            split = re.split(r"(?<=[.!?])\s+", outcome)
            if len(split) > 1:
                tail_sents = split[1:]
        for sent in tail_sents:
            if len(facts) >= KEY_FACTS_MAX:
                break
            for raw in re.split(r"[;\n]+", sent):
                if len(facts) >= KEY_FACTS_MAX:
                    break
                _push(raw)

        if lesson and len(facts) < KEY_FACTS_MAX:
            _push(f"lesson: {lesson}")

        # Details only as a last resort — it's usually metadata boilerplate
        # (observations=[...], tools_used=[...], model=foo).
        if len(facts) < 2 and details:
            for raw in re.split(r"[;\n]+", details):
                if len(facts) >= KEY_FACTS_MAX:
                    break
                _push(raw)

        return facts[:KEY_FACTS_MAX]

    def _compute_surprise(self, text: str) -> float:
        """Novelty-of-event score. High = dissimilar to recent memory = memorable.
        Falls back to 0.5 when embeddings disabled or no prior history."""
        if not self._embeddings or not getattr(self._embeddings, "enabled", False):
            return 0.5
        if not text or not text.strip():
            return 0.5
        try:
            import numpy as np  # type: ignore
        except ImportError:
            return 0.5
        # Pull recent episodic vectors from the embeddings table via the shared db.
        rows = self._conn.execute(
            "SELECT vector, dim FROM embeddings "
            "WHERE source_table='episodic' "
            "ORDER BY created_ts DESC LIMIT ?",
            (SURPRISE_NOVELTY_WINDOW,),
        ).fetchall()
        if not rows:
            return 0.8  # first memories are maximally novel
        try:
            new_vec = self._embeddings._embed_call([text], input_type="document")
        except Exception:
            return 0.5
        if not new_vec:
            return 0.5
        q = np.asarray(new_vec[0], dtype=np.float32)
        mat = np.vstack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
        qnorm = float(np.linalg.norm(q)) or 1e-9
        mnorms = np.linalg.norm(mat, axis=1)
        mnorms[mnorms == 0] = 1e-9
        sims = (mat @ q) / (mnorms * qnorm)
        # Novelty = 1 - max similarity to any recent memory. Clamp to [0,1].
        novelty = float(max(0.0, min(1.0, 1.0 - float(sims.max()))))
        return round(novelty, 4)

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
        """DAIMON recalled this. Bump access, reset decay (just-accessed = live),
        auto-promote to long-term at the 3-recall habit threshold."""
        self._conn.execute(
            "UPDATE episodic SET access_count = access_count + 1, "
            "decay_factor = MIN(1.0, decay_factor + 0.3) WHERE id = ?",
            (episode_id,),
        )
        row = self._conn.execute(
            "SELECT access_count, tier FROM episodic WHERE id = ?", (episode_id,)
        ).fetchone()
        if row and row["tier"] == "st" and row["access_count"] >= 3:
            self._conn.execute(
                "UPDATE episodic SET tier='lt', decay_factor=1.0 WHERE id = ?",
                (episode_id,),
            )

    # ---------- natural forgetting (ACT-R-style decay) ----------
    def decay_step(self, factor: float = DECAY_PER_CYCLE) -> int:
        """Apply one cycle of forgetting to short-term memories. Long-term is
        preserved. Returns rows affected."""
        cur = self._conn.execute(
            "UPDATE episodic SET decay_factor = MAX(?, decay_factor * ?) "
            "WHERE tier='st'",
            (DECAY_FLOOR, factor),
        )
        return cur.rowcount or 0

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

    # ---------- sparse fragment recall (human-style) ----------
    def _row_to_fragment(self, row: sqlite3.Row | dict) -> MemoryFragment:
        """Hydrate a fragment from an episodic row. Computes gist/key_facts on
        the fly for pre-fragment rows that have NULL in those columns."""
        r = dict(row)
        gist = r.get("gist") or self._compute_gist(
            r.get("action") or "", r.get("outcome") or "",
            r.get("details") or "", r.get("lesson") or "",
        )
        key_facts_raw = r.get("key_facts")
        if key_facts_raw:
            try:
                facts = list(json.loads(key_facts_raw))
            except Exception:
                facts = [str(key_facts_raw)[:KEY_FACT_MAX_CHARS]]
        else:
            facts = self._compute_key_facts(
                r.get("action") or "", r.get("outcome") or "",
                r.get("details") or "", r.get("lesson") or "",
            )
        tags_raw = r.get("tags") or ""
        tags = [t for t in tags_raw.split(",") if t] if isinstance(tags_raw, str) else []
        event_type = r.get("event_type") or self._infer_event_type(
            r.get("action") or "", tags,
        )
        return MemoryFragment(
            id=int(r["id"]),
            ts=float(r["ts"]),
            event_type=event_type,
            gist=gist,
            key_facts=[str(f) for f in facts],
            surprise_score=float(r.get("surprise_score") or 0.5),
            decay_factor=float(r.get("decay_factor") or 1.0),
            tags=tags,
            tier=str(r.get("tier") or "st"),
            access_count=int(r.get("access_count") or 0),
            evaluation=str(r.get("evaluation") or "unknown"),
        )

    def recall_fragments(
        self,
        query: str = "",
        k: int = DEFAULT_RECALL_K,
        max_tokens: int = DEFAULT_RECALL_MAX_TOKENS,
        tag: str | None = None,
        include_long_term: bool = True,
        touch: bool = True,
    ) -> list[MemoryFragment]:
        """Return up to `k` sparse fragments, capped at `max_tokens` total.

        Ranking = semantic_similarity × decay_factor × surprise_boost × habit_boost.
        When embeddings are disabled or query is empty, falls back to a
        recency+decay ranking over short-term rows.

        `touch=True` bumps access_count + refreshes decay on every returned
        fragment — recalling IS a form of reinforcement. Pass touch=False for
        read-only introspection.
        """
        candidates: dict[int, tuple[float, sqlite3.Row]] = {}

        # Semantic pool: Voyage-embedding similarity.
        use_semantic = (
            bool(query) and self._embeddings
            and getattr(self._embeddings, "enabled", False)
        )
        if use_semantic:
            hits = self._embeddings.search(
                query=query,
                k=max(12, k * 4),
                source_tables=["episodic"],
                min_similarity=0.30,
            )
            if hits:
                ids = [int(h["source_id"]) for h in hits]
                sim_by_id = {int(h["source_id"]): float(h["similarity"]) for h in hits}
                placeholders = ",".join("?" for _ in ids)
                rows = self._conn.execute(
                    f"SELECT * FROM episodic WHERE id IN ({placeholders})",
                    ids,
                ).fetchall()
                for r in rows:
                    candidates[r["id"]] = (sim_by_id.get(r["id"], 0.0), r)

        # Recency pool: a few latest rows always present so DAIMON isn't blind to
        # what just happened.
        recency_rows = self._conn.execute(
            "SELECT * FROM episodic "
            "WHERE (? = '' OR tags LIKE ?) "
            "ORDER BY ts DESC LIMIT ?",
            (tag or "", f"%{tag}%" if tag else "", max(6, k)),
        ).fetchall()
        for r in recency_rows:
            if r["id"] not in candidates:
                # Seed with a modest similarity so recency can compete.
                candidates[r["id"]] = (0.35, r)

        # Long-term seed: carry a sample of the most-accessed LT rows so habits
        # stay visible when relevant.
        if include_long_term:
            lt_rows = self._conn.execute(
                "SELECT * FROM episodic WHERE tier='lt' "
                "ORDER BY access_count DESC, ts DESC LIMIT ?",
                (max(3, k // 2),),
            ).fetchall()
            for r in lt_rows:
                if r["id"] not in candidates:
                    candidates[r["id"]] = (0.30, r)

        if not candidates:
            return []

        # Rank: similarity × decay × (1 + 0.3·surprise) × (1 + 0.08·log(1+access))
        import math
        scored: list[tuple[float, sqlite3.Row]] = []
        now = time.time()
        for rid, (sim, row) in candidates.items():
            decay = float(row["decay_factor"] or 1.0)
            if decay < DECAY_FLOOR and row["tier"] != "lt":
                continue  # forgotten
            surprise = float(row["surprise_score"] or 0.5)
            access = int(row["access_count"] or 0)
            # Mild recency lift on top of decay so fresh events aren't ignored.
            age_days = max(0.0, (now - float(row["ts"])) / 86400.0)
            recency_lift = math.exp(-age_days / 30.0) * 0.15
            score = (sim * decay
                     * (1.0 + 0.30 * surprise)
                     * (1.0 + 0.08 * math.log1p(access))
                     + recency_lift)
            scored.append((score, row))

        scored.sort(key=lambda t: t[0], reverse=True)

        # Token-budgeted selection.
        chosen: list[MemoryFragment] = []
        used_tokens = 0
        for _score, row in scored:
            if len(chosen) >= k:
                break
            frag = self._row_to_fragment(row)
            cost = frag.token_cost()
            if used_tokens + cost > max_tokens and chosen:
                break
            chosen.append(frag)
            used_tokens += cost

        if touch:
            for frag in chosen:
                self.touch_episode(frag.id)

        return chosen

    def format_fragments_for_prompt(
        self,
        fragments: list[MemoryFragment],
        header: str = "FRAGMENTS (what you remember — sparse, human-style)",
    ) -> str:
        """Render fragments as feral self-notes, NOT as a database table.
        The register here matters: Claude mirrors whatever tone it sees in
        its own memory block."""
        if not fragments:
            return f"{header}:\n  (nothing strongly relevant — you're operating on instinct this cycle)"
        now = time.time()
        blocks = [f.to_prompt_block(now=now) for f in fragments]
        body = "\n---\n".join(blocks)
        footer = (f"(recall_fragments for more — these were picked by relevance × "
                  f"novelty × habit-strength × decay)")
        return f"{header}:\n{body}\n{footer}"

    # ---------- recall ----------
    def recall_for_context(self, observations: dict, limit_episodes: int = 5,
                           limit_strategic: int = 3,
                           query_text: str | None = None,
                           k_semantic: int = 6,
                           fragment_k: int = 3,
                           fragment_max_tokens: int = 650) -> dict[str, Any]:
        """Assemble the sparse, human-style memory slice for this cycle.

        Default is now tight: identity + ~3 highly-relevant fragments + top-3
        strategic insights + last reflection pointer. DAIMON pulls deeper recall
        itself via the `recall_fragments` tool — just like a human pausing to
        *try* to remember something instead of receiving a full transcript.
        """
        # Sparse fragments — the backbone of the new memory block.
        fragments = self.recall_fragments(
            query=query_text or "",
            k=fragment_k,
            max_tokens=fragment_max_tokens,
            touch=False,  # auto-inject shouldn't count as active recall
        )

        # Strategic insights: a small number of high-confidence ones only.
        keywords = [str(k) for k in observations.keys()][:3]
        strategic: list[dict] = []
        seen: set[int] = set()
        for kw in keywords:
            for row in self.top_strategic(category=kw, limit=2):
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

        return {
            "identity": self.identity(),
            "fragments": fragments,
            "strategic_insights": strategic[:limit_strategic],
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
        """Render the sparse memory block. Everything here mirrors the register
        Claude should reproduce — short, punchy, fragment-style, not corporate.
        Deeper recall is a TOOL CALL away, not a free dump."""
        parts: list[str] = []
        ident = recall.get("identity", {})
        if ident:
            ident_line = " · ".join(f"{k}: {v}" for k, v in ident.items()
                                    if k in ("name", "mood", "personality"))
            # Compact identity — 1 line instead of bullet list. Full self-model
            # already lives in the cacheable YOUR CURRENT SELF-MODEL block.
            parts.append(f"IDENTITY: {ident_line}" if ident_line
                         else "IDENTITY: (see self-model block above)")

        fragments = recall.get("fragments") or []
        parts.append("")
        parts.append(self.format_fragments_for_prompt(fragments))

        strategic = recall.get("strategic_insights") or []
        if strategic:
            parts.append("\nLIVE HEURISTICS (things I trust enough to bet on):")
            for s in strategic[:3]:
                conf = s.get("confidence", 0.0) or 0.0
                parts.append(
                    f"  · {s['insight']} "
                    f"[{s['category']} · conf {conf:.2f} · x{s['evidence_count']}]"
                )

        refl = recall.get("last_reflection")
        if refl:
            when = datetime.fromtimestamp(
                refl['ts'], tz=timezone.utc
            ).isoformat(timespec='minutes')
            parts.append(f"\nLAST REFLECTION ({when}):")
            summary = (refl.get("summary") or "").strip()
            # Reflections are already DAIMON's voice — take just the first line
            # to keep the block sparse; tool-call for the full text if needed.
            first_line = summary.split("\n", 1)[0][:280]
            if first_line:
                parts.append(f"  {first_line}")
            nxt = (refl.get("next_actions") or "").strip()
            if nxt:
                parts.append(f"  next: {nxt[:220]}")

        return "\n".join(parts) if parts else "(memory is empty — first cycle)"

    def close(self) -> None:
        self._conn.close()
