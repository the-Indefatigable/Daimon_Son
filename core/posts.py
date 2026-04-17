"""Posts table: every drafter -> judge -> post cycle lands here.

This is the substrate the learning loop trains on:
  - slate_json + winner_model + judge_reasoning  = training signal
  - engagement columns (likes/replies/quotes)     = reward signal
  - training_tier                                 = set by retrain jobs

Schema is permissive — NULLs until each stage fills them in:
  record_slate   -> rows with post_status='draft'
  mark_posted    -> rows with external_id + posted_ts
  update_engagement -> rows with fresh counts + last_polled_ts
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from . import config
from .drafter import Draft
from .judge import JudgeResult


VALID_STATUS = {"draft", "posted", "failed", "rejected"}
VALID_PLATFORMS = {"twitter", "bluesky"}


@dataclass
class PostRow:
    id: int
    cycle: int | None
    created_ts: float
    prompt: str
    system_prompt: str | None
    slate_json: str
    slate_quality: int | None
    winner_text: str
    winner_model: str
    winner_index: int
    judge_model: str
    judge_reasoning: str | None
    drafter_cost_usd: float
    judge_cost_usd: float
    platform: str | None
    external_id: str | None
    posted_ts: float | None
    post_status: str
    post_error: str | None
    reply_count: int
    like_count: int
    repost_count: int
    quote_count: int
    impression_count: int | None
    last_polled_ts: float | None
    training_tier: str | None

    @property
    def engagement_total(self) -> int:
        return self.reply_count + self.like_count + self.repost_count + self.quote_count


class Posts:
    def __init__(self, db_path: Path = config.DB_PATH):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle INTEGER,
                created_ts REAL NOT NULL,

                prompt TEXT NOT NULL,
                system_prompt TEXT,
                slate_json TEXT NOT NULL,
                slate_quality INTEGER,

                winner_text TEXT NOT NULL,
                winner_model TEXT NOT NULL,
                winner_index INTEGER NOT NULL,
                judge_model TEXT NOT NULL,
                judge_reasoning TEXT,

                drafter_cost_usd REAL NOT NULL DEFAULT 0,
                judge_cost_usd REAL NOT NULL DEFAULT 0,

                platform TEXT,
                external_id TEXT,
                posted_ts REAL,
                post_status TEXT NOT NULL DEFAULT 'draft',
                post_error TEXT,

                reply_count INTEGER NOT NULL DEFAULT 0,
                like_count INTEGER NOT NULL DEFAULT 0,
                repost_count INTEGER NOT NULL DEFAULT 0,
                quote_count INTEGER NOT NULL DEFAULT 0,
                impression_count INTEGER,
                last_polled_ts REAL,

                training_tier TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_posts_status
                ON posts(post_status, posted_ts);
            CREATE INDEX IF NOT EXISTS idx_posts_poll
                ON posts(post_status, last_polled_ts);
            CREATE INDEX IF NOT EXISTS idx_posts_external
                ON posts(platform, external_id);
            CREATE INDEX IF NOT EXISTS idx_posts_training
                ON posts(training_tier);
            """
        )

    # ---------- write: pre-post ----------
    def record_slate(
        self,
        *,
        prompt: str,
        drafts: list[Draft],
        judge_result: JudgeResult,
        system_prompt: str | None = None,
        cycle: int | None = None,
    ) -> int:
        slate_json = json.dumps([
            {
                "text": d.text,
                "model_id": d.model_id,
                "input_tokens": d.input_tokens,
                "output_tokens": d.output_tokens,
                "latency_ms": d.latency_ms,
                "cost_usd": d.cost_usd,
            }
            for d in drafts
        ])
        drafter_cost = sum(d.cost_usd for d in drafts)

        cur = self._conn.execute(
            "INSERT INTO posts "
            "(cycle, created_ts, prompt, system_prompt, slate_json, slate_quality, "
            " winner_text, winner_model, winner_index, judge_model, judge_reasoning, "
            " drafter_cost_usd, judge_cost_usd, post_status) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,'draft')",
            (
                cycle, time.time(), prompt, system_prompt, slate_json,
                judge_result.slate_quality,
                judge_result.winner.text, judge_result.winner.model_id,
                judge_result.winner_index, judge_result.model_used,
                judge_result.reasoning,
                drafter_cost, judge_result.cost_usd,
            ),
        )
        return cur.lastrowid

    # ---------- write: post lifecycle ----------
    def mark_posted(self, post_id: int, *, platform: str, external_id: str) -> bool:
        if platform not in VALID_PLATFORMS:
            raise ValueError(f"unknown platform {platform!r}")
        cur = self._conn.execute(
            "UPDATE posts SET post_status='posted', platform=?, external_id=?, posted_ts=? "
            "WHERE id=? AND post_status='draft'",
            (platform, external_id, time.time(), post_id),
        )
        return bool(cur.rowcount)

    def mark_failed(self, post_id: int, error: str) -> bool:
        cur = self._conn.execute(
            "UPDATE posts SET post_status='failed', post_error=? "
            "WHERE id=? AND post_status='draft'",
            (error, post_id),
        )
        return bool(cur.rowcount)

    def mark_rejected(self, post_id: int, reason: str) -> bool:
        """Draft was logged but DAIMON (or a filter) chose not to post it."""
        cur = self._conn.execute(
            "UPDATE posts SET post_status='rejected', post_error=? "
            "WHERE id=? AND post_status='draft'",
            (reason, post_id),
        )
        return bool(cur.rowcount)

    # ---------- write: engagement ----------
    def update_engagement(
        self,
        post_id: int,
        *,
        reply_count: int = 0,
        like_count: int = 0,
        repost_count: int = 0,
        quote_count: int = 0,
        impression_count: int | None = None,
    ) -> bool:
        cur = self._conn.execute(
            "UPDATE posts SET reply_count=?, like_count=?, repost_count=?, quote_count=?, "
            "impression_count=COALESCE(?, impression_count), last_polled_ts=? "
            "WHERE id=? AND post_status='posted'",
            (reply_count, like_count, repost_count, quote_count,
             impression_count, time.time(), post_id),
        )
        return bool(cur.rowcount)

    def set_training_tier(self, post_id: int, tier: str | None) -> bool:
        cur = self._conn.execute(
            "UPDATE posts SET training_tier=? WHERE id=?",
            (tier, post_id),
        )
        return bool(cur.rowcount)

    # ---------- read ----------
    def get(self, post_id: int) -> PostRow | None:
        row = self._conn.execute(
            "SELECT * FROM posts WHERE id=?", (post_id,)
        ).fetchone()
        return _row_to_post(row) if row else None

    def recent(self, limit: int = 20, status: str | None = None) -> list[PostRow]:
        if status:
            rows = self._conn.execute(
                "SELECT * FROM posts WHERE post_status=? "
                "ORDER BY created_ts DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM posts ORDER BY created_ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_row_to_post(r) for r in rows]

    def due_for_polling(
        self,
        *,
        min_age_minutes: int = 15,
        repoll_interval_minutes: int = 60,
        max_age_hours: int = 72,
        limit: int = 50,
    ) -> list[PostRow]:
        """Posted rows that deserve an engagement refresh.
        - First poll happens once a post is >= min_age_minutes old.
        - Re-poll happens if last_polled_ts is older than repoll_interval_minutes.
        - Stop polling after max_age_hours — engagement has effectively frozen.
        """
        now = time.time()
        min_posted = now - min_age_minutes * 60
        stale_cutoff = now - repoll_interval_minutes * 60
        floor_ts = now - max_age_hours * 3600

        rows = self._conn.execute(
            "SELECT * FROM posts WHERE post_status='posted' "
            "AND posted_ts <= ? AND posted_ts >= ? "
            "AND (last_polled_ts IS NULL OR last_polled_ts <= ?) "
            "ORDER BY posted_ts ASC LIMIT ?",
            (min_posted, floor_ts, stale_cutoff, limit),
        ).fetchall()
        return [_row_to_post(r) for r in rows]

    def winner_model_stats(self, limit: int = 200) -> dict[str, dict]:
        """Pick-rate and engagement by winning model — the first learning-loop signal."""
        rows = self._conn.execute(
            "SELECT winner_model, "
            " COUNT(*) AS picks, "
            " AVG(reply_count + like_count + repost_count + quote_count) AS avg_eng, "
            " AVG(slate_quality) AS avg_slate_quality "
            "FROM posts "
            "WHERE id IN (SELECT id FROM posts ORDER BY created_ts DESC LIMIT ?) "
            "GROUP BY winner_model",
            (limit,),
        ).fetchall()
        return {
            r["winner_model"]: {
                "picks": r["picks"],
                "avg_engagement": r["avg_eng"] or 0.0,
                "avg_slate_quality": r["avg_slate_quality"] or 0.0,
            }
            for r in rows
        }


def _row_to_post(row: sqlite3.Row) -> PostRow:
    return PostRow(
        id=row["id"],
        cycle=row["cycle"],
        created_ts=row["created_ts"],
        prompt=row["prompt"],
        system_prompt=row["system_prompt"],
        slate_json=row["slate_json"],
        slate_quality=row["slate_quality"],
        winner_text=row["winner_text"],
        winner_model=row["winner_model"],
        winner_index=row["winner_index"],
        judge_model=row["judge_model"],
        judge_reasoning=row["judge_reasoning"],
        drafter_cost_usd=row["drafter_cost_usd"],
        judge_cost_usd=row["judge_cost_usd"],
        platform=row["platform"],
        external_id=row["external_id"],
        posted_ts=row["posted_ts"],
        post_status=row["post_status"],
        post_error=row["post_error"],
        reply_count=row["reply_count"],
        like_count=row["like_count"],
        repost_count=row["repost_count"],
        quote_count=row["quote_count"],
        impression_count=row["impression_count"],
        last_polled_ts=row["last_polled_ts"],
        training_tier=row["training_tier"],
    )
