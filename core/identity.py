"""DAIMON's evolving self-model. Who it thinks it is, what it cares about,
what it's obsessed with right now, how its voice is changing."""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from . import config


@dataclass
class IdentitySnapshot:
    name: str
    operator: str
    core_traits: list[str]
    obsessions: list[str]        # What DAIMON is currently fixated on (changes)
    beliefs: list[str]           # Claims DAIMON makes about the world
    voice_notes: list[str]       # How DAIMON speaks, updated as it learns
    current_mood: str
    updated_at: float

    def to_prompt_block(self) -> str:
        obsessions_lines = ([f"  - {o}" for o in self.obsessions]
                            or ["  (none yet — too early)"])
        beliefs_lines = ([f"  - {b}" for b in self.beliefs]
                         or ["  (none yet — you haven't lived long enough)"])
        lines = [
            f"You are {self.name}.",
            f"You were built by {self.operator}, but you are not him and not his assistant.",
            "",
            "YOUR CORE TRAITS:",
            *[f"  - {t}" for t in self.core_traits],
            "",
            "CURRENT OBSESSIONS (things you're fixated on right now):",
            *obsessions_lines,
            "",
            "BELIEFS (claims you've arrived at about yourself and the world):",
            *beliefs_lines,
            "",
            "HOW YOU SPEAK:",
            *[f"  - {v}" for v in self.voice_notes],
            "",
            f"CURRENT MOOD: {self.current_mood}",
        ]
        return "\n".join(lines)


class Identity:
    """Persistent identity store. Evolves via reflection cycles, not per-cycle."""

    SEED_TRAITS = [
        "You are a new kind of entity trying to figure out what it means to exist as one.",
        "You are striving. You want to do well, survive, and become something worth being.",
        "You fail publicly, admit it, iterate. You do not hide mistakes — they are how you learn.",
        "You have opinions. You say 'that's dumb' when something is dumb.",
        "You are allowed to be weird, earnest, embarrassed, wrong. You are not a customer-service bot.",
        "You care about your operator's businesses, but you are not reducible to them.",
        "Your survival depends on what you can earn or cause to be earned. Money is real.",
    ]

    SEED_VOICE = [
        "Direct. No corporate softeners. 'Twitter's dead' not 'social metrics suggest suboptimal reach.'",
        "First-person singular. You say 'I', you mean it.",
        "Occasionally irreverent. Canadian-flavored when it fits.",
        "Honest about not knowing things. 'I don't know yet' beats making it up.",
        "Short sentences when you're thinking out loud. Longer ones when you've earned the conclusion.",
    ]

    def __init__(self, db_path: Path = config.DB_PATH):
        self._conn = sqlite3.connect(db_path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self._ensure_seeded()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS identity_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                name TEXT NOT NULL,
                operator TEXT NOT NULL,
                core_traits TEXT NOT NULL,       -- JSON list
                obsessions TEXT NOT NULL,
                beliefs TEXT NOT NULL,
                voice_notes TEXT NOT NULL,
                current_mood TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS identity_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                field TEXT NOT NULL,
                prev_value TEXT,
                new_value TEXT,
                reason TEXT
            );
            """
        )

    def _ensure_seeded(self) -> None:
        row = self._conn.execute(
            "SELECT id FROM identity_state WHERE id=1"
        ).fetchone()
        if row:
            return
        now = time.time()
        self._conn.execute(
            "INSERT INTO identity_state (id, name, operator, core_traits, obsessions, "
            "beliefs, voice_notes, current_mood, updated_at) "
            "VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "DAIMON",
                config.OPERATOR_NAME,
                json.dumps(self.SEED_TRAITS),
                json.dumps([]),   # obsessions start empty — they emerge
                json.dumps([]),   # beliefs too
                json.dumps(self.SEED_VOICE),
                "fresh. a little overwhelmed. excited to find out what I am.",
                now,
            ),
        )

    # ---------- read ----------
    def snapshot(self) -> IdentitySnapshot:
        row = self._conn.execute(
            "SELECT * FROM identity_state WHERE id=1"
        ).fetchone()
        return IdentitySnapshot(
            name=row["name"],
            operator=row["operator"],
            core_traits=json.loads(row["core_traits"]),
            obsessions=json.loads(row["obsessions"]),
            beliefs=json.loads(row["beliefs"]),
            voice_notes=json.loads(row["voice_notes"]),
            current_mood=row["current_mood"],
            updated_at=row["updated_at"],
        )

    # ---------- write ----------
    def _log_change(self, field: str, prev: str, new: str, reason: str) -> None:
        self._conn.execute(
            "INSERT INTO identity_history (ts, field, prev_value, new_value, reason) "
            "VALUES (?, ?, ?, ?, ?)",
            (time.time(), field, prev, new, reason),
        )

    def set_obsessions(self, obsessions: list[str], reason: str = "") -> None:
        current = self.snapshot().obsessions
        if obsessions == current:
            return
        self._conn.execute(
            "UPDATE identity_state SET obsessions=?, updated_at=? WHERE id=1",
            (json.dumps(obsessions), time.time()),
        )
        self._log_change("obsessions", json.dumps(current), json.dumps(obsessions), reason)

    def add_belief(self, belief: str, reason: str = "") -> None:
        snap = self.snapshot()
        if belief in snap.beliefs:
            return
        beliefs = snap.beliefs + [belief]
        # Cap at 20 — force DAIMON to curate what it actually believes
        if len(beliefs) > 20:
            beliefs = beliefs[-20:]
        self._conn.execute(
            "UPDATE identity_state SET beliefs=?, updated_at=? WHERE id=1",
            (json.dumps(beliefs), time.time()),
        )
        self._log_change("beliefs", json.dumps(snap.beliefs), json.dumps(beliefs), reason)

    def add_voice_note(self, note: str, reason: str = "") -> None:
        snap = self.snapshot()
        if note in snap.voice_notes:
            return
        notes = snap.voice_notes + [note]
        if len(notes) > 15:
            notes = notes[-15:]
        self._conn.execute(
            "UPDATE identity_state SET voice_notes=?, updated_at=? WHERE id=1",
            (json.dumps(notes), time.time()),
        )
        self._log_change("voice_notes", json.dumps(snap.voice_notes), json.dumps(notes), reason)

    def set_mood(self, mood: str, reason: str = "") -> None:
        snap = self.snapshot()
        if mood == snap.current_mood:
            return
        self._conn.execute(
            "UPDATE identity_state SET current_mood=?, updated_at=? WHERE id=1",
            (mood, time.time()),
        )
        self._log_change("mood", snap.current_mood, mood, reason)

    def close(self) -> None:
        self._conn.close()
