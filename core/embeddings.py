"""EmbeddingService: semantic recall via Voyage embeddings.

Replaces keyword-only recall with embedding similarity. When `VOYAGE_API_KEY`
is set + voyageai/numpy installed, `enabled=True` and writes to memory tables
auto-embed. When not, this whole layer no-ops cleanly — keyword recall still
works.

Storage: a single `embeddings` table keyed by (source_table, source_id), so
any number of source tables (episodic, strategic, repo_facts) can share it
without schema bloat. Vectors stored as raw float32 bytes via numpy.

Search: brute-force cosine over all stored vectors. Fast enough for the
~thousands-of-rows scale DAIMON will hit in the first year. If/when row
count crosses ~50k, swap in faiss or sqlite-vec without changing the API.
"""
from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from typing import Iterable

from . import config

try:
    import voyageai  # type: ignore
    _VOYAGE_AVAILABLE = True
except ImportError:
    _VOYAGE_AVAILABLE = False

try:
    import numpy as np  # type: ignore
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False


DEFAULT_MODEL = "voyage-3"
MAX_INPUT_CHARS = 30_000      # ~7-8k tokens, well under voyage-3's 32k limit
BATCH_SIZE = 128              # voyage allows up to 128 inputs per call


class EmbeddingService:
    """Voyage-backed semantic recall. Safe to instantiate even without keys."""

    def __init__(self, db_path: Path = config.DB_PATH, model: str = DEFAULT_MODEL):
        self.db_path = db_path
        self.model = model
        self._conn = sqlite3.connect(db_path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

        api_key = os.getenv("VOYAGE_API_KEY")
        self.enabled = bool(api_key) and _VOYAGE_AVAILABLE and _NUMPY_AVAILABLE
        self._reason_disabled = ""
        if not api_key:
            self._reason_disabled = "VOYAGE_API_KEY not set"
        elif not _VOYAGE_AVAILABLE:
            self._reason_disabled = "voyageai package not installed"
        elif not _NUMPY_AVAILABLE:
            self._reason_disabled = "numpy not installed"

        self._client = voyageai.Client(api_key=api_key) if self.enabled else None

    # ---------- schema ----------
    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                source_table TEXT NOT NULL,
                source_id INTEGER NOT NULL,
                text TEXT NOT NULL,
                vector BLOB NOT NULL,
                dim INTEGER NOT NULL,
                model TEXT NOT NULL,
                created_ts REAL NOT NULL,
                PRIMARY KEY (source_table, source_id)
            );
            CREATE INDEX IF NOT EXISTS idx_emb_model ON embeddings(model);
            """
        )

    # ---------- embed ----------
    def _embed_call(self, texts: list[str], input_type: str) -> list[list[float]] | None:
        if not self.enabled:
            return None
        chunked = [t[:MAX_INPUT_CHARS] for t in texts if t and t.strip()]
        if not chunked:
            return []
        try:
            r = self._client.embed(chunked, model=self.model, input_type=input_type)
            return r.embeddings
        except Exception as e:
            print(f"[embeddings] voyage call failed: {e}")
            return None

    def embed_and_store(self, source_table: str, source_id: int, text: str) -> bool:
        if not self.enabled or not text or not text.strip():
            return False
        embs = self._embed_call([text], input_type="document")
        if not embs:
            return False
        self._store_vector(source_table, source_id, text, embs[0])
        return True

    def embed_and_store_batch(self, triples: list[tuple[str, int, str]]) -> int:
        """triples = [(source_table, source_id, text), ...]. Returns # stored."""
        if not self.enabled or not triples:
            return 0
        # Filter out empty texts up front
        valid = [(t, i, txt) for (t, i, txt) in triples if txt and txt.strip()]
        if not valid:
            return 0
        stored = 0
        for start in range(0, len(valid), BATCH_SIZE):
            chunk = valid[start:start + BATCH_SIZE]
            texts = [c[2] for c in chunk]
            embs = self._embed_call(texts, input_type="document")
            if embs is None:
                continue
            for (table, sid, text), vec in zip(chunk, embs):
                self._store_vector(table, sid, text, vec)
                stored += 1
        return stored

    def _store_vector(self, source_table: str, source_id: int,
                      text: str, vec: list[float]) -> None:
        import numpy as np
        arr = np.asarray(vec, dtype=np.float32)
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings "
            "(source_table, source_id, text, vector, dim, model, created_ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (source_table, source_id, text[:5000], arr.tobytes(),
             int(arr.shape[0]), self.model, time.time()),
        )

    # ---------- search ----------
    def search(
        self,
        query: str,
        k: int = 5,
        source_tables: list[str] | None = None,
        min_similarity: float = 0.0,
    ) -> list[dict]:
        if not self.enabled or not query or not query.strip():
            return []
        import numpy as np
        embs = self._embed_call([query], input_type="query")
        if not embs:
            return []
        qvec = np.asarray(embs[0], dtype=np.float32)

        if source_tables:
            placeholders = ",".join("?" for _ in source_tables)
            rows = self._conn.execute(
                f"SELECT source_table, source_id, text, vector, dim "
                f"FROM embeddings WHERE source_table IN ({placeholders})",
                source_tables,
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT source_table, source_id, text, vector, dim FROM embeddings"
            ).fetchall()
        if not rows:
            return []

        mat = np.vstack([
            np.frombuffer(r["vector"], dtype=np.float32) for r in rows
        ])
        qnorm = float(np.linalg.norm(qvec)) or 1e-9
        mnorms = np.linalg.norm(mat, axis=1)
        mnorms[mnorms == 0] = 1e-9
        sims = (mat @ qvec) / (mnorms * qnorm)

        top_idx = np.argsort(-sims)[:k]
        out = []
        for i in top_idx:
            sim = float(sims[i])
            if sim < min_similarity:
                continue
            r = rows[i]
            out.append({
                "source_table": r["source_table"],
                "source_id": r["source_id"],
                "text": r["text"][:400],
                "similarity": round(sim, 4),
            })
        return out

    # ---------- maintenance ----------
    def stored_count(self, source_table: str | None = None) -> int:
        if source_table:
            row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM embeddings WHERE source_table=?",
                (source_table,),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM embeddings"
            ).fetchone()
        return int(row["c"]) if row else 0

    def missing_for_table(self, source_table: str, source_table_pk: str = "id",
                          source_table_name: str | None = None) -> list[int]:
        """Return source IDs from source_table that don't yet have embeddings."""
        table = source_table_name or source_table
        rows = self._conn.execute(
            f"SELECT s.{source_table_pk} AS sid FROM {table} s "
            f"LEFT JOIN embeddings e "
            f"ON e.source_table=? AND e.source_id=s.{source_table_pk} "
            f"WHERE e.source_id IS NULL",
            (source_table,),
        ).fetchall()
        return [r["sid"] for r in rows]

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "model": self.model if self.enabled else None,
            "reason_disabled": self._reason_disabled,
            "stored_total": self.stored_count(),
            "by_table": {
                t: self.stored_count(t)
                for t in ("episodic", "strategic", "repo_facts")
            },
        }

    def close(self) -> None:
        self._conn.close()
