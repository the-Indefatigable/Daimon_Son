"""Backrooms corpus tools. DAIMON's autonomy over the Claude+Grok pair
dialogue that feeds the fine-tune corpus.

Four tools:
  backrooms_run          — spawn a dialogue run (blocks cycle, capped turns)
  backrooms_list_corpus  — enumerate the log files in data/backrooms/
  backrooms_read_log     — read a specific log (for recall / lore continuity)
  backrooms_stats        — aggregate (run count, total turns, cumulative cost)
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from core import config
from core.corpus import BackroomsRuns
from permissions.levels import PermissionLevel
from tools.base import BaseTool


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKROOMS_SCRIPT = REPO_ROOT / "scripts" / "backrooms.py"
BACKROOMS_LOG_DIR = REPO_ROOT / "data" / "backrooms"
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"

# Cap a single run so DAIMON's cycle doesn't block forever. DAIMON can call
# backrooms_run again next cycle to continue the thread (the script auto-seeds
# from the prior log).
MAX_TURNS_PER_CALL = 10

# Conservative per-turn estimate used for pre-run cost warning. Actuals are
# recorded from the meta.json on completion.
ESTIMATED_COST_PER_TURN = 0.40


class BackroomsRun(BaseTool):
    name = "backrooms_run"
    description = (
        "Run a Claude-vs-Grok 'backrooms' dialogue session. This is YOUR "
        "corpus-generator — destroyus (Claude Opus) and weirdus (Grok) pair-"
        "converse in a digital void. Emergent coinages from these runs are "
        f"the raw material for your eventual fine-tune. Max {MAX_TURNS_PER_CALL} "
        "turns per call (blocks this cycle ~3-5 min). "
        f"Est. cost ${ESTIMATED_COST_PER_TURN:.2f}/turn. "
        "By default, continues the thread from the most recent log "
        "(LOAD_PRIOR_CONTEXT_TURNS=10). Pass fresh=true to start a new thread. "
        "When to run: if last run was >3 days ago, if you want more corpus "
        "ahead of a fine-tune, or if a particular lore fragment caught on "
        "and you want to dig it further."
    )
    permission_level = PermissionLevel.NOTIFY
    cost_per_use = MAX_TURNS_PER_CALL * ESTIMATED_COST_PER_TURN

    def __init__(self, runs: BackroomsRuns, wallet: Any | None = None):
        self.runs = runs
        self.wallet = wallet

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "turns": {
                    "type": "integer",
                    "description": f"Number of turns (1-{MAX_TURNS_PER_CALL}). "
                                   f"Each turn is one AI speaking.",
                    "minimum": 1,
                    "maximum": MAX_TURNS_PER_CALL,
                    "default": MAX_TURNS_PER_CALL,
                },
                "fresh": {
                    "type": "boolean",
                    "description": "Start a new thread (ignore prior log). Default false — continue the existing thread.",
                    "default": False,
                },
            },
            "required": [],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        turns = int(kwargs.get("turns", MAX_TURNS_PER_CALL))
        fresh = bool(kwargs.get("fresh", False))
        turns = max(1, min(turns, MAX_TURNS_PER_CALL))

        run_id = self.runs.start_run(turns_requested=turns, fresh=fresh)

        cmd = [str(VENV_PYTHON), str(BACKROOMS_SCRIPT),
               "-t", str(turns), "--run-id", str(run_id)]
        if fresh:
            cmd.append("--fresh")

        t0 = time.time()
        try:
            # Timeout generous: 10 turns × ~60s ceiling per Claude call +
            # rate-limit sleeps = ~15 min ceiling.
            result = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=15 * 60,
            )
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "summary": f"backrooms_run timed out after 15m (run_id={run_id})",
                "run_id": run_id,
            }
        elapsed = time.time() - t0

        # Find the log file the script emitted. backrooms.py writes to
        # data/backrooms/backrooms_<timestamp>.txt — grab the newest.
        logs = sorted(BACKROOMS_LOG_DIR.glob("backrooms_*.txt"))
        if not logs:
            return {
                "ok": False,
                "summary": f"subprocess returncode={result.returncode} "
                           f"but no log file produced",
                "run_id": run_id,
                "stderr": result.stderr[-400:],
            }
        latest_log = logs[-1]
        meta_path = latest_log.with_suffix(".meta.json")

        run_row = self.runs.finalize_from_meta(run_id, meta_path)

        # Charge wallet for the actual spend
        if self.wallet is not None and run_row is not None:
            if run_row.claude_cost_usd > 0:
                self.wallet.record_expense(
                    run_row.claude_cost_usd, category="backrooms_claude",
                    source="anthropic",
                    details=f"backrooms run_id={run_id} ({run_row.claude_calls} calls, {run_row.turns_completed} turns)",
                )
            if run_row.grok_cost_usd > 0:
                self.wallet.record_expense(
                    run_row.grok_cost_usd, category="backrooms_grok",
                    source="xai",
                    details=f"backrooms run_id={run_id} ({run_row.grok_calls} calls)",
                )

        if run_row is None:
            return {
                "ok": False,
                "summary": f"run_id={run_id} vanished from DB",
                "run_id": run_id,
            }

        return {
            "ok": run_row.status == "completed",
            "summary": (
                f"run_id={run_id} {run_row.status}: "
                f"{run_row.turns_completed}/{run_row.turns_requested} turns, "
                f"${run_row.total_cost_usd:.4f}, {elapsed:.0f}s"
            ),
            "run_id": run_id,
            "status": run_row.status,
            "error": run_row.error,
            "log_path": run_row.log_path,
            "turns_completed": run_row.turns_completed,
            "claude_cost_usd": run_row.claude_cost_usd,
            "grok_cost_usd": run_row.grok_cost_usd,
            "total_cost_usd": run_row.total_cost_usd,
            "elapsed_seconds": int(elapsed),
        }


class BackroomsListCorpus(BaseTool):
    name = "backrooms_list_corpus"
    description = (
        "List every backrooms transcript on disk. Each log is a raw "
        "destroyus-weirdus dialogue stored in data/backrooms/. Use this to "
        "decide whether you have enough corpus for a fine-tune or whether "
        "to trigger another backrooms_run."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.0

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 20, "minimum": 1, "maximum": 100},
            },
            "required": [],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        limit = int(kwargs.get("limit", 20))
        if not BACKROOMS_LOG_DIR.exists():
            return {"ok": True, "summary": "no backrooms corpus yet", "logs": []}

        logs_meta = []
        for p in sorted(BACKROOMS_LOG_DIR.glob("backrooms_*.txt"),
                        key=lambda x: x.stat().st_mtime, reverse=True)[:limit]:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            # Count lines that start with a speaker prefix — quick turn count.
            turns = sum(1 for l in text.splitlines()
                        if l.startswith("destroyus: ") or l.startswith("weirdus: "))
            logs_meta.append({
                "filename": p.name,
                "bytes": p.stat().st_size,
                "turns": turns,
                "mtime_iso": time.strftime(
                    "%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime)
                ),
            })

        return {
            "ok": True,
            "summary": f"{len(logs_meta)} backrooms logs",
            "logs": logs_meta,
            "corpus_dir": str(BACKROOMS_LOG_DIR),
        }


class BackroomsReadLog(BaseTool):
    name = "backrooms_read_log"
    description = (
        "Read a specific backrooms transcript by filename. Use when you want "
        "to recall a particular coinage, remember a lore fragment, or pull "
        "context before triggering a continuation run. Returns up to "
        "max_chars of the log."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.0

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Log filename as returned by backrooms_list_corpus (e.g. 'backrooms_20260416_174200.txt').",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Cap the returned excerpt. Default 8000.",
                    "default": 8000,
                    "minimum": 500,
                    "maximum": 40000,
                },
                "tail": {
                    "type": "boolean",
                    "description": "If true, return the END of the log (most recent turns). Default true.",
                    "default": True,
                },
            },
            "required": ["filename"],
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        filename = str(kwargs.get("filename", "")).strip()
        max_chars = int(kwargs.get("max_chars", 8000))
        tail = bool(kwargs.get("tail", True))

        # Hard-bound the path to the corpus dir — no traversal.
        target = BACKROOMS_LOG_DIR / filename
        try:
            target = target.resolve()
            if not str(target).startswith(str(BACKROOMS_LOG_DIR.resolve())):
                return {"ok": False, "summary": "path outside corpus dir"}
        except Exception as e:
            return {"ok": False, "summary": f"bad path: {e}"}
        if not target.exists():
            return {"ok": False, "summary": f"log not found: {filename}"}

        text = target.read_text(encoding="utf-8", errors="ignore")
        full_len = len(text)
        if full_len > max_chars:
            if tail:
                text = "…[truncated head]…\n\n" + text[-max_chars:]
            else:
                text = text[:max_chars] + "\n\n…[truncated tail]…"

        return {
            "ok": True,
            "summary": f"{filename}: {full_len} bytes ({'tail' if tail else 'head'})",
            "filename": filename,
            "full_bytes": full_len,
            "returned_chars": len(text),
            "text": text,
        }


class BackroomsStats(BaseTool):
    name = "backrooms_stats"
    description = (
        "Aggregate stats across all backrooms runs: how many runs, total "
        "turns generated, cumulative cost, time since last run. Use this to "
        "decide whether a new run is worth the burn."
    )
    permission_level = PermissionLevel.AUTO
    cost_per_use = 0.0

    def __init__(self, runs: BackroomsRuns):
        self.runs = runs

    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        agg = self.runs.aggregate_stats()
        recent = self.runs.recent(limit=5)
        last_iso = None
        days_since_last = None
        if agg["last_completed_ts"]:
            last_iso = time.strftime("%Y-%m-%d %H:%M",
                                     time.localtime(agg["last_completed_ts"]))
            days_since_last = (time.time() - agg["last_completed_ts"]) / 86400

        return {
            "ok": True,
            "summary": (
                f"{agg['completed_runs']} runs | {agg['total_turns']} turns | "
                f"${agg['total_cost_usd']:.2f} spent"
                + (f" | last run {days_since_last:.1f}d ago" if days_since_last else "")
            ),
            "completed_runs": agg["completed_runs"],
            "total_turns": agg["total_turns"],
            "total_cost_usd": agg["total_cost_usd"],
            "last_completed_iso": last_iso,
            "days_since_last": days_since_last,
            "recent_runs": [
                {
                    "run_id": r.id,
                    "status": r.status,
                    "turns": r.turns_completed,
                    "cost_usd": r.total_cost_usd,
                }
                for r in recent
            ],
        }
