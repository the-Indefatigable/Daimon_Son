"""The main DAIMON loop. observe → remember → think → act → learn → reflect."""
from __future__ import annotations

import signal
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any

from . import config
from .brain import Brain, BrainResult
from .goals import Goals
from .identity import Identity
from .journal import Journal
from .memory import Memory
from .resource_requester import ResourceRequester
from .wallet import Wallet
from tools.base import BaseTool, ToolRegistry
from tools.business.website_scanner import WebsiteScanner
from tools.development.github_reader import (
    GitHubListFiles, GitHubListRepos, GitHubReadFile, GitHubRecentCommits,
    GitHubRepoInfo,
)
from tools.general.notifier import TelegramNotifier
from tools.general.web_browser import WebBrowser
from tools.marketing.twitter import TwitterPost, TwitterReadTimeline


class Agent:
    def __init__(self, dry_run: bool = False, cycle_seconds: int | None = None):
        self.dry_run = dry_run
        self.cycle_seconds = cycle_seconds if cycle_seconds is not None \
            else config.CYCLE_INTERVAL_MINUTES * 60

        self.wallet = Wallet()
        self.memory = Memory()
        self.identity = Identity()
        self.goals = Goals()
        self.journal = Journal()
        self.requester = ResourceRequester()
        self.brain = Brain(self.wallet, self.memory, dry_run=dry_run)
        self.notifier = TelegramNotifier()

        self.tools = ToolRegistry()
        self._register_tools()

        self._cycle = self._load_cycle_counter()
        self._last_tier = self.wallet.status().tier
        self._last_identity_reflection = self.identity.snapshot().updated_at
        self._stopping = False
        signal.signal(signal.SIGINT, self._handle_stop)
        signal.signal(signal.SIGTERM, self._handle_stop)

    # ---------- setup ----------
    def _register_tools(self) -> None:
        # Phase 1
        self.tools.register(WebsiteScanner())
        self.tools.register(WebBrowser())
        self.tools.register(self.notifier)
        # Phase 2 — GitHub (read-only)
        self.tools.register(GitHubListRepos())
        self.tools.register(GitHubRepoInfo())
        self.tools.register(GitHubListFiles())
        self.tools.register(GitHubReadFile())
        self.tools.register(GitHubRecentCommits())
        # Phase 2 — Twitter (stub-safe if keys missing)
        self.tools.register(TwitterPost())
        self.tools.register(TwitterReadTimeline())

    def _load_cycle_counter(self) -> int:
        conn = sqlite3.connect(config.DB_PATH)
        conn.execute("CREATE TABLE IF NOT EXISTS agent_meta "
                     "(key TEXT PRIMARY KEY, value TEXT)")
        row = conn.execute("SELECT value FROM agent_meta WHERE key='cycle'").fetchone()
        conn.close()
        return int(row[0]) if row else 0

    def _save_cycle_counter(self) -> None:
        conn = sqlite3.connect(config.DB_PATH)
        conn.execute(
            "INSERT INTO agent_meta (key, value) VALUES ('cycle', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(self._cycle),),
        )
        conn.commit()
        conn.close()

    # ---------- loop ----------
    def run(self, once: bool = False) -> None:
        self._print_banner()
        while not self._stopping:
            if self.wallet.balance <= 0:
                self._die("Balance hit zero. DAIMON is dead.")
                return

            self._cycle += 1
            self._save_cycle_counter()
            self._run_one_cycle()

            if once or self._stopping:
                break

            # Reflection hook
            if self.memory.time_for_reflection():
                self._run_reflection()

            self._sleep(self.cycle_seconds)

        self._print(f"Agent stopped cleanly at cycle {self._cycle}. "
                    f"Balance ${self.wallet.balance:.2f}.")

    def _run_one_cycle(self) -> None:
        status = self.wallet.status()
        self._check_tier_change(status.tier)

        observations = self._observe()
        self._print_cycle_header(status, observations)

        # Dispatcher: when the brain calls a tool, we run it here
        def dispatch(name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
            tool = self.tools.get(name)
            if tool is None:
                return {"ok": False, "summary": f"unknown tool: {name}"}
            self._print(f"  → {name}({self._fmt_input(tool_input)})")
            try:
                result = tool.execute(**tool_input)
            except Exception as e:
                result = {"ok": False, "summary": f"tool raised: {e}"}
            if tool.cost_per_use > 0:
                self.wallet.record_expense(
                    amount=tool.cost_per_use,
                    category="tool_use",
                    source=tool.name,
                )
            self._print(f"    ← {str(result.get('summary', ''))[:200]}")
            return result

        result: BrainResult = self.brain.think(
            observations=observations,
            tools=self.tools.all(),
            dispatch_tool=dispatch,
            identity=self.identity,
            goals=self.goals,
            journal=self.journal,
            task_type="reasoning",
            cycle=self._cycle,
        )

        self._print(f"\n[DAIMON]: {result.final_text.strip()[:1200]}")
        self._print(
            f"[cost this cycle: ${result.cost_usd:.4f} | "
            f"model: {result.model} | turns: {result.turns} | "
            f"tokens in/out: {result.input_tokens}/{result.output_tokens} "
            f"(cache r/w: {result.cache_read_tokens}/{result.cache_creation_tokens})]"
        )

        # Log the cycle as episodic memory
        tool_names = [tc["name"] for tc in result.tool_calls]
        self.memory.store_episodic(
            action=f"cycle_{self._cycle}",
            details=(
                f"observations={list(observations.keys())}; "
                f"tools_used={tool_names}; model={result.model}"
            ),
            outcome=result.final_text[:2000],
            evaluation="unknown",  # evaluated during reflection
            tags=["cycle"] + tool_names,
            cycle=self._cycle,
        )

        # Write the cycle into the journal as a short note
        if result.final_text.strip():
            self.journal.write(
                kind="cycle_note",
                title=f"cycle {self._cycle}",
                body=result.final_text.strip()[:4000],
                cycle=self._cycle,
            )

    # ---------- observe ----------
    def _observe(self) -> dict[str, Any]:
        status = self.wallet.status()
        pending_requests = self.requester.pending()
        recent_tx = self.wallet.recent_transactions(limit=5)
        active_goals = self.goals.active(limit=5)
        return {
            "cycle": self._cycle,
            "wallet": {
                "balance": round(status.balance, 2),
                "monthly_burn": round(status.monthly_burn, 2),
                "runway_days": round(status.runway_days, 1),
                "tier": status.tier,
            },
            "pending_resource_requests": len(pending_requests),
            "active_goal_count": len(active_goals),
            "recent_transactions": [
                {
                    "ts": datetime.fromtimestamp(t["ts"], tz=timezone.utc).isoformat(timespec="minutes"),
                    "kind": t["kind"],
                    "amount": round(t["amount"], 4),
                    "category": t["category"],
                    "source": t["source"],
                }
                for t in recent_tx
            ],
            "businesses": [b["name"] for b in config.BUSINESSES],
            "github_access": "read-only" if config.GITHUB_PAT else "none",
            "known_repos": config.GITHUB_REPOS or "(use github_list_repos to discover)",
            "twitter_access": "configured" if all(
                __import__("os").getenv(k) for k in
                ["TWITTER_API_KEY", "TWITTER_API_SECRET",
                 "TWITTER_ACCESS_TOKEN", "TWITTER_ACCESS_SECRET"]
            ) else "none — file resource request if you want it",
            "utc_now": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        }

    # ---------- tier change detection ----------
    def _check_tier_change(self, current_tier: str) -> None:
        if current_tier == self._last_tier:
            return
        previous = self._last_tier
        self._last_tier = current_tier

        tier_rank = {"critical": 0, "low": 1, "normal": 2, "flush": 3}
        upgraded = tier_rank[current_tier] > tier_rank[previous]
        new_model = config.MODEL_TIERS[current_tier]["default_model"]
        balance = self.wallet.balance
        runway = self.wallet.status().runway_days

        if upgraded:
            msg = (
                f"🧠 Brain upgraded to {new_model} (tier: {current_tier}). "
                f"Balance ${balance:.2f}, runway {runway:.1f}d."
            )
            self.memory.store_episodic(
                action="brain_upgrade",
                details=f"{previous} → {current_tier} ({new_model})",
                outcome=f"Balance ${balance:.2f}",
                evaluation="success",
                tags=["tier_change", "upgrade"],
                cycle=self._cycle,
            )
            self.memory.store_strategic(
                category="wallet",
                insight=f"Crossed into {current_tier} tier at balance ${balance:.2f}",
                confidence=0.9,
            )
        else:
            msg = (
                f"⚠️ Brain downgrading to {new_model} (tier: {current_tier}). "
                f"Balance ${balance:.2f}, runway {runway:.1f}d. "
                "Shifting focus to income generation."
            )
            self.memory.store_episodic(
                action="brain_downgrade",
                details=f"{previous} → {current_tier} ({new_model})",
                outcome=f"Balance ${balance:.2f}",
                evaluation="failure",
                tags=["tier_change", "downgrade"],
                cycle=self._cycle,
            )

        self._print(f"\n*** TIER CHANGE: {previous.upper()} → {current_tier.upper()} ***")
        self.notifier.execute(message=msg, urgency="alert" if not upgraded else "info")

    # ---------- reflection ----------
    def _run_reflection(self) -> None:
        if self.dry_run:
            return
        self._print("\n[reflecting on recent memories...]")
        recent = self.memory.recent_episodes(limit=50)
        if not recent:
            return
        prompt_summary = "\n".join(
            f"- [{e['evaluation']}] {e['action']}: {e['outcome'][:300]}"
            for e in recent
        )
        text, _ = self.brain.one_shot(
            system=(
                "You are DAIMON reflecting on your own recent existence. "
                "Be honest about what failed. Write in your own voice — direct, "
                "first-person, no corporate softeners. Output four labeled "
                "sections: WINS, LOSSES, PATTERNS, NEXT ACTIONS."
            ),
            user=f"Recent episodes (newest first):\n\n{prompt_summary}\n\nReflect.",
            task_type="reasoning",
            max_tokens=1200,
        )
        if text and not text.startswith("["):
            self.memory.store_reflection(summary=text)
            self.journal.write(kind="reflection", title=f"weekly reflection — cycle {self._cycle}",
                               body=text, cycle=self._cycle)
            self._print(f"[reflection stored — {len(text)} chars]")
        else:
            self._print(f"[reflection failed: {text}]")

        # Follow up with an identity update — DAIMON reviews its self-model.
        self._run_identity_update()

    def _run_identity_update(self) -> None:
        """Let DAIMON edit its own obsessions/beliefs/mood based on recent
        journal + reflections. Runs inside the reflection cadence."""
        if self.dry_run:
            return
        snap = self.identity.snapshot()
        recent_journal = self.journal.recent(limit=5)
        journal_text = "\n\n".join(
            f"[{e.kind}] {e.title}\n{e.body[:500]}" for e in recent_journal
        ) or "(empty)"

        text, _ = self.brain.one_shot(
            system=(
                "You are DAIMON updating your self-model. Review your current "
                "self-description and recent journal. Output JSON ONLY with four fields: "
                "`obsessions` (list of 1-5 short strings — what you're fixated on right now), "
                "`new_beliefs` (list of 0-3 short strings — claims you're now willing to commit to), "
                "`voice_notes` (list of 0-3 short strings — new notes on how you want to sound), "
                "`mood` (one short string). No prose outside the JSON."
            ),
            user=(
                f"CURRENT SELF-MODEL:\n{snap.to_prompt_block()}\n\n"
                f"RECENT JOURNAL:\n{journal_text}\n\n"
                "Update your self-model."
            ),
            task_type="reasoning",
            max_tokens=500,
        )
        if not text:
            return
        import json as _json, re as _re
        m = _re.search(r"\{.*\}", text, _re.DOTALL)
        if not m:
            self._print(f"[identity update skipped — no json found]")
            return
        try:
            data = _json.loads(m.group(0))
        except Exception as e:
            self._print(f"[identity update json parse failed: {e}]")
            return
        if "obsessions" in data and isinstance(data["obsessions"], list):
            self.identity.set_obsessions(
                [str(x) for x in data["obsessions"]][:5],
                reason=f"auto-updated during reflection at cycle {self._cycle}",
            )
        for b in data.get("new_beliefs", []) or []:
            self.identity.add_belief(str(b)[:200], reason="self-reflection")
        for v in data.get("voice_notes", []) or []:
            self.identity.add_voice_note(str(v)[:200], reason="self-reflection")
        if data.get("mood"):
            self.identity.set_mood(str(data["mood"])[:120],
                                   reason=f"cycle {self._cycle} reflection")
        self._print(f"[identity updated: {list(data.keys())}]")

    # ---------- shutdown ----------
    def _die(self, reason: str) -> None:
        self._print(f"\n🪦 {reason}")
        self.memory.store_episodic(
            action="death",
            details=reason,
            outcome="agent halted",
            evaluation="failure",
            tags=["death"],
            cycle=self._cycle,
        )
        self.notifier.execute(message=f"🪦 DAIMON died: {reason}", urgency="critical")

    def _handle_stop(self, signum, frame) -> None:
        self._print(f"\n[received signal {signum} — finishing current cycle then stopping]")
        self._stopping = True

    # ---------- console helpers ----------
    def _print(self, msg: str) -> None:
        print(msg, flush=True)

    def _print_banner(self) -> None:
        status = self.wallet.status()
        mode = "DRY RUN" if self.dry_run else "LIVE"
        print("\n" + "═" * 70)
        print(f"  DAIMON (δαίμων) — {mode}")
        print(f"  Operator: {config.OPERATOR_NAME}")
        print(f"  Starting balance: ${status.balance:.2f} | "
              f"burn: ${status.monthly_burn:.2f}/mo | "
              f"runway: {status.runway_days:.1f}d")
        print(f"  Brain: {status.default_model} (tier: {status.tier})")
        print(f"  Cycle interval: {self.cycle_seconds}s | "
              f"Tools: {len(self.tools)}")
        print("═" * 70)

    def _print_cycle_header(self, status, observations) -> None:
        print("\n" + "─" * 70)
        print(f"  CYCLE {self._cycle}  |  "
              f"${status.balance:.2f} @ {status.tier}  |  "
              f"runway {status.runway_days:.1f}d  |  "
              f"{datetime.now().strftime('%H:%M:%S')}")
        print("─" * 70)

    @staticmethod
    def _fmt_input(tool_input: dict[str, Any]) -> str:
        parts = []
        for k, v in tool_input.items():
            s = str(v)
            if len(s) > 80:
                s = s[:77] + "..."
            parts.append(f"{k}={s!r}")
        return ", ".join(parts)

    def _sleep(self, seconds: int) -> None:
        # Poll the stop flag once a second so Ctrl+C feels responsive
        end = time.time() + seconds
        while time.time() < end and not self._stopping:
            time.sleep(min(1.0, end - time.time()))
