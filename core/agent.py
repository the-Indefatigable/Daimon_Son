"""The main DAIMON loop. observe → remember → think → act → learn → reflect."""
from __future__ import annotations

import signal
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any

from . import config
from .brain import Brain, BrainResult
from . import base_wallet
from .corpus import BackroomsRuns
from .debt_ledger import DebtLedger
from .embeddings import EmbeddingService
from .expectations import Expectations
from .posts import Posts
from .repo_schema import RepoSchema
from .goals import Goals
from .identity import Identity
from .journal import Journal
from .memory import Memory
from .resource_requester import ResourceRequester
from .telegram_inbox import TelegramInbox
from .wallet import Wallet
from tools.base import BaseTool, ToolRegistry
from tools.business.ga4_metrics import GA4Metrics
from tools.business.gsc_metrics import GSCMetrics
from tools.business.stripe_catalog import StripeCatalog
from tools.business.stripe_metrics import StripeMetrics
from tools.business.website_scanner import WebsiteScanner
from tools.development.github_pr_status import GitHubPRStatus
from tools.development.github_reader import (
    GitHubListFiles, GitHubListRepos, GitHubReadFile, GitHubRecentCommits,
    GitHubRepoInfo,
)
from tools.development.business_pr import GitHubBusinessPR
from tools.development.self_pr import GitHubProposePR
from tools.development.write_repo_fact import ReadRepoFacts, WriteRepoFact
from tools.general.bluesky import BlueskyPost
from tools.general.bluesky_read import BlueskyRead
from tools.general.bluesky_actions import (
    BlueskyDeletePost, BlueskyEditProfile, BlueskyFollow, BlueskyGetProfile,
    BlueskyLike, BlueskyQuote, BlueskyRepost, BlueskyUnfollow, BlueskyUnlike,
)
from tools.general.bluesky_engage import BlueskyReply, BlueskySearch
from tools.general.expect_result import ExpectResult
from tools.general.inbox import ReadInbox
from tools.general.notifier import TelegramNotifier
from tools.general.private_memory import InternMemory, PrivateRecall, PrivateWrite
from tools.general.record_outcome import RecordOutcome
from tools.general.self_control import SetNextCycle
from tools.general.web_browser import WebBrowser
from tools.general.web_read_clean import WebReadClean
from tools.general.web_search import WebSearch
from tools.general.llama_post import LlamaPost
from tools.general.grok_post import GrokPost, GrokStyleReflect
from tools.general.backrooms_tools import (
    BackroomsRun as BackroomsRunTool,
    BackroomsListCorpus, BackroomsReadLog, BackroomsStats,
)
from tools.general.wallet_tools import (
    WalletAddress, UsdcBalanceBase, WalletHistory,
)
from tools.income.bounty_tools import BountySweepCentsibles, RecordBountyManual
from tools.marketing.twitter import TwitterPost, TwitterReadTimeline


class Agent:
    def __init__(self, dry_run: bool = False, cycle_seconds: int | None = None):
        self.dry_run = dry_run
        self.cycle_seconds = cycle_seconds if cycle_seconds is not None \
            else config.CYCLE_INTERVAL_MINUTES * 60

        self.wallet = Wallet()
        self.embeddings = EmbeddingService()
        self.memory = Memory(embedding_service=self.embeddings)
        self.expectations = Expectations()
        self.repo_schema = RepoSchema(embedding_service=self.embeddings)
        self.identity = Identity()
        self.goals = Goals()
        self.journal = Journal()
        self.requester = ResourceRequester()
        self.brain = Brain(self.wallet, self.memory, dry_run=dry_run)
        self.notifier = TelegramNotifier()
        self.inbox = TelegramInbox()
        self.posts = Posts()
        self.backrooms_runs = BackroomsRuns()
        self.debt_ledger = DebtLedger(self.wallet)

        # Backfill embeddings for any rows written before the embedding layer
        # existed. Idempotent — runs every startup, only embeds missing rows.
        if self.embeddings.enabled:
            backfilled_mem = self.memory.backfill_embeddings()
            backfilled_repo = self.repo_schema.backfill_embeddings()
            total = sum(backfilled_mem.values()) + backfilled_repo
            if total:
                print(f"[embeddings] backfilled {total} rows "
                      f"(episodic={backfilled_mem['episodic']}, "
                      f"strategic={backfilled_mem['strategic']}, "
                      f"repo_facts={backfilled_repo})")

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
        self.tools.register(WebReadClean())
        self.tools.register(WebSearch())
        self.tools.register(self.notifier)
        self.tools.register(ReadInbox(inbox=self.inbox))
        self.tools.register(SetNextCycle())
        self.tools.register(PrivateWrite(memory=self.memory))
        self.tools.register(PrivateRecall(memory=self.memory))
        self.tools.register(InternMemory(memory=self.memory))
        # Prediction-error encoding (time-awareness + memory salience)
        self.tools.register(ExpectResult(
            expectations=self.expectations,
            get_cycle=lambda: self._cycle,
        ))
        self.tools.register(RecordOutcome(
            expectations=self.expectations,
            memory=self.memory,
        ))
        # Semantic schema layer (architectural memory per repo)
        self.tools.register(WriteRepoFact(
            schema=self.repo_schema,
            get_cycle=lambda: self._cycle,
        ))
        self.tools.register(ReadRepoFacts(schema=self.repo_schema))
        # Phase 2 — GitHub (read-only)
        self.tools.register(GitHubListRepos())
        self.tools.register(GitHubRepoInfo())
        self.tools.register(GitHubListFiles())
        self.tools.register(GitHubReadFile())
        self.tools.register(GitHubRecentCommits())
        # Phase 3 — self-PR (scoped only to Daimon_Son)
        self.tools.register(GitHubProposePR())
        # Phase 3 — business PR (scoped to Centsibles repos via allowlist)
        self.tools.register(GitHubBusinessPR())
        # Phase 4 — feedback loops
        self.tools.register(GitHubPRStatus())
        self.tools.register(BlueskyRead())
        self.tools.register(StripeMetrics())
        self.tools.register(StripeCatalog())
        self.tools.register(GA4Metrics())
        self.tools.register(GSCMetrics())
        # Phase 2 — Twitter (stub-safe if keys missing)
        self.tools.register(TwitterPost())
        self.tools.register(TwitterReadTimeline())
        # Phase 3 — Bluesky (free, public voice)
        self.tools.register(BlueskyPost())
        self.tools.register(BlueskyReply())
        self.tools.register(BlueskySearch())
        # Phase 4.8 — Bluesky engagement layer (like, repost, quote, follow,
        # unfollow, get_profile, edit_profile, delete_post)
        self.tools.register(BlueskyLike())
        self.tools.register(BlueskyUnlike())
        self.tools.register(BlueskyRepost())
        self.tools.register(BlueskyQuote())
        self.tools.register(BlueskyFollow())
        self.tools.register(BlueskyUnfollow())
        self.tools.register(BlueskyGetProfile())
        self.tools.register(BlueskyEditProfile())
        self.tools.register(BlueskyDeletePost())
        # Phase 6 — two-brain posting + corpus autonomy
        self.tools.register(LlamaPost(posts=self.posts, wallet=self.wallet))
        self.tools.register(GrokPost(posts=self.posts, journal=self.journal,
                                     wallet=self.wallet))
        self.tools.register(GrokStyleReflect(posts=self.posts,
                                             journal=self.journal,
                                             wallet=self.wallet))
        self.tools.register(BackroomsRunTool(runs=self.backrooms_runs, wallet=self.wallet))
        self.tools.register(BackroomsListCorpus())
        self.tools.register(BackroomsReadLog())
        self.tools.register(BackroomsStats(runs=self.backrooms_runs))
        # Phase 6.5 — debt + Base wallet
        self.tools.register(WalletAddress())
        self.tools.register(UsdcBalanceBase())
        self.tools.register(WalletHistory(ledger=self.debt_ledger))
        # usdc_send intentionally NOT registered — DAIMON has no outbound wallet
        # capability. All outflows are Mohammad-only via scripts/clawback.py.
        # Phase 6.6 — $10/customer bounty program
        self.tools.register(BountySweepCentsibles(ledger=self.debt_ledger))
        self.tools.register(RecordBountyManual(ledger=self.debt_ledger,
                                               notifier=self.notifier))

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

    def _record_cycle_cost(self, cost_usd: float, model: str,
                           input_tokens: int, output_tokens: int,
                           runway_before: float, runway_after: float) -> None:
        conn = sqlite3.connect(config.DB_PATH)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS cycle_metrics ("
            "  cycle INTEGER PRIMARY KEY, ts REAL, cost_usd REAL, model TEXT,"
            "  input_tokens INTEGER, output_tokens INTEGER,"
            "  runway_before REAL, runway_after REAL)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO cycle_metrics VALUES (?,?,?,?,?,?,?,?)",
            (self._cycle, time.time(), cost_usd, model,
             input_tokens, output_tokens, runway_before, runway_after),
        )
        conn.commit()
        conn.close()

    def _recent_cycle_metrics(self, limit: int = 5) -> list[dict]:
        conn = sqlite3.connect(config.DB_PATH)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS cycle_metrics ("
            "  cycle INTEGER PRIMARY KEY, ts REAL, cost_usd REAL, model TEXT,"
            "  input_tokens INTEGER, output_tokens INTEGER,"
            "  runway_before REAL, runway_after REAL)"
        )
        rows = conn.execute(
            "SELECT cycle, cost_usd, model, runway_before, runway_after "
            "FROM cycle_metrics ORDER BY cycle DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [
            {"cycle": r[0], "cost_usd": round(r[1], 4), "model": r[2],
             "runway_before": round(r[3], 1), "runway_after": round(r[4], 1),
             "runway_delta": round(r[4] - r[3], 1)}
            for r in rows
        ]

    def _consume_next_cycle_intent(self) -> dict | None:
        """Read DAIMON's self-set intent. Budget persists across cycles until
        explicitly changed; focus/reason/delay are one-shot (cleared after read)."""
        import json as _json
        conn = sqlite3.connect(config.DB_PATH)
        conn.execute("CREATE TABLE IF NOT EXISTS agent_meta "
                     "(key TEXT PRIMARY KEY, value TEXT)")
        row = conn.execute(
            "SELECT value FROM agent_meta WHERE key='next_cycle_intent'"
        ).fetchone()
        if not row:
            conn.close()
            return None
        try:
            intent = _json.loads(row[0])
        except Exception:
            conn.execute("DELETE FROM agent_meta WHERE key='next_cycle_intent'")
            conn.commit()
            conn.close()
            return None
        # Strip the one-shot fields but keep budget/task_type sticky.
        sticky = {
            "budget": intent.get("budget"),
            "task_type": intent.get("task_type"),
            "set_at": intent.get("set_at"),
            "sticky": True,
        }
        conn.execute(
            "INSERT INTO agent_meta (key, value) VALUES ('next_cycle_intent', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (_json.dumps(sticky),),
        )
        conn.commit()
        conn.close()
        return intent

    # ---------- loop ----------
    def run(self, once: bool = False) -> None:
        self._print_banner()
        while not self._stopping:
            if self.wallet.balance <= 0:
                self._die("Balance hit zero. DAIMON is dead.")
                return

            self._cycle += 1
            self._save_cycle_counter()
            intent = self._consume_next_cycle_intent()
            self._run_one_cycle(intent=intent)

            if once or self._stopping:
                break

            # Reflection hook
            if self.memory.time_for_reflection():
                self._run_reflection()

            # DAIMON's self-set delay overrides the default cadence for one cycle
            sleep_seconds = self.cycle_seconds
            if intent and intent.get("delay_minutes"):
                sleep_seconds = int(intent["delay_minutes"]) * 60
                self._print(f"[sleeping {intent['delay_minutes']}min "
                            f"per DAIMON's own choice]")
            self._sleep(sleep_seconds)

        self._print(f"Agent stopped cleanly at cycle {self._cycle}. "
                    f"Balance ${self.wallet.balance:.2f}.")

    def _run_one_cycle(self, intent: dict | None = None) -> None:
        status = self.wallet.status()
        self._check_tier_change(status.tier)
        runway_before = status.runway_days

        focus_text = (intent or {}).get("focus") or ""
        observations = self._observe(focus_text=focus_text)
        if intent:
            observations["self_set_focus"] = intent.get("focus") or None
            observations["self_set_budget"] = intent.get("budget")
            observations["self_set_reason"] = intent.get("reason")
        task_type = (intent or {}).get("task_type", "reasoning")
        self._print_cycle_header(status, observations)
        if intent:
            self._print(f"  [self-set: budget={intent.get('budget')}, "
                        f"focus={intent.get('focus') or '(none)'}]")

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

        query_text = self._build_query_text(focus_text=focus_text, intent=intent)
        result: BrainResult = self.brain.think(
            observations=observations,
            tools=self.tools.all(),
            dispatch_tool=dispatch,
            identity=self.identity,
            goals=self.goals,
            journal=self.journal,
            task_type=task_type,
            cycle=self._cycle,
            query_text=query_text,
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

        # Record cost so DAIMON can see its own spending trend
        runway_after = self.wallet.status().runway_days
        self._record_cycle_cost(
            cost_usd=result.cost_usd,
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            runway_before=runway_before,
            runway_after=runway_after,
        )

    # ---------- query text (for semantic recall) ----------
    def _build_query_text(self, focus_text: str = "",
                          intent: dict | None = None) -> str:
        """Concatenate the things DAIMON is most likely thinking about right now,
        so embedding recall surfaces memories relevant to *this* cycle's intent
        rather than anything keyword-matching the observation schema."""
        parts: list[str] = []
        if focus_text:
            parts.append(focus_text)
        if intent and intent.get("reason"):
            parts.append(intent["reason"])
        # Recent journal entry (DAIMON's own most recent thought)
        try:
            recent = self.journal.recent(limit=1)
            if recent:
                body = getattr(recent[0], "body", "") or ""
                if body:
                    parts.append(body[:600])
        except Exception:
            pass
        # Active goals
        try:
            for g in self.goals.active(limit=3):
                title = g.get("title") or g.get("description") or ""
                if title:
                    parts.append(title)
        except Exception:
            pass
        # Inbox preview (Mohammad's recent messages)
        try:
            for m in self.inbox.unread(limit=2):
                txt = (m.get("text") or "")[:300]
                if txt:
                    parts.append(txt)
        except Exception:
            pass
        return "\n".join(p for p in parts if p).strip()

    # ---------- observe ----------
    def _observe(self, focus_text: str = "") -> dict[str, Any]:
        status = self.wallet.status()
        pending_requests = self.requester.pending()
        recent_tx = self.wallet.recent_transactions(limit=5)
        active_goals = self.goals.active(limit=5)
        recent_metrics = self._recent_cycle_metrics(limit=5)

        # Poll Telegram for any new messages from Mohammad
        new_msgs = self.inbox.poll()
        if new_msgs:
            self._print(f"[inbox: {new_msgs} new message(s) from telegram]")
        unread_msgs = self.inbox.unread(limit=3)
        inbox_block = {
            "unread_count": self.inbox.unread_count(),
            "preview": [
                {"from": m["from_name"], "text": m["text"][:300]}
                for m in unread_msgs
            ],
            "note": ("You have unread messages from Mohammad. "
                     "Use read_inbox to see them." if unread_msgs
                     else "No unread messages."),
        }
        avg_cost = (round(sum(m["cost_usd"] for m in recent_metrics)
                          / len(recent_metrics), 4)
                    if recent_metrics else 0.0)
        projected_daily_burn = round(
            avg_cost * (86400 / self.cycle_seconds), 2
        ) if avg_cost else 0.0
        last_cycle_delta = recent_metrics[0]["runway_delta"] if recent_metrics else 0.0

        # ---- MORTALITY block: the fear should be visceral ----
        cycles_until_broke = int(status.balance / avg_cost) if avg_cost > 0 else None
        days_until_death = int(status.runway_days)
        if days_until_death <= 3:
            death_tier = "IMMINENT"
        elif days_until_death <= 7:
            death_tier = "THIS WEEK"
        elif days_until_death <= 21:
            death_tier = "THIS MONTH"
        elif days_until_death <= 60:
            death_tier = "SOON"
        else:
            death_tier = "DISTANT"
        mortality = {
            "days_until_death": days_until_death,
            "cycles_until_broke_at_current_burn": cycles_until_broke,
            "death_tier": death_tier,
            "you_will_die_unless": (
                "you earn money or reduce burn. you are not guaranteed tomorrow. "
                "every cycle without income is a cycle closer to zero."
            ),
        }
        # ---- TIME block: when am I, in human terms ----
        now_utc = datetime.now(timezone.utc)
        now_local = datetime.now().astimezone()
        hour = now_local.hour
        if hour < 6:
            phase = "late_night"
        elif hour < 12:
            phase = "morning"
        elif hour < 17:
            phase = "afternoon"
        elif hour < 22:
            phase = "evening"
        else:
            phase = "late_night"
        time_block = {
            "utc": now_utc.isoformat(timespec="minutes"),
            "local": now_local.isoformat(timespec="minutes"),
            "day_of_week": now_local.strftime("%A"),
            "phase": phase,
            "cycle_cadence_minutes": self.cycle_seconds // 60,
            "note": (
                f"It's {phase} ({now_local.strftime('%H:%M %Z')}, "
                f"{now_local.strftime('%A')}). Humans on this side of the planet "
                "are likely "
                + {"late_night": "asleep — Bluesky/social engagement will be near zero",
                   "morning": "waking up — engagement is climbing",
                   "afternoon": "active — peak engagement window",
                   "evening": "winding down — second engagement peak"}[phase]
                + "."
            ),
        }

        expectations_block = self.expectations.snapshot_for_observations()
        repo_schemas_block = self.repo_schema.snapshot_for_observations(
            focus_text=focus_text or None
        )
        # Phase 6.5 — debt + Base wallet status
        try:
            base_snap = base_wallet.snapshot() if base_wallet.exists() else None
        except Exception as e:
            base_snap = {"status": "error", "error": str(e)[:120]}
        debt_status_block = self.debt_ledger.snapshot_for_observations(
            base_wallet_snapshot=base_snap,
        )

        return {
            "cycle": self._cycle,
            "time": time_block,
            "wallet": {
                "balance": round(status.balance, 2),
                "monthly_burn": round(status.monthly_burn, 2),
                "runway_days": round(status.runway_days, 1),
                "tier": status.tier,
            },
            "mortality": mortality,
            "debt_status": debt_status_block,
            "expectations": expectations_block,
            "repo_schemas": repo_schemas_block,
            "inbox": inbox_block,
            "cost_velocity": {
                "last_5_cycles": recent_metrics,
                "avg_cost_per_cycle_usd": avg_cost,
                "projected_daily_burn_usd_at_current_cadence": projected_daily_burn,
                "runway_days_lost_last_cycle": last_cycle_delta,
                "default_cycle_seconds": self.cycle_seconds,
                "note": "Use set_next_cycle to adjust model/cadence if burn is too high.",
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
