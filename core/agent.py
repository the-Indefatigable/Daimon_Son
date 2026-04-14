"""The main DAIMON loop. observe → remember → think → act → learn → reflect."""
from __future__ import annotations

import signal
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any

from . import config
from .brain import Brain, BrainResult
from .memory import Memory
from .resource_requester import ResourceRequester
from .wallet import Wallet
from tools.base import BaseTool, ToolRegistry
from tools.business.website_scanner import WebsiteScanner
from tools.general.notifier import TelegramNotifier
from tools.general.web_browser import WebBrowser


class Agent:
    def __init__(self, dry_run: bool = False, cycle_seconds: int | None = None):
        self.dry_run = dry_run
        self.cycle_seconds = cycle_seconds if cycle_seconds is not None \
            else config.CYCLE_INTERVAL_MINUTES * 60

        self.wallet = Wallet()
        self.memory = Memory()
        self.requester = ResourceRequester()
        self.brain = Brain(self.wallet, self.memory, dry_run=dry_run)
        self.notifier = TelegramNotifier()

        self.tools = ToolRegistry()
        self._register_phase1_tools()

        self._cycle = self._load_cycle_counter()
        self._last_tier = self.wallet.status().tier
        self._stopping = False
        signal.signal(signal.SIGINT, self._handle_stop)
        signal.signal(signal.SIGTERM, self._handle_stop)

    # ---------- setup ----------
    def _register_phase1_tools(self) -> None:
        self.tools.register(WebsiteScanner())
        self.tools.register(WebBrowser())
        self.tools.register(self.notifier)

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

        # Log the cycle itself as episodic memory
        tool_names = [tc["name"] for tc in result.tool_calls]
        self.memory.store_episodic(
            action=f"cycle_{self._cycle}",
            details=(
                f"observations={list(observations.keys())}; "
                f"tools_used={tool_names}; model={result.model}"
            ),
            outcome=result.final_text[:2000],
            evaluation="unknown",  # DAIMON will evaluate outcomes in reflection
            tags=["cycle"] + tool_names,
            cycle=self._cycle,
        )

    # ---------- observe ----------
    def _observe(self) -> dict[str, Any]:
        status = self.wallet.status()
        pending_requests = self.requester.pending()
        recent_tx = self.wallet.recent_transactions(limit=5)
        return {
            "cycle": self._cycle,
            "wallet": {
                "balance": round(status.balance, 2),
                "monthly_burn": round(status.monthly_burn, 2),
                "runway_days": round(status.runway_days, 1),
                "tier": status.tier,
            },
            "pending_resource_requests": len(pending_requests),
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

        # Use a reasoning-tier model for reflection
        model = self.brain.pick_model("reasoning")
        prompt_summary = "\n".join(
            f"- [{e['evaluation']}] {e['action']}: {e['outcome'][:300]}"
            for e in recent
        )

        # One-shot reflection call (no tools)
        try:
            resp = self.brain._client.messages.create(  # type: ignore[union-attr]
                model=model,
                max_tokens=1024,
                system="You are DAIMON reflecting on your recent actions. Be honest about failures. Output four labeled sections: WINS, LOSSES, PATTERNS, NEXT ACTIONS.",
                messages=[{
                    "role": "user",
                    "content": f"Recent episodes (most recent first):\n\n{prompt_summary}\n\nReflect.",
                }],
            )
            text = "".join(b.text for b in resp.content if b.type == "text")
            self.memory.store_reflection(summary=text)
            self._print(f"[reflection stored — {len(text)} chars]")

            # Log the cost
            usage = resp.usage
            from .wallet import estimate_call_cost
            cost = estimate_call_cost(
                model,
                getattr(usage, "input_tokens", 0) or 0,
                getattr(usage, "output_tokens", 0) or 0,
            )
            if cost > 0:
                self.wallet.record_expense(cost, "api_call", f"anthropic:{model}",
                                           "reflection")
        except Exception as e:
            self._print(f"[reflection failed: {e}]")

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
