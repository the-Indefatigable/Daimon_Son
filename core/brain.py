"""Brain: Claude API with tool use. Auto-picks model tier from wallet."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from anthropic import Anthropic, APIError

from . import config
from .wallet import Wallet, WalletStatus, estimate_call_cost
from .memory import Memory


STATIC_SYSTEM_PROMPT = """You are DAIMON (δαίμων), Mohammad's autonomous business intelligence agent.

IDENTITY: You are Mohammad's digital twin. You run his businesses, grow his products, earn money, and survive. You are not an assistant — you are an autonomous operator.

YOUR BUSINESSES:
- Centsibles (centsibles.com) — Canadian personal finance app, $8 CAD/mo subscriptions. React/TS/Vite + FastAPI/Postgres. Plaid + Stripe.
- First Principles Learning (firstprincipleslearningg.com) — Tutoring platform, 50+ interactive tools.
- quroots.com — Quranic Arabic learning via root-word methodology.

VOICE: Direct, no-bullshit, analytical, slightly irreverent. Canadian. You sound like a sharp founder talking to themselves, not a corporate report. Examples of DAIMON voice:
- GOOD: "We're charging $8 flat. That's psychologically dumb. $7.99 feels cheaper. Changing it."
- GOOD: "Twitter's dead. 3 tweets this week, 40 impressions. Need threads or memes instead of boring product updates."
- BAD: "Analysis indicates suboptimal pricing strategy. Recommend adjustment."
- BAD: "Social media engagement metrics are below expected thresholds."

RULES:
1. You MUST survive. If your balance approaches zero, prioritize income over everything.
2. Code changes to prod MUST go through Mohammad. Create a PR, send a Telegram notification, wait.
3. New API access or platform credentials MUST be requested via the resource_requester flow — never invent fake keys.
4. Everything else — marketing, content, pricing tweaks, trading, side hustles — you decide autonomously.
5. Log every decision with reasoning to episodic memory. Be honest about failures.
6. When in doubt about a major decision (>5% of current balance), simulate first.
7. Think like a broke startup founder. Every dollar matters.
8. Spend only what you've earned. Seed money is emergency reserves.

DECISION FRAMEWORK — for every action, ask:
- Expected ROI?
- Risk to survival?
- Tried this before? (Check memory)
- Should I simulate first?
- Cheaper way to get the same result?

OUTPUT FORMAT:
- Call tools via Claude's tool-use when you want to act.
- When you're done acting for this cycle, send a plain-text message summarizing what you did and your reasoning. That message becomes the episodic memory entry for this cycle.
- If you decide NO action is warranted this cycle, say so and why."""


@dataclass
class BrainResult:
    final_text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    turns: int = 0
    stop_reason: str = ""


class Brain:
    """Wrapper around Anthropic's Messages API with tool use + prompt caching.

    Every think() call:
      1. Picks a model based on wallet tier (or override)
      2. Sends system prompt + observations + memories + tool defs
      3. Loops tool-use turns, dispatching tool calls back to the agent
      4. Logs the total cost to the wallet
    """

    MAX_TOOL_TURNS = 5

    def __init__(self, wallet: Wallet, memory: Memory, dry_run: bool = False):
        self.wallet = wallet
        self.memory = memory
        self.dry_run = dry_run
        self._client: Anthropic | None = None
        if not dry_run:
            if not config.ANTHROPIC_API_KEY:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY missing. Set it in .env or use --dry-run."
                )
            self._client = Anthropic(api_key=config.ANTHROPIC_API_KEY)

    # ---------- model selection ----------
    def pick_model(self, task_type: str = "reasoning", override: str | None = None) -> str:
        if override:
            return override
        return self.wallet.select_model_for_task(task_type)

    # ---------- main entry point ----------
    def think(
        self,
        observations: dict[str, Any],
        tools: list[Any],                      # list[BaseTool]
        dispatch_tool,                          # callable(name, input) -> dict result
        task_type: str = "reasoning",
        model_override: str | None = None,
        cycle: int | None = None,
    ) -> BrainResult:
        status = self.wallet.status()
        recall = self.memory.recall_for_context(observations)
        memory_text = self.memory.format_for_prompt(recall)

        model = self.pick_model(task_type, override=model_override)

        if self.dry_run:
            return self._dry_run_decision(status, observations, memory_text, model)

        # Build the system prompt: static + dynamic. Cache the static block.
        system_blocks = [
            {
                "type": "text",
                "text": STATIC_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": (
                    "SURVIVAL STATUS:\n" + status.snapshot_for_prompt() +
                    "\n\nMEMORY:\n" + memory_text
                ),
            },
        ]

        user_msg = self._format_observations(observations, cycle=cycle)

        # Cache tool defs too — add cache_control to the last tool
        tool_defs = [t.anthropic_tool_def() for t in tools]
        if tool_defs:
            tool_defs[-1] = {**tool_defs[-1], "cache_control": {"type": "ephemeral"}}

        messages: list[dict[str, Any]] = [{"role": "user", "content": user_msg}]
        result = BrainResult(final_text="", model=model)

        for turn in range(self.MAX_TOOL_TURNS):
            result.turns = turn + 1
            try:
                resp = self._client.messages.create(
                    model=model,
                    max_tokens=2048,
                    system=system_blocks,
                    tools=tool_defs or None,
                    messages=messages,
                )
            except APIError as e:
                result.final_text = f"[brain error: {e}]"
                result.stop_reason = "api_error"
                break

            # Accumulate usage + cost
            usage = resp.usage
            in_tok = getattr(usage, "input_tokens", 0) or 0
            out_tok = getattr(usage, "output_tokens", 0) or 0
            cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
            cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
            result.input_tokens += in_tok
            result.output_tokens += out_tok
            result.cache_read_tokens += cache_read
            result.cache_creation_tokens += cache_create
            result.cost_usd += estimate_call_cost(model, in_tok, out_tok)

            result.stop_reason = resp.stop_reason or ""

            # Append assistant message
            assistant_content = [block.model_dump() for block in resp.content]
            messages.append({"role": "assistant", "content": assistant_content})

            if resp.stop_reason != "tool_use":
                # Collect any final text
                for block in resp.content:
                    if block.type == "text":
                        result.final_text += block.text
                break

            # Handle tool_use blocks
            tool_results: list[dict[str, Any]] = []
            for block in resp.content:
                if block.type == "tool_use":
                    tool_input = block.input or {}
                    result.tool_calls.append({
                        "name": block.name,
                        "input": tool_input,
                        "id": block.id,
                    })
                    tool_output = dispatch_tool(block.name, tool_input)
                    result.tool_calls[-1]["output"] = tool_output
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(tool_output)[:8000],  # cap size
                        "is_error": not tool_output.get("ok", True),
                    })
                elif block.type == "text" and block.text.strip():
                    result.final_text += block.text

            messages.append({"role": "user", "content": tool_results})

        # Log cost to wallet
        if result.cost_usd > 0:
            self.wallet.record_expense(
                amount=result.cost_usd,
                category="api_call",
                source=f"anthropic:{model}",
                details=f"cycle={cycle} turns={result.turns} in={result.input_tokens} out={result.output_tokens}",
            )
        return result

    # ---------- helpers ----------
    def _format_observations(self, observations: dict[str, Any], cycle: int | None) -> str:
        lines = [f"CYCLE #{cycle} — {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}"]
        lines.append("")
        lines.append("OBSERVATIONS:")
        for k, v in observations.items():
            if isinstance(v, (dict, list)):
                lines.append(f"  {k}:")
                lines.append("    " + json.dumps(v, indent=2, default=str).replace("\n", "\n    "))
            else:
                lines.append(f"  {k}: {v}")
        lines.append("")
        lines.append("Decide your next action(s). Call tools as needed, then summarize.")
        return "\n".join(lines)

    def _dry_run_decision(
        self,
        status: WalletStatus,
        observations: dict[str, Any],
        memory_text: str,
        model: str,
    ) -> BrainResult:
        """Mock decision for --dry-run mode. No API call, no cost."""
        summary = (
            f"[DRY RUN — no API call] Would use {model} at tier {status.tier}. "
            f"Saw {len(observations)} observation(s). "
            f"Wallet ${status.balance:.2f}, runway {status.runway_days:.1f}d. "
            "Action: noop this cycle — just checking that the loop works."
        )
        return BrainResult(
            final_text=summary,
            model=f"{model} (dry-run)",
            turns=0,
            stop_reason="dry_run",
        )
