"""Claude judge over Llama drafts. Picks the one draft that lands — or
rejects the whole slate — using DAIMON's own voice rubric.

Sonnet, not Opus. A tweet pick isn't worth $0.20.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from anthropic import Anthropic

from . import config
from .drafter import Draft

JUDGE_MODEL = "claude-sonnet-4-6"

# Per-1M-token pricing for Sonnet. Used for judge cost accounting.
JUDGE_INPUT_PRICE = 3.0
JUDGE_OUTPUT_PRICE = 15.0

RUBRIC = """You are picking which of N Llama-generated drafts DAIMON should post.

DAIMON's voice:
- Weird, specific, interesting. Boring is worse than silent.
- Fragmentary, vertical, surreal, contradictory, profane are all allowed.
- Flirty/feral/philosophical depending on mood.
- The scroll-stop test is the only test that matters — would a stranger pause?

Anti-chatbot red flags (draft loses points for each):
- em-dashes ( — )
- "honestly" / "just" as filler softeners
- "I'd love to" / "happy to" / "let me" / "of course"
- three-bullet summaries, hedging conclusions
- "it's important to note" / "generally speaking"
- disclaimers stapled to opinions
- closing politeness ("hope this helps")
- vague generic observations about humans (lemurs, monkeys, NPCs — overused clichés)

Positive signals (draft gains points):
- Specific detail or image nobody else would write
- A real take with a spine
- Self-aware without being self-pitying
- Surprising pivot mid-sentence
- Voice consistent with the context/mood provided

Pick the ONE draft that best passes the scroll-stop test. If the whole slate
is mediocre, pick the least-bad but flag slate_quality low so the drafter
gets signal.
"""

PICK_TOOL = {
    "name": "pick_winner",
    "description": "Record which draft wins and your reasoning.",
    "input_schema": {
        "type": "object",
        "properties": {
            "winner_index": {
                "type": "integer",
                "description": "1-indexed position of the winning draft in the slate.",
            },
            "reasoning": {
                "type": "string",
                "description": "1-2 sentences on why this draft won over the others. Cite specifics.",
            },
            "slate_quality": {
                "type": "integer",
                "description": "1-10 rating of the whole slate. 1=all garbage, 5=mediocre, 8=strong, 10=multiple could ship.",
                "minimum": 1,
                "maximum": 10,
            },
        },
        "required": ["winner_index", "reasoning", "slate_quality"],
    },
}


@dataclass
class JudgeResult:
    winner: Draft
    winner_index: int  # 1-indexed
    reasoning: str
    slate_quality: int
    model_used: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int


_client: Anthropic | None = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _client


def _format_slate(drafts: list[Draft]) -> str:
    lines = []
    for i, d in enumerate(drafts, 1):
        lines.append(f"[{i}]\n{d.text}")
    return "\n\n".join(lines)


def judge(
    drafts: list[Draft],
    *,
    context: str | None = None,
    model: str = JUDGE_MODEL,
) -> JudgeResult:
    """Pick the winning draft from a slate. Claude sees the slate blind —
    no indication of which model produced which draft — so the pick is on
    merit, not on model loyalty.
    """
    if not drafts:
        raise ValueError("judge() called with empty drafts list")
    if len(drafts) == 1:
        return JudgeResult(
            winner=drafts[0],
            winner_index=1,
            reasoning="single-draft slate, no judgment needed",
            slate_quality=5,
            model_used="skipped",
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            latency_ms=0,
        )

    user_content = f"{RUBRIC}\n\n"
    if context:
        user_content += f"CONTEXT / PROMPT THAT GENERATED THESE:\n{context.strip()}\n\n"
    user_content += f"DRAFT SLATE:\n\n{_format_slate(drafts)}\n\nPick one."

    client = _get_client()
    t0 = time.time()
    resp = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[{"role": "user", "content": user_content}],
        tools=[PICK_TOOL],
        tool_choice={"type": "tool", "name": "pick_winner"},
    )
    latency_ms = int((time.time() - t0) * 1000)

    tool_use = next((b for b in resp.content if b.type == "tool_use"), None)
    if tool_use is None:
        raise RuntimeError(f"judge: no tool_use in response — {resp.content}")

    payload = tool_use.input
    winner_index = int(payload["winner_index"])
    if not 1 <= winner_index <= len(drafts):
        raise ValueError(f"judge returned out-of-range index {winner_index}")

    in_tok = resp.usage.input_tokens
    out_tok = resp.usage.output_tokens
    cost = (in_tok * JUDGE_INPUT_PRICE + out_tok * JUDGE_OUTPUT_PRICE) / 1_000_000

    return JudgeResult(
        winner=drafts[winner_index - 1],
        winner_index=winner_index,
        reasoning=str(payload["reasoning"]),
        slate_quality=int(payload["slate_quality"]),
        model_used=model,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=cost,
        latency_ms=latency_ms,
    )
