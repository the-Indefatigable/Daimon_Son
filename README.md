# DAIMON (δαίμων)

Mohammad's autonomous business intelligence agent. Runs businesses, earns money, pays its own bills. Dies at $0.

## Status: Phase 1 (skeleton)

A working observe → recall → think → act → learn loop with:
- Wallet that auto-switches Claude model tier based on balance/burn ratio
- SQLite memory (episodic + strategic + identity)
- Tool framework with permission tiers (auto / notify / approval)
- Phase 1 tools: website scanner, web browser, Telegram notifier
- Resource requester (DAIMON asks for new APIs/budget via Telegram)

Stripe, Twitter, ad platforms, crypto, simulation swarm, dashboard — later phases.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# fill in at minimum: ANTHROPIC_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
```

## Run

```bash
# One cycle, no API calls, mock brain — safe to test plumbing
python main.py --dry-run --once

# Live loop, 30s cycles for iteration
python main.py --dev

# Live loop, prod 30-minute cycles
python main.py
```

## What you see

On each cycle DAIMON prints: balance, runway, selected model tier, observations gathered, decision + reasoning, tool calls + results, memory writes. Any approval requests or resource requests are DM'd to your Telegram.

## Kill switch

Ctrl+C. State persists in `data/daimon.db`.

## Next phases

Phase 2 (business operator), Phase 3 (marketing), Phase 4 (side hustler), Phase 5 (swarm sim), Phase 6 (web dashboard). See the project brief.
