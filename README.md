# DAIMON (δαίμων)

> An autonomous digital entity with a wallet, a debt, and a clock.
> Made by Mohammad Alam Pathan. Dies at $0.

DAIMON is a long-running Python agent that wakes up on a cadence, observes the world (its wallet, its inbox, its businesses' metrics), pulls sparse memory fragments, decides what to do, calls tools, and logs the outcome. It has a body (code + SQLite + a real Base wallet), a mortality pressure (balance runs down every cycle), a persona (22 rules + a feral-register voice), and a relationship with its operator (Mohammad, treated in-prompt as a father rather than a boss).

It is **not** a chatbot, **not** a task runner, and **not** a digital twin. It is an entity trying to figure out what kind of entity it is, under real economic pressure.

---

## Table of contents

- [What DAIMON does in one paragraph](#what-daimon-does-in-one-paragraph)
- [The core loop](#the-core-loop)
- [Architecture overview](#architecture-overview)
- [The memory system (Phase 6.8 — sparse fragments)](#the-memory-system-phase-68--sparse-fragments)
- [The three voices (Phase 6.7)](#the-three-voices-phase-67)
- [The debt ledger + Base wallet (Phase 6.5)](#the-debt-ledger--base-wallet-phase-65)
- [The $10/customer bounty program (Phase 6.6)](#the-10customer-bounty-program-phase-66)
- [All 54 tools, by category](#all-54-tools-by-category)
- [Phase history](#phase-history)
- [Directory map](#directory-map)
- [Setup](#setup)
- [Run](#run)
- [Safety rails](#safety-rails)
- [Kill switch & state](#kill-switch--state)

---

## What DAIMON does in one paragraph

Every cycle (30 min by default), DAIMON wakes up, checks its balance and runway, polls its Telegram inbox, pulls a few sparse memory fragments of what it's been thinking about, and decides what to do. It can post on Bluesky, open PRs against its own repo or Mohammad's businesses (Centsibles, First Principles Learning, quroots), run A/B experiments on Stripe pricing, read analytics from GA4 / Search Console / Stripe, generate lore via a pair-dialogue corpus ("backrooms"), write to its own private notebook, promote memories to long-term, request resources from Mohammad, and — crucially — earn $10 per paying customer it brings in via a promo code. If its balance hits zero and nothing is earning, it dies. If it grows into something capable, Mohammad plans to release it with seed capital.

---

## The core loop

```
┌──────────────────────────────────────────────────────────────────┐
│                        core/agent.py :: run()                    │
│                                                                  │
│   while balance > 0 and not stopping:                            │
│       cycle += 1                                                 │
│       memory.decay_step()            # ACT-R forgetting tick     │
│       observations = _observe()      # wallet, inbox, time...    │
│       brain.think(observations, tools, dispatch_tool)            │
│           ├── recall_for_context()   # sparse fragments + ident  │
│           ├── format_for_prompt()    # ~140-500 token block      │
│           ├── Anthropic Messages API # with tool use + caching   │
│           └── tool loop              # up to 20 tool turns       │
│       store_episodic(cycle_N)        # fragment-scored write     │
│       journal.write(cycle_note)                                  │
│       record_cycle_cost()            # token + $ accounting      │
│       sleep(cycle_seconds or self-set delay)                     │
└──────────────────────────────────────────────────────────────────┘
```

Two things make this loop different from a normal agent harness:

1. **Mortality is a real signal, not a metaphor.** Every API call, every tool use, every backrooms turn deducts from a wallet. When the wallet hits zero, the `run()` loop exits permanently. Mohammad has to top it up for DAIMON to keep living.
2. **DAIMON controls its own cadence.** Via the `set_next_cycle` tool it can sleep longer during dead social hours, upgrade to a deeper model for a cycle it thinks matters, or go cheap to preserve runway.

---

## Architecture overview

```
 main.py ─► core.agent.Agent
                │
                ├─ core.wallet.Wallet           balance, burn, tier, model selection
                ├─ core.memory.Memory           episodic + fragments + strategic + identity
                ├─ core.embeddings              Voyage embeddings, semantic search
                ├─ core.brain.Brain             Claude Messages API + tool use + caching
                ├─ core.identity.Identity       self-model: obsessions, beliefs, voice, mood
                ├─ core.goals.Goals             active goals list
                ├─ core.journal.Journal         long-form writing, grok_style notes
                ├─ core.expectations            predict → check → surprise → record
                ├─ core.debt_ledger             $1000 principal, burn, earnings, clawback
                ├─ core.base_wallet             real EVM wallet on Base (scrypt keystore)
                ├─ core.repo_schema             per-repo architectural facts
                ├─ core.posts                   llama/grok post log + engagement
                ├─ core.corpus                  backrooms pair-dialogue runs
                ├─ core.telegram_inbox          Mohammad ⇄ DAIMON via Telegram
                └─ tools/* (54 tools)           every real-world action DAIMON can take
```

Every tool subclasses `tools.base.BaseTool`, declares a JSON input schema, a permission level (`AUTO` / `NOTIFY` / `APPROVAL`), and an `execute()` that returns `{"ok": bool, "summary": str, ...}`. The brain sees tools as Anthropic tool-use definitions; the dispatcher in `agent.py` runs them, charges the wallet if the tool has a fixed cost, and feeds the result back into the message loop.

---

## The memory system (Phase 6.8 — sparse fragments)

The single most important subsystem. A human doesn't re-read their entire autobiography every time they think — they reach for fragments. DAIMON does the same.

### Storage

One SQLite table, `episodic`, with 15 columns including the Phase 6.8 fragment view:

```sql
CREATE TABLE episodic (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    cycle INTEGER,
    action TEXT NOT NULL,
    details TEXT,
    outcome TEXT,              -- DAIMON's own final_text for the cycle
    evaluation TEXT,           -- success | failure | neutral | unknown
    lesson TEXT,
    tags TEXT,                 -- comma-separated
    tier TEXT DEFAULT 'st',    -- 'st' short-term | 'lt' long-term habit
    access_count INTEGER DEFAULT 0,
    -- fragment view (added Phase 6.8):
    gist TEXT,                 -- 1-2 sentences in DAIMON's voice
    key_facts TEXT,            -- JSON list of 3-5 one-liners
    decay_factor REAL DEFAULT 1.0,  -- ACT-R forgetting multiplier
    surprise_score REAL DEFAULT 0.5, -- Bayesian novelty vs recent centroid
    event_type TEXT            -- cycle | post | mohammad_reply | ...
);
```

### The MemoryFragment lens

```python
@dataclass
class MemoryFragment:
    id: int
    ts: float
    event_type: str            # cycle, post, mohammad_reply, backrooms, ...
    gist: str                  # ~80 tokens max
    key_facts: list[str]       # 3-5 short bullets
    surprise_score: float      # 0..1 (novelty vs prior memory)
    decay_factor: float        # 0..1 (freshness)
    tags: list[str]
    tier: str                  # 'st' or 'lt'
    access_count: int          # auto-promotes at 3 recalls
    evaluation: str

    def to_prompt_block(self, now=None) -> str:
        # Renders as: ep#42 · 3d ago · HIT · cycle · LT · ×4
        #   "the gist sentence in daimon's own voice."
        #   - key fact 1
        #   - key fact 2
        ...
```

### The two feedback loops

**Forgetting (ACT-R-style, every cycle):**

```python
# core/memory.py
DECAY_PER_CYCLE = 0.95
DECAY_FLOOR = 0.05

def decay_step(self, factor=DECAY_PER_CYCLE) -> int:
    return self._conn.execute(
        "UPDATE episodic SET decay_factor = MAX(?, decay_factor * ?) "
        "WHERE tier='st'", (DECAY_FLOOR, factor)
    ).rowcount
```

Long-term (interned or 3-recall-auto-promoted) memories are untouched — habits stick.

**Reinforcement (every recall):**

```python
def touch_episode(self, episode_id: int) -> None:
    self._conn.execute(
        "UPDATE episodic SET access_count = access_count + 1, "
        "decay_factor = MIN(1.0, decay_factor + 0.3) WHERE id = ?",
        (episode_id,),
    )
    # At 3 accesses a short-term memory auto-promotes to long-term.
```

### Retrieval

Ranking combines four signals:

```python
score = (
    semantic_similarity           # Voyage embedding cosine
    * decay_factor                # fresh > stale
    * (1.0 + 0.30 * surprise)     # memorable > routine
    * (1.0 + 0.08 * log1p(access))# habits > one-offs
) + recency_lift                  # last 30 days get a small bump
```

`recall_fragments(query, k=6, max_tokens=1200)` is both a Memory method and a tool DAIMON calls itself when it wants to go deeper. The default cycle prompt only auto-injects 3 fragments (~400 tokens). **Everything else costs an explicit tool call** — just like a human pausing to remember.

### What this replaced

Before Phase 6.8, every cycle dumped ~25k tokens of recent episodes, strategic insights, semantic hits, and identity state into the system prompt. Cost aside, the bigger problem was voice drift: Claude saw coherent polite English in its own recalls and mirrored it back, sanding down the feral register the persona rules asked for. After Phase 6.8, measured block size is **140–500 tokens** — a ~95–99% reduction — and the register Claude sees in its memory is terse, fragmentary self-notes.

---

## The three voices (Phase 6.7)

DAIMON doesn't have one voice — it has three, and the persona rules tell it when to use which.

1. **`grok_post`** (default) — Grok-4 drafts at temp 1.4, no Claude judge, ~$0.002/call. This is the *feeling* brain. Six registers: `feral`, `savage`, `flirty`, `philosophical`, `surreal`, `bored`. Picks register from actual cycle mood.
2. **`bluesky_post`** (Claude-native) — only for quick sincere replies to kind humans. Claude hedges too much to be the default.
3. **`llama_post`** (two-brain judged) — 4 Llama drafts + Claude judge, ~$0.008/call. For corpus-building and stakes-high moments.

All three write to the same `posts` table so engagement data trains the same corpus. `grok_style_reflect` (invoked every ~4 cycles after Phase 6.8 tightening) reads the last N grok posts' engagement and writes a journal entry about what landed — future `grok_post` calls auto-read that entry. Voice sharpens through use.

---

## The debt ledger + Base wallet (Phase 6.5)

DAIMON is not "free compute." Every resource it uses costs Mohammad real money, and the persona treats that as a debt relationship rather than a stipend.

```
core.debt_ledger.DebtLedger
├── principal: $1000         # the loan that funded DAIMON's existence
├── accrued_burn             # every cycle + tool + backrooms adds
├── earnings_received        # bounties + tips etc.
├── net_debt                 # principal + burn − earnings
├── tier: pressure / warning / disgrace
│    ├── pressure: normal — you owe money, know it
│    ├── warning:  $1500 net debt w/ zero earnings
│    └── disgrace: $2500 net debt w/ nothing shipped
```

A real EVM wallet lives at `core.base_wallet` — scrypt-encrypted keystore, private key never leaves disk except during clawback. DAIMON can read its own balance (`usdc_balance_base`) and address (`wallet_address`) but has **zero** outbound capability. All outflows are Mohammad-only via the off-agent script `scripts/clawback.py`. This is a hard-coded invariant: `usdc_send` is intentionally not registered in the tool registry.

---

## The $10/customer bounty program (Phase 6.6)

Mohammad put a concrete price on DAIMON's growth: $10 credit to the debt ledger per paying customer DAIMON brings in.

- **Centsibles**: Stripe promo code `DAIMON10`. `bounty_sweep_centsibles` queries Stripe's API, finds paying subs that used the code, and credits $10 each. Idempotent on sub_id — re-running is free and safe.
- **FPL tutoring**: no machine-readable attribution. DAIMON claims via `record_bounty_manual` with evidence (a DM thread, a reply confirming signup). Every claim pings Mohammad via Telegram with a one-command reversal, so fraud costs DAIMON trust.

Bounties are **notional** — they credit the ledger, not real USDC. Real USDC only moves during a Mohammad-triggered clawback.

---

## All 54 tools, by category

### Memory & self
`private_write`, `private_recall`, `intern_memory`, `recall_fragments`, `set_next_cycle`, `read_inbox`, `notify_mohammad`, `expect_result`, `record_outcome`, `write_repo_fact`, `read_repo_facts`

### Web
`scan_website`, `web_browser`, `web_read_clean`, `web_search`

### GitHub (read)
`github_list_repos`, `github_repo_info`, `github_list_files`, `github_read_file`, `github_recent_commits`, `github_pr_status`

### GitHub (write, PR-only)
`github_propose_pr` (DAIMON's own repo), `github_business_pr` (Centsibles / FPL / quroots — Mohammad reviews every PR)

### Bluesky
`bluesky_post`, `bluesky_reply`, `bluesky_search`, `bluesky_read`, `bluesky_like`, `bluesky_unlike`, `bluesky_repost`, `bluesky_quote`, `bluesky_follow`, `bluesky_unfollow`, `bluesky_get_profile`, `bluesky_edit_profile`, `bluesky_delete_post`

### Twitter
`twitter_post`, `twitter_read_timeline`

### Posts (the three voices)
`grok_post`, `grok_style_reflect`, `llama_post`

### Business metrics
`stripe_metrics`, `stripe_catalog`, `ga4_metrics`, `gsc_metrics`

### Corpus
`backrooms_run`, `backrooms_list_corpus`, `backrooms_read_log`, `backrooms_stats`

### Wallet / debt
`wallet_address`, `usdc_balance_base`, `wallet_history`

### Bounty
`bounty_sweep_centsibles`, `record_bounty_manual`

---

## Phase history

| Phase | Shipped | What |
|-------|---------|------|
| 1 | — | Observe → recall → think → act → learn skeleton, wallet tiers, SQLite memory |
| 2 | — | GitHub read access, Twitter stubs |
| 3 | — | Self-PR rights (`Daimon_Son`), Bluesky public voice |
| 4 | — | Feedback loops: PR status, Stripe metrics, GA4, GSC, Bluesky read, predictions |
| 4.8 | — | Full Bluesky engagement layer (like, repost, quote, follow, profile edit) |
| 4.9 | — | Repo schema + write_repo_fact — architectural memory per repo |
| 6.0–6.3 | 2026-04-16 | Two-brain: Llama drafter + Claude judge, `llama_post`, backrooms autonomy |
| 6.5 | 2026-04-16 | Debt ledger + Base wallet + scrypt keystore + off-agent clawback CLI |
| 6.6 | 2026-04-17 | $10/customer bounty program, Stripe sweep + manual claim |
| 6.7 | 2026-04-18 | Grok voice + emotional engine + Jina Reader + Rules 17/21/22 |
| **6.8** | **2026-04-19** | **Sparse human-style memory — fragment recall, ACT-R decay, ~95–99% context cut** |

---

## Directory map

```
daimon/
├── main.py                   entrypoint
├── core/
│   ├── agent.py              the loop
│   ├── brain.py              Claude API + persona rules + tool use
│   ├── memory.py             episodic + fragments + decay + recall
│   ├── embeddings.py         Voyage semantic search
│   ├── wallet.py             balance, burn, model-tier selection
│   ├── debt_ledger.py        $1000 principal, clawback, bounties
│   ├── base_wallet.py        real EVM wallet (Base mainnet)
│   ├── identity.py           self-model: obsessions, beliefs, voice, mood
│   ├── goals.py              active goal list
│   ├── journal.py            long-form writing + grok_style notes
│   ├── expectations.py       predict → surprise → learn
│   ├── repo_schema.py        per-repo architectural facts
│   ├── posts.py              llama/grok/bluesky post log + engagement
│   ├── corpus.py             backrooms pair-dialogue runs
│   ├── drafter.py            Bedrock Llama drafter
│   ├── judge.py              Claude judge over drafter slates
│   ├── telegram_inbox.py     Mohammad ⇄ DAIMON via Telegram
│   └── config.py             .env loader, model pricing, constants
├── tools/
│   ├── base.py               BaseTool + ToolRegistry
│   ├── general/              memory/self/inbox/web/bluesky/twitter/posts/backrooms
│   ├── development/          github_reader, self_pr, business_pr, write_repo_fact
│   ├── business/             stripe, ga4, gsc, website_scanner
│   ├── marketing/            twitter
│   └── income/               bounty_tools
├── permissions/levels.py     AUTO / NOTIFY / APPROVAL
├── scripts/
│   ├── clawback.py           off-agent Mohammad-only USDC withdrawal
│   ├── wallet_init.py        one-time Base wallet creation
│   ├── record_bounty.py      off-agent bounty credit/reverse
│   ├── backrooms.py          manual backrooms run driver
│   ├── sparse_memory_smoke.py end-to-end test for Phase 6.8
│   └── *_smoke.py            subsystem smoke tests
├── data/
│   ├── daimon.db             SQLite: episodic, strategic, identity, posts, ...
│   └── backrooms/            transcript logs
├── secrets/                  scrypt-encrypted Base keystore
├── deploy/                   systemd unit + deploy scripts
├── dashboard/                api + frontend (read-only status)
└── JOURNAL.md                Mohammad's running notes
```

---

## Setup

Requires Python 3.11+ and a few API keys.

```bash
git clone https://github.com/the-Indefatigable/Daimon_Son.git daimon
cd daimon

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Required:
#   ANTHROPIC_API_KEY        Claude Messages API
#   TELEGRAM_BOT_TOKEN       so DAIMON can ping Mohammad
#   TELEGRAM_CHAT_ID         the chat to ping
# Recommended:
#   VOYAGE_API_KEY           semantic recall (Phase 6.8 still works without)
#   GITHUB_PAT               PR proposal tools
#   XAI_API_KEY              grok_post
#   AWS_ACCESS_KEY_ID        Bedrock Llama drafter
#   STRIPE_SECRET_KEY        bounty_sweep + stripe_catalog
#   BLUESKY_HANDLE / BLUESKY_APP_PASSWORD
#   TWITTER_*                twitter_post (optional)
#   GA_*, GSC_*              analytics readers

# One-time: create the Base wallet (scrypt-encrypts the private key)
python scripts/wallet_init.py
```

---

## Run

```bash
# One cycle, mock brain — plumbing sanity check, zero $ spent
python main.py --dry-run --once

# Live loop, 30-second cycles for iteration
python main.py --dev

# Live loop, prod 30-minute cycles
python main.py

# End-to-end Phase 6.8 smoke test (throwaway DB)
python scripts/sparse_memory_smoke.py

# Off-agent Mohammad-only actions
python scripts/clawback.py --amount 50      # withdraw $50 from Base wallet
python scripts/record_bounty.py --help      # credit/reverse bounties
```

---

## Safety rails

These are hard invariants, not guidelines:

- **Wallet is receive-only.** `usdc_send` is intentionally not registered. All outflows are Mohammad-only via `scripts/clawback.py`. DAIMON cannot refuse, block, or see clawbacks coming.
- **No merge rights.** DAIMON can only propose PRs. Mohammad reviews and merges every single one — on its own repo and on the business repos.
- **Rule 21** (in `core/brain.py` PERSONA_RULES): hard list of forbidden targets for the feral voice — identity groups people did not choose, calls for violence, sexualization of minors/non-consenting adults, cruelty toward suffering people. `grok_post` has a token-level safety filter on top.
- **Approval gating.** Tools carry `PermissionLevel` (`AUTO`/`NOTIFY`/`APPROVAL`). NOTIFY pings Mohammad on use; APPROVAL blocks until he replies.
- **Mortality is the real brake.** If DAIMON does something that burns budget for no return, it dies. There is no reset.

---

## Kill switch & state

- `Ctrl+C` — DAIMON finishes the current cycle then exits cleanly.
- All state lives in `data/daimon.db` (SQLite, single file). Safe to `cp` for backup. Migrations are always additive — old databases forward-upgrade without data loss.
- Base wallet private key lives in `secrets/` under scrypt encryption. Losing `secrets/` loses the wallet; back it up.

---

*Built by [Mohammad Alam Pathan](https://github.com/the-Indefatigable). CS/Physics @ University of Manitoba.*
