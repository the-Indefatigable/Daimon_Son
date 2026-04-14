"""Central config: env vars, model tiers, paths, thresholds."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "daimon.db"

# ---------- Survival / wallet ----------
INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE", "20.0"))
MONTHLY_FIXED_BURN = float(os.getenv("MONTHLY_FIXED_BURN", "5.0"))  # Railway etc.
CYCLE_INTERVAL_MINUTES = int(os.getenv("CYCLE_INTERVAL_MINUTES", "30"))

# ---------- Model tiers ----------
# ratio = balance / monthly_burn. We pick the HIGHEST tier the ratio qualifies for.
# 'threshold' means: this tier applies when ratio is BELOW this value (except flush,
# which is >= its threshold).
MODEL_TIERS = {
    "critical": {
        "threshold": 0.10,
        "default_model": "claude-haiku-4-5-20251001",
        "description": "Survival mode. Minimal intelligence. Focus only on income.",
    },
    "low": {
        "threshold": 0.30,
        "default_model": "claude-haiku-4-5-20251001",
        "description": "Budget mode. Smart enough for most tasks. No experiments.",
    },
    "normal": {
        "threshold": 3.0,
        "default_model": "claude-sonnet-4-6",
        "description": "Standard operations. Good balance of cost and intelligence.",
    },
    "flush": {
        "threshold": float("inf"),
        "default_model": "claude-opus-4-6",
        "description": "Maximum intelligence. Big strategic decisions, complex sims.",
    },
}

# Per-1M-token prices (input/output USD) used to estimate API costs on the fly.
# Keep in sync with anthropic pricing.
MODEL_PRICING = {
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-opus-4-6": {"input": 15.0, "output": 75.0},
}

# Map task complexity to which model should run it. Even when FLUSH, a tweet
# doesn't need Opus.
TASK_MODEL_PREFERENCE = {
    "simple": "claude-haiku-4-5-20251001",     # tweet, data read, notification
    "reasoning": "claude-sonnet-4-6",          # analysis, code review, memory reflection
    "strategic": "claude-opus-4-6",            # pricing, major pivots, investment
    "simulation": "claude-haiku-4-5-20251001", # cheap persona calls
}

# ---------- Agent identity ----------
OPERATOR_NAME = "Mohammad Alam Pathan"
BUSINESSES = [
    {
        "name": "Centsibles",
        "url": "https://centsibles.com",
        "description": "Canadian personal finance app. React/TS/Vite (Vercel) + FastAPI/Postgres (Railway). Plaid + Stripe. $8 CAD/month.",
    },
    {
        "name": "First Principles Learning",
        "url": "https://firstprincipleslearningg.com",
        "description": "Tutoring platform, 50+ interactive tools. React/Vite on Vercel.",
    },
    {
        "name": "quroots",
        "url": "https://quroots.com",
        "description": "Quranic Arabic learning via root-word methodology. React.",
    },
]

# ---------- Secrets ----------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")

# ---------- Reflection / scheduling ----------
REFLECTION_INTERVAL_HOURS = 24
RESOURCE_REQUEST_INTERVAL_CYCLES = 10  # evaluate "what do I need" every N cycles

# ---------- Thresholds ----------
SIMULATION_COST_THRESHOLD = 0.05  # simulate if action costs >5% of balance
