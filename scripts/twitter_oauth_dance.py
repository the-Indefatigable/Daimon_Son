"""One-shot OAuth helper. Authorize the existing X App as a different user.

Usage:
    .venv/bin/python scripts/twitter_oauth_dance.py

Walks you through:
  1. opens an auth URL (you log in as the target X account in your browser)
  2. you click Authorize, X gives you a PIN
  3. paste PIN here -> prints new Access Token + Secret to swap into .env
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv("/Users/alam/dev/daimon/.env")

try:
    import tweepy
except ImportError:
    sys.exit("tweepy not installed in this venv")

CK = os.getenv("TWITTER_API_KEY", "").strip()
CS = os.getenv("TWITTER_API_SECRET", "").strip()
if not CK or not CS:
    sys.exit("TWITTER_API_KEY / TWITTER_API_SECRET missing in .env")

# callback="oob" -> PIN-based flow, no callback server needed
handler = tweepy.OAuth1UserHandler(CK, CS, callback="oob")
auth_url = handler.get_authorization_url()

print()
print("=" * 70)
print("STEP 1 — open this URL in a browser logged in as the TARGET X account:")
print()
print(f"    {auth_url}")
print()
print("STEP 2 — click 'Authorize app'. X will show you a 7-digit PIN.")
print("=" * 70)
print()

pin = input("Paste the PIN here: ").strip()
if not pin:
    sys.exit("no PIN provided")

access_token, access_token_secret = handler.get_access_token(pin)

print()
print("=" * 70)
print("Success. Swap these into .env:")
print()
print(f"TWITTER_ACCESS_TOKEN={access_token}")
print(f"TWITTER_ACCESS_SECRET={access_token_secret}")
print("=" * 70)

# Verify by fetching the auth'd user
client = tweepy.Client(
    consumer_key=CK,
    consumer_secret=CS,
    access_token=access_token,
    access_token_secret=access_token_secret,
)
me = client.get_me()
if me.data:
    print(f"\nVerified: authenticated as @{me.data.username} (id {me.data.id})")
