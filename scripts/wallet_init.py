#!/usr/bin/env python
"""ONE-TIME: generate DAIMON's Base wallet + encrypt to keystore.

Mohammad runs this once. After this, data/wallet.keystore.json exists and
DAIMON can read its address + on-chain balance every cycle.

Flow:
  1. Read DAIMON_WALLET_PASSWORD from env (set it in .env before running).
  2. Generate fresh BIP39 mnemonic + secp256k1 key via eth_account.
  3. Print the mnemonic to STDOUT — Mohammad writes it down OFFLINE.
     (It is never written to disk. Paper-only backup.)
  4. Encrypt the private key with the password → standard Web3 keystore JSON
     at data/wallet.keystore.json (chmod 0600).
  5. Cache the address at data/wallet.address.txt so DAIMON's observations
     can show it without decrypting.

After this runs:
  - Send a tiny amount of ETH on Base to the printed address for gas (~0.001 ETH is plenty).
  - Optionally share the address publicly as a tip jar.
  - Set MOHAMMAD_BASE_ADDRESS in .env (where clawbacks will go).

Safety:
  - Refuses to run if keystore already exists (won't overwrite).
  - Password must be ≥8 chars.
  - Seed phrase is printed ONCE. If you lose it AND the encrypted file AND
    the password, the wallet is gone forever.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running as `python scripts/wallet_init.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import base_wallet


def main() -> int:
    password = os.getenv("DAIMON_WALLET_PASSWORD", "").strip()
    if not password:
        print("❌ DAIMON_WALLET_PASSWORD is not set in your environment.")
        print("   Put it in .env first (8+ chars, something you'll remember),")
        print("   then re-run:  .venv/bin/python scripts/wallet_init.py")
        return 1
    if len(password) < 8:
        print("❌ Password must be at least 8 characters.")
        return 1

    if base_wallet.exists():
        print(f"❌ A keystore already exists at {base_wallet.KEYSTORE_PATH}.")
        print("   Refusing to overwrite. Delete it manually only if you're sure,")
        print("   and only after confirming you have the seed phrase.")
        return 1

    print("\n" + "═" * 72)
    print("  DAIMON — Base wallet init")
    print("═" * 72)
    print("  Generating BIP39 mnemonic + secp256k1 keypair…")
    new = base_wallet.create(password=password)
    print(f"  Keystore written: {new.keystore_path}")
    print(f"  Address cached:   {base_wallet.ADDRESS_CACHE_PATH}")
    print()
    print("─" * 72)
    print("  ADDRESS (safe to share — this is the tip jar / receiving address):")
    print()
    print(f"    {new.address}")
    print()
    print("─" * 72)
    print("  SEED PHRASE (write this on paper NOW — never photograph, never type):")
    print()
    words = new.mnemonic.split()
    # Print in 3 rows of 4 for easier transcription.
    for i in range(0, len(words), 4):
        row = "   " + "  ".join(f"{i+j+1:>2}. {w}" for j, w in enumerate(words[i:i+4]))
        print(row)
    print()
    print("─" * 72)
    print("  NEXT STEPS:")
    print(f"    1. Write the 12 words above on PAPER. Store somewhere fireproof.")
    print(f"    2. Send ~0.001 ETH on Base to {new.address} for gas.")
    print(f"       (Bridge: https://bridge.base.org  — or any CEX that supports Base withdrawals)")
    print("    3. Set MOHAMMAD_BASE_ADDRESS in .env to your personal Base address")
    print("       — that's where scripts/clawback.py will send DAIMON's earnings.")
    print("    4. Optionally add the address to the Bluesky bio as a tip jar.")
    print("═" * 72)
    print("  Done. DAIMON will see its address + on-chain balance next cycle.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
