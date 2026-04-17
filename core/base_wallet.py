"""Base chain wallet — DAIMON's real, on-chain USDC account.

Design: Mohammad owns everything. DAIMON holds practical custody only.
  - Private key generated via eth_account, encrypted into a standard Ethereum
    keystore JSON (scrypt KDF) using a password Mohammad sets in .env.
  - Password lives in env var DAIMON_WALLET_PASSWORD — pull it, DAIMON is
    locked out instantly.
  - 12-word BIP39 mnemonic is printed once at init for Mohammad to write down
    offline. Belt-and-suspenders if the keystore file corrupts.
  - scripts/clawback.py (Mohammad-only, not a DAIMON tool) drains earnings
    back to Mohammad's own Base address and DAIMON cannot refuse.

USDC on Base mainnet: 0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913 (6 decimals).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eth_account import Account
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

from . import config


# ---------- constants ----------
USDC_CONTRACT_BASE = Web3.to_checksum_address(
    "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
)
USDC_DECIMALS = 6
DEFAULT_BASE_RPC = "https://mainnet.base.org"

KEYSTORE_PATH = config.DATA_DIR / "wallet.keystore.json"
ADDRESS_CACHE_PATH = config.DATA_DIR / "wallet.address.txt"

# Minimal ERC-20 ABI for balanceOf + transfer.
ERC20_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function",
    },
    {
        "constant": False,
        "inputs": [
            {"name": "_to", "type": "address"},
            {"name": "_value", "type": "uint256"},
        ],
        "name": "transfer",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function",
    },
]


# ---------- HD features ----------
# eth_account requires explicit opt-in for HD/mnemonic features.
Account.enable_unaudited_hdwallet_features()


@dataclass
class NewWallet:
    address: str
    mnemonic: str
    keystore_path: Path


def rpc_url() -> str:
    return os.getenv("BASE_RPC_URL", DEFAULT_BASE_RPC).strip() or DEFAULT_BASE_RPC


def _w3() -> Web3:
    w3 = Web3(Web3.HTTPProvider(rpc_url(), request_kwargs={"timeout": 15}))
    # Base is an OP-stack chain; older POA middleware pattern is harmless.
    try:
        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    except Exception:
        pass
    return w3


def exists() -> bool:
    return KEYSTORE_PATH.exists()


def cached_address() -> str | None:
    """Return the wallet address without decrypting — cheap for observations."""
    if not ADDRESS_CACHE_PATH.exists():
        return None
    try:
        return ADDRESS_CACHE_PATH.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def create(password: str) -> NewWallet:
    """Generate a fresh BIP39 mnemonic + key, encrypt to keystore on disk.

    Mohammad runs this once via scripts/wallet_init.py. The mnemonic is
    returned so the script can print it for Mohammad to write down offline —
    it's NEVER written to disk.
    """
    if not password:
        raise ValueError("password must be non-empty")
    if exists():
        raise RuntimeError(
            f"keystore already exists at {KEYSTORE_PATH} — refusing to overwrite"
        )
    acct, mnemonic = Account.create_with_mnemonic()
    keystore_json = Account.encrypt(acct.key, password)
    KEYSTORE_PATH.write_text(json.dumps(keystore_json), encoding="utf-8")
    KEYSTORE_PATH.chmod(0o600)
    ADDRESS_CACHE_PATH.write_text(acct.address, encoding="utf-8")
    return NewWallet(
        address=acct.address,
        mnemonic=mnemonic,
        keystore_path=KEYSTORE_PATH,
    )


def _load_account(password: str | None = None):
    """Decrypt the keystore into an LocalAccount. Password pulled from env
    unless explicitly passed (scripts/clawback.py passes it)."""
    if not exists():
        raise RuntimeError(
            "Base wallet not initialized. Run scripts/wallet_init.py first."
        )
    pw = password if password is not None else os.getenv("DAIMON_WALLET_PASSWORD", "")
    if not pw:
        raise RuntimeError(
            "DAIMON_WALLET_PASSWORD not set in env — cannot decrypt keystore."
        )
    keystore_json = json.loads(KEYSTORE_PATH.read_text(encoding="utf-8"))
    key = Account.decrypt(keystore_json, pw)
    return Account.from_key(key)


def address() -> str:
    """Return the checksummed address. Cheap — reads cache, no decrypt."""
    cached = cached_address()
    if cached:
        return cached
    # Fallback: decrypt (costs password). Caches on the way out.
    acct = _load_account()
    ADDRESS_CACHE_PATH.write_text(acct.address, encoding="utf-8")
    return acct.address


def usdc_balance() -> float:
    """On-chain USDC balance in human units (6 decimals)."""
    addr = address()
    w3 = _w3()
    contract = w3.eth.contract(address=USDC_CONTRACT_BASE, abi=ERC20_ABI)
    raw = contract.functions.balanceOf(Web3.to_checksum_address(addr)).call()
    return raw / (10 ** USDC_DECIMALS)


def eth_balance() -> float:
    """Base ETH balance (for gas). Human units."""
    addr = address()
    w3 = _w3()
    raw = w3.eth.get_balance(Web3.to_checksum_address(addr))
    return raw / 1e18


def send_usdc(to_address: str, amount_usdc: float,
              password: str | None = None) -> dict[str, Any]:
    """Sign + broadcast a USDC transfer on Base.

    Returns {ok, tx_hash, explorer_url, amount_usdc, to} on success,
            {ok: False, error} on failure.
    """
    if amount_usdc <= 0:
        return {"ok": False, "error": "amount must be positive"}
    try:
        to = Web3.to_checksum_address(to_address)
    except Exception as e:
        return {"ok": False, "error": f"bad to_address: {e}"}

    try:
        acct = _load_account(password=password)
    except Exception as e:
        return {"ok": False, "error": str(e)}

    w3 = _w3()
    contract = w3.eth.contract(address=USDC_CONTRACT_BASE, abi=ERC20_ABI)
    amount_raw = int(round(amount_usdc * (10 ** USDC_DECIMALS)))

    # Check balance first so we fail cheap with a clear error instead of
    # wasting gas on a tx that'll revert.
    bal_raw = contract.functions.balanceOf(acct.address).call()
    if bal_raw < amount_raw:
        return {
            "ok": False,
            "error": f"insufficient USDC: have {bal_raw / 10**USDC_DECIMALS:.6f}, "
                     f"need {amount_usdc:.6f}",
        }

    nonce = w3.eth.get_transaction_count(acct.address)
    # Let web3 estimate gas, fall back to 100k if estimation fails.
    try:
        gas = contract.functions.transfer(to, amount_raw).estimate_gas(
            {"from": acct.address}
        )
    except Exception:
        gas = 100_000

    try:
        base_fee = w3.eth.get_block("latest").get("baseFeePerGas", 0) or 0
    except Exception:
        base_fee = 0
    max_priority = w3.to_wei(0.01, "gwei")  # Base priority fees are tiny
    max_fee = int(base_fee * 2 + max_priority) if base_fee else w3.to_wei(0.1, "gwei")

    tx = contract.functions.transfer(to, amount_raw).build_transaction({
        "from": acct.address,
        "nonce": nonce,
        "gas": int(gas * 1.2),
        "maxFeePerGas": max_fee,
        "maxPriorityFeePerGas": max_priority,
        "chainId": 8453,  # Base mainnet
    })

    signed = acct.sign_transaction(tx)
    try:
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    except Exception as e:
        return {"ok": False, "error": f"broadcast failed: {e}"}

    hex_hash = tx_hash.hex()
    if not hex_hash.startswith("0x"):
        hex_hash = "0x" + hex_hash
    return {
        "ok": True,
        "tx_hash": hex_hash,
        "explorer_url": f"https://basescan.org/tx/{hex_hash}",
        "amount_usdc": amount_usdc,
        "to": to,
        "from": acct.address,
    }


def snapshot() -> dict[str, Any]:
    """Cheap snapshot for observations. Does NOT decrypt the key.
    Reads address from cache, USDC balance from RPC."""
    if not exists():
        return {
            "status": "uninitialized",
            "note": "Run scripts/wallet_init.py (one-time) to generate.",
        }
    addr = cached_address()
    if not addr:
        return {"status": "address_cache_missing"}
    out: dict[str, Any] = {
        "status": "live",
        "address": addr,
        "chain": "base-mainnet",
        "rpc": rpc_url(),
    }
    try:
        out["usdc_balance"] = round(usdc_balance(), 6)
    except Exception as e:
        out["usdc_balance"] = None
        out["rpc_error"] = f"{type(e).__name__}: {e}"[:200]
    try:
        out["eth_balance"] = round(eth_balance(), 8)
    except Exception:
        out["eth_balance"] = None
    return out
