"""Load the verified token registry (config/tokens.json) into a heuristic.TokenRegistry.

The registry is the ground truth for detection: `legit_addresses` drives zero-type (ZTT)
classification, and `legit_names`/`legit_symbols` drive fake-type (FTT) impersonation
checks. Built + verified by scripts/build_token_registry.py + verify_token_registry.py.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .heuristic import TokenInfo, TokenRegistry, normalize_address

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "tokens.json"

# Canonical brand names/symbols by (uppercased) symbol. On L2s the legit token is often a
# BRIDGED variant ("zkSync Bridged USDC"), but fake tokens impersonate the canonical brand
# ("USD Coin" / "USDC"). Seed legit_names/legit_symbols with these aliases so FTT catches
# name- and symbol-based impersonation regardless of the bridged variant's own metadata.
CANONICAL_ALIASES = {
    "USDC": ({"usd coin", "usdc"}, {"usdc"}),
    "USDT": ({"tether usd", "tether", "usdt"}, {"usdt"}),
    "WETH": ({"wrapped ether", "weth"}, {"weth"}),
    "WBTC": ({"wrapped btc", "wrapped bitcoin", "wbtc"}, {"wbtc"}),
    "DAI": ({"dai", "dai stablecoin"}, {"dai"}),
    "USDE": ({"usde", "ethena usde"}, {"usde"}),
}


def load_token_registry(chain: str, config_path: str | Path | None = None) -> TokenRegistry:
    """Build the TokenRegistry for one chain from the verified config."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    data = json.loads(Path(path).read_text())
    entries = data.get("chains", {}).get(chain, {})

    legit_addresses = set()
    legit_names = set()
    legit_symbols = set()
    token_info = {}
    for addr, info in entries.items():
        na = normalize_address(addr)
        legit_addresses.add(na)
        name, symbol = info.get("name", ""), info.get("symbol", "")
        if name:
            legit_names.add(name.casefold())
        if symbol:
            legit_symbols.add(symbol.casefold())
        # add canonical brand aliases (bridged tokens carry variant metadata; fakes
        # impersonate the canonical brand)
        alias_names, alias_syms = CANONICAL_ALIASES.get(
            (info.get("canonical_symbol") or symbol).upper(), (set(), set()))
        legit_names.update(alias_names)
        legit_symbols.update(alias_syms)
        token_info[na] = TokenInfo(address="0x" + na, name=name, symbol=symbol)

    return TokenRegistry(
        legit_addresses=frozenset(legit_addresses),
        legit_names=frozenset(legit_names),
        legit_symbols=frozenset(legit_symbols),
        token_info=token_info,
    )


def available_chains(config_path: str | Path | None = None) -> list[str]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    return sorted(json.loads(Path(path).read_text()).get("chains", {}))


# --- valuation metadata (shared by value_scan.py and longitudinal.py) -------------------------
# USD reference price per symbol. Pegged stablecoins are $1 and time-invariant, giving a
# price-assumption-free loss floor; volatile assets carry conservative round references, are
# recorded for traceability, and are excluded from that floor.
STABLE_SYMBOLS = {"USDC", "USDC.E", "USDBC", "USDT", "USDT0", "USDTE", "DAI", "USDE"}
REFERENCE_PRICE = {"WETH": 3000.0, "WBTC": 60000.0, "CBBTC": 60000.0,
                   "ARB": 1.0, "OP": 1.0, "LINK": 15.0}
DECIMALS_CACHE = Path(__file__).resolve().parents[1] / "config" / "token_decimals.json"


def token_value_meta(chain: str, rpc_url: str | None = None, strict: bool = True) -> dict[str, tuple]:
    """{address: (symbol, decimals, usd_price, is_stable)} for a chain's registry tokens.

    Decimals are read from each contract ON-CHAIN rather than hardcoded -- they are chain- and
    token-specific (USDC is 6 on one chain, 18 elsewhere) and a hand-maintained table silently goes
    stale when a chain is added, mis-scaling a loss figure by orders of magnitude.

    They are also CACHED on disk, because public RPCs throttle non-deterministically: on Base,
    three identical requests resolved 5/6, 0/6 and 0/6 tokens, with a different token missing each
    time. Without a cache the published loss would depend on which lookups happened to succeed.
    Decimals are immutable per contract, so caching is safe and makes valuation reproducible
    offline.

    With `strict` (default) an unresolved token raises rather than being dropped: silently omitting
    a token understates the loss, which is exactly the failure this guard exists to prevent.
    """
    from . import token_meta                      # local import: only needed when valuing
    reg = load_token_registry(chain)
    addrs = ["0x" + a for a in reg.legit_addresses]

    cache = json.loads(DECIMALS_CACHE.read_text()) if DECIMALS_CACHE.exists() else {}
    known = dict(cache.get(chain, {}))
    missing = [a for a in addrs if a.lower() not in known]
    if missing:
        rpc = rpc_url or token_meta.rpc_for(chain)
        fetched = token_meta.fetch_decimals(missing, rpc_url=rpc,
                                            batch=token_meta.rpc_batch_for(chain))
        for attempt in range(4):                  # throttling is transient; retry the stragglers
            still = [a for a in missing if a.lower() not in fetched]
            if not still:
                break
            time.sleep(3 * (attempt + 1))
            fetched.update(token_meta.fetch_decimals(still, rpc_url=rpc, batch=max(len(still), 1)))
        known.update(fetched)
        cache[chain] = known
        DECIMALS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        DECIMALS_CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")

    out, unresolved = {}, []
    for a in addrs:
        info = reg.token_info[a[2:]]
        sym = (info.symbol or "?").upper()
        d = known.get(a.lower())
        if d is None:
            unresolved.append(f"{sym} {a}")
            continue
        stable = sym in STABLE_SYMBOLS
        out[a.lower()] = (info.symbol or sym, d,
                          1.0 if stable else REFERENCE_PRICE.get(sym, 0.0), stable)
    if unresolved and strict:
        raise RuntimeError(
            f"decimals unresolved for {chain}: {unresolved}. Excluding them would understate the "
            f"loss; re-run to retry the RPC, or add them to {DECIMALS_CACHE.name}.")
    return out
