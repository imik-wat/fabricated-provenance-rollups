"""Fetch ERC-20 name()/symbol() via JSON-RPC eth_call — to identify FAKE tokens.

Fake-type poisoning (FTT) uses a counterfeit contract whose name/symbol impersonate a legit
token. The pulled logs carry no metadata, so we query it on-chain. Batched eth_call over a
public RPC; handles both string and legacy bytes32 return encodings.

name()   selector 0x06fdde03 ; symbol() selector 0x95d89b41
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ZKSYNC_RPC = "https://mainnet.era.zksync.io"
ARBITRUM_RPC = "https://arb1.arbitrum.io/rpc"
OPTIMISM_RPC = "https://mainnet.optimism.io"
BASE_RPC = "https://mainnet.base.org"
# Polygon zkEVM's sequencer was shut down 1-3 Jul 2026, but the RPC still serves historical state, so
# metadata for its contracts remains resolvable (verified: name()/symbol() answered in 0.2 s).
POLYGON_ZKEVM_RPC = "https://zkevm-rpc.com"
# Ethereum mainnet: used by a separate analysis, not part of
# the five-chain rollup study. Measured reachable and returning chainId 0x1 (2026-08-24);
# eth.llamarpc.com returned HTTP 521 and rpc.ankr.com/eth returned no chainId at all, so neither
# is listed -- an endpoint that answers without identifying its network is exactly how a metadata
# sweep ends up describing the wrong chain.
ETHEREUM_RPC = "https://ethereum-rpc.publicnode.com"
CHAIN_RPC = {"zksync-era": ZKSYNC_RPC, "arbitrum-one": ARBITRUM_RPC,
             "optimism": OPTIMISM_RPC, "base": BASE_RPC,
             "polygon-zkevm": POLYGON_ZKEVM_RPC, "eth": ETHEREUM_RPC}
# Public endpoints rate-limit per-IP and the official one is usually the strictest. Measured on
# Base: mainnet.base.org answered "over rate limit" (-32016) to every call while three community
# endpoints answered the identical batch correctly. Rotating is what makes a large sweep finish.
CHAIN_RPC_FALLBACKS = {
    "base": ["https://base-rpc.publicnode.com", "https://1rpc.io/base", "https://base.drpc.org"],
    "optimism": ["https://optimism-rpc.publicnode.com", "https://1rpc.io/op"],
    "arbitrum-one": ["https://arbitrum-one-rpc.publicnode.com"],
    # Measured response times: drpc 0.7 s, 1rpc 10.7 s. rpc.ankr.com/polygon_zkevm returns empty
    # bodies and is deliberately excluded -- an empty body is the failure mode that reads as "this
    # contract has no name" rather than as an error.
    "polygon-zkevm": ["https://polygon-zkevm.drpc.org", "https://1rpc.io/polygon/zkevm"],
    "eth": ["https://cloudflare-eth.com", "https://eth.drpc.org", "https://1rpc.io/eth"],
}


def rpc_chain_for(chain: str) -> list[str]:
    """Primary endpoint followed by fallbacks, in preference order."""
    return [rpc_for(chain)] + CHAIN_RPC_FALLBACKS.get(chain, [])
# Browser UA: some public RPCs (Arbitrum arb1 behind Cloudflare) 403 the default urllib agent.
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
NAME_SELECTOR = "0x06fdde03"
SYMBOL_SELECTOR = "0x95d89b41"
DECIMALS_SELECTOR = "0x313ce567"


def fetch_decimals(contracts: list[str], rpc_url: str, batch: int = 25) -> dict[str, int]:
    """On-chain ERC-20 decimals() per contract.

    Value decoding needs the exponent, and it is chain- and token-specific (USDC is 6 on one
    chain, 18 elsewhere); reading it from the contract removes a class of silent, order-of-
    magnitude errors that a hardcoded table invites. Contracts that do not answer are omitted, so
    callers can fail loudly rather than assume 18.
    """
    out: dict[str, int] = {}
    for i in range(0, len(contracts), batch):
        chunk = contracts[i:i + batch]
        calls = [{"jsonrpc": "2.0", "id": f"d{j}", "method": "eth_call",
                  "params": [{"to": a, "data": DECIMALS_SELECTOR}, "latest"]}
                 for j, a in enumerate(chunk)]
        by_id = {r.get("id"): r for r in _rpc_batch(calls, rpc_url) if isinstance(r, dict)}
        for j, a in enumerate(chunk):
            res = (by_id.get(f"d{j}") or {}).get("result")
            if isinstance(res, str) and len(res) >= 3:
                try:
                    out[a.lower()] = int(res, 16)
                except ValueError:
                    pass
    return out


# Max JSON-RPC batch size per chain. Public endpoints differ: Arbitrum's arb1 happily answers 25
# contracts per request, whereas Optimism's silently returns an EMPTY body above ~5 -- no error, no
# partial result, which reads as "this chain has no fake tokens" rather than as a failure. Sizes are
# measured, not assumed (batch<=5 gives 100% coverage on Optimism, batch>=10 gives 0%).
CHAIN_RPC_BATCH = {"arbitrum-one": 25, "zksync-era": 25, "optimism": 5, "eth": 10}
DEFAULT_RPC_BATCH = 5          # conservative: correctness over speed on an unknown endpoint


def rpc_for(chain: str) -> str:
    """JSON-RPC endpoint for a chain's eth_call metadata queries.

    Raises KeyError for an unknown chain. It previously defaulted to the zkSync Era endpoint, which
    made a missing entry silently query THE WRONG CHAIN: because `polygon-zkevm` was absent, a
    Polygon zkEVM sweep asked zkSync Era about Polygon contracts and got 40/40 successful responses
    with 0 names and 0 ERC-165 hits -- indistinguishable from the true finding "these contracts have
    no metadata". Every request succeeded, so no retry, coverage guard or rpc_error path fired.

    A wrong answer that looks like a finding is the failure class this module's guards exist to catch,
    so an unknown chain must fail loudly instead. Callers wanting the permissive behaviour should ask
    for it explicitly via CHAIN_RPC.get().
    """
    try:
        return CHAIN_RPC[chain]
    except KeyError:
        raise KeyError(
            f"no JSON-RPC endpoint configured for chain {chain!r}; "
            f"known chains: {sorted(CHAIN_RPC)}. Add an entry to CHAIN_RPC rather than letting the "
            f"call fall through to another chain's endpoint."
        ) from None


def rpc_batch_for(chain: str) -> int:
    """Largest batch the chain's endpoint actually answers (see CHAIN_RPC_BATCH)."""
    return CHAIN_RPC_BATCH.get(chain, DEFAULT_RPC_BATCH)


def _decode_string(hexret: str) -> str:
    if not hexret or hexret == "0x":
        return ""
    b = bytes.fromhex(hexret[2:])
    # dynamic string: [offset(32)][length(32)][bytes]
    if len(b) >= 64:
        offset = int.from_bytes(b[:32], "big")
        if offset == 32 and len(b) >= 64:
            length = int.from_bytes(b[32:64], "big")
            if 0 <= length <= len(b) - 64:
                return b[64:64 + length].decode("utf-8", "ignore").rstrip("\x00").strip()
    # legacy bytes32
    return b.rstrip(b"\x00").decode("utf-8", "ignore").strip()


def _rpc_batch(calls: list[dict], rpc_url: str, retries: int = 6) -> list[dict]:
    payload = json.dumps(calls).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(rpc_url, data=payload,
                                         headers={"Content-Type": "application/json",
                                                  "User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                resp = json.loads(r.read())
            if isinstance(resp, list) and resp:  # non-empty batch response
                return resp
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError):
            pass
        time.sleep(1.5 * (attempt + 1))  # backoff on failure/empty (rate limit)
    return []


def _sweep_metadata(contracts: list[str], rpc_url: str, batch: int,
                    sleep_s: float) -> dict[str, dict]:
    """One pass of name/symbol eth_calls. Unresolved addresses come back as empty strings."""
    out: dict[str, dict] = {}
    for i in range(0, len(contracts), batch):
        chunk = contracts[i:i + batch]
        calls = []
        for j, addr in enumerate(chunk):
            calls.append({"jsonrpc": "2.0", "id": f"n{j}", "method": "eth_call",
                          "params": [{"to": addr, "data": NAME_SELECTOR}, "latest"]})
            calls.append({"jsonrpc": "2.0", "id": f"s{j}", "method": "eth_call",
                          "params": [{"to": addr, "data": SYMBOL_SELECTOR}, "latest"]})
        by_id = {r.get("id"): r for r in _rpc_batch(calls, rpc_url) if isinstance(r, dict)}
        for j, addr in enumerate(chunk):
            rn, rs = by_id.get(f"n{j}", {}), by_id.get(f"s{j}", {})
            n, s = rn.get("result"), rs.get("result")
            rec = {"name": _decode_string(n) if n else "",
                   "symbol": _decode_string(s) if s else ""}
            # An RPC-level error is NOT evidence that the contract lacks name()/symbol().
            # Conflating the two is what let a rate-limited Base sweep report 3.6% coverage as
            # though 96% of tokens were nameless.
            err = (rn.get("error") or rs.get("error") or {}).get("message")
            if err and not (rec["name"] or rec["symbol"]):
                rec["rpc_error"] = err
            out[addr.lower()] = rec
        time.sleep(sleep_s)
    return out


def fetch_metadata(contracts: list[str], rpc_url: str = ZKSYNC_RPC,
                   batch: int = 25, sleep_s: float = 0.4,
                   retries: int = 4, chain: str | None = None) -> dict[str, dict]:
    """address -> {'name':..., 'symbol':...}, resilient to a throttling public RPC.

    A single sweep is not trustworthy: a throttled endpoint returns an empty body, which this
    layer cannot distinguish from a contract that genuinely implements neither name() nor
    symbol(). On Base a 5000-token sweep resolved 3.6% -- enough to make the impersonator set,
    and therefore an FTT result, an artifact of RPC luck. So unresolved addresses are RETRIED
    with shrinking batches and growing backoff.

    When `chain` is given, resolved entries are cached on disk across runs. Only NON-EMPTY
    results are cached: caching an empty one would bake a throttle artifact in permanently,
    whereas re-querying a genuinely nameless contract each run is merely a little wasted work.
    """
    cached = _metadata_cache(chain) if chain else {}
    todo = [c for c in contracts if c.lower() not in cached]
    out = {c.lower(): dict(cached[c.lower()]) for c in contracts if c.lower() in cached}
    if not todo:
        return out

    endpoints = rpc_chain_for(chain) if chain else [rpc_url]
    if rpc_url and rpc_url not in endpoints:
        endpoints = [rpc_url] + endpoints

    out.update(_sweep_metadata(todo, endpoints[0], batch, sleep_s))
    for attempt in range(retries):
        missing = [a for a in todo if not (out.get(a.lower(), {}).get("name")
                                           or out.get(a.lower(), {}).get("symbol"))]
        if not missing:
            break
        # Rotate endpoints rather than only backing off. When the failures are explicit
        # rate-limit errors, waiting longer on the SAME endpoint does not help -- a different
        # one answers the identical batch immediately.
        url = endpoints[(attempt + 1) % len(endpoints)]
        time.sleep(2.0 * (attempt + 1) if url == endpoints[0] else 0.5)
        # Shrink the batch each round: throttling is usually per-request-size, so a batch the
        # endpoint refused at 5 may well be answered at 2.
        smaller = max(1, batch // (2 ** (attempt + 1)))
        out.update({k: v for k, v in
                    _sweep_metadata(missing, url, smaller, sleep_s * (attempt + 2)).items()
                    if v.get("name") or v.get("symbol")})

    if chain:
        _metadata_cache_save(chain, {k: v for k, v in out.items()
                                     if v.get("name") or v.get("symbol")})
    return out


METADATA_CACHE = Path(__file__).resolve().parents[1] / "config" / "token_metadata.json"


def _metadata_cache(chain: str) -> dict[str, dict]:
    if not METADATA_CACHE.exists():
        return {}
    return json.loads(METADATA_CACHE.read_text()).get(chain, {})


def _metadata_cache_save(chain: str, resolved: dict[str, dict]) -> None:
    all_ = json.loads(METADATA_CACHE.read_text()) if METADATA_CACHE.exists() else {}
    all_.setdefault(chain, {}).update(resolved)
    METADATA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    METADATA_CACHE.write_text(json.dumps(all_, indent=2, sort_keys=True) + "\n")


def coverage(metadata: dict[str, dict]) -> float:
    """Fraction of contracts with any name/symbol resolved (metadata reliability check)."""
    if not metadata:
        return 0.0
    ok = sum(1 for m in metadata.values() if m["name"] or m["symbol"])
    return ok / len(metadata)


import re


def find_impersonators(metadata: dict[str, dict], legit_names: set[str],
                       legit_symbols: set[str], legit_addresses: set[str]) -> dict[str, dict]:
    """Contracts impersonating a legit token by name/symbol, not being the legit contract.

    Matches (a) an EXACT legit name/symbol, or (b) a DECORATED variant — a legit symbol
    appearing as a standalone word in the name/symbol (e.g. "zkSync (ZK)", "USD Coin
    (claim at ...)") — which is how real fake/phishing tokens disguise the impersonation.
    Each fake carries a `match` field: 'exact' or 'decorated'.
    """
    fakes = {}
    for addr, m in metadata.items():
        if addr in legit_addresses:
            continue
        name, sym = m["name"].casefold(), m["symbol"].casefold()
        if name in legit_names or sym in legit_symbols:
            fakes[addr] = {**m, "match": "exact"}
            continue
        words = set(re.findall(r"[a-z0-9]+", name)) | set(re.findall(r"[a-z0-9]+", sym))
        if words & legit_symbols:
            fakes[addr] = {**m, "match": "decorated"}
    return fakes
