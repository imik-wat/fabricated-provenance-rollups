"""Deployed bytecode for a set of contracts, via ``eth_getCode``, with the throttling guards.

Why this is a sibling of ``token_meta`` and not a function inside it
-------------------------------------------------------------------
``token_meta`` answers "what does this token call itself" with ``eth_call``. This answers "what code
is actually deployed here" with ``eth_getCode``. The transport plumbing is genuinely shared and is
imported rather than restated -- endpoint rotation, browser user-agent, batching, backoff -- but the
*semantics of an empty answer* are different, and that difference is the reason for a separate module.

THE DANGEROUS CASE, and how it differs from token_meta's
--------------------------------------------------------
For metadata, an empty string means "no ``name()``" and a throttled endpoint also looks empty, so
``token_meta`` caches only NON-EMPTY results: caching an empty one would bake a throttle artefact in
permanently.

For code, ``"0x"`` is not a missing answer -- it is a *positive claim that the address holds no code*,
i.e. that it is an EOA or a self-destructed contract. Accepting that claim from a throttled endpoint
would classify a live contract as an EOA and erase the batching-contract population from the study.
So this module treats ``"0x"`` as **provisional**: any address answering ``"0x"`` is re-queried on a
DIFFERENT endpoint before the answer is accepted, and the count of addresses that answer ``"0x"``
everywhere is reported prominently rather than folded into the results. For the populations this
study asks about -- token contracts that demonstrably emitted Transfer logs -- ``"0x"`` should be
impossible, so a non-zero count is a finding or a fault, never a detail.

An RPC-level error is recorded as ``rpc_error`` and is never cached, never counted as ``"0x"``, and
never allowed to look like an answer.

Failure modes (all loud)
------------------------
- An unknown chain raises via ``token_meta.rpc_for``: no falling through to another network.
- ``fetch_code`` returns a record for EVERY requested address, so a caller can always reconcile
  counts; nothing is dropped.
- ``require_coverage`` raises below a caller-declared floor, matching the ``run_ftt --min-coverage``
  idiom: a sweep that resolved a fraction of its targets must not reach an aggregate.

Complexity: O(addresses / batch) round trips per endpoint pass.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from .token_meta import _rpc_batch, rpc_batch_for, rpc_chain_for

#: Cache of resolved code digests. Keyed by chain then address. Only DEFINITE answers are stored;
#: see the module docstring for why "0x" is not automatically definite.
CODE_CACHE = Path(__file__).resolve().parents[1] / "config" / "contract_code.json"

EMPTY_CODE = "0x"
RPC_ERROR = "rpc_error"

#: EIP-1167 minimal proxy. The runtime is exactly 45 bytes and embeds the implementation address in
#: the middle, so the delegate target is recoverable from the code alone -- no storage read, no ABI.
#: This matters for counterfeit-token morphology: a proxy's code hash describes the PROXY, not the
#: token logic, so a family of proxies will never share a hash with a directly deployed token even
#: when they behave identically. Reading a zero overlap as "different programs" without checking for
#: proxies would be the wrong conclusion from a correct measurement.
EIP1167_PREFIX = "363d3d373d3d3d363d73"
EIP1167_SUFFIX = "5af43d82803e903d91602b57fd5bf3"


def detect_eip1167(code_hex: str) -> str | None:
    """The implementation address an EIP-1167 minimal proxy delegates to, or None.

    Pure function of the bytecode. Returns a 0x-prefixed lowercase address.
    """
    b = (code_hex[2:] if code_hex.startswith("0x") else code_hex).lower()
    if len(b) != 90 or not b.startswith(EIP1167_PREFIX) or not b.endswith(EIP1167_SUFFIX):
        return None
    return "0x" + b[len(EIP1167_PREFIX):len(EIP1167_PREFIX) + 40]


def _cache_load() -> dict:
    if CODE_CACHE.exists():
        try:
            return json.loads(CODE_CACHE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _cache_save(chain: str, resolved: dict[str, dict]) -> None:
    data = _cache_load()
    data.setdefault(chain, {}).update(resolved)
    CODE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    CODE_CACHE.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


def code_digest(code_hex: str) -> str:
    """SHA-256 of the deployed bytecode. Identical digests mean byte-identical deployments.

    A digest rather than the code itself: the code is up to 24 KB per contract and the question is
    only ever "is this the same program", so storing 64 hex characters keeps the artefact small
    enough to commit while remaining exactly as discriminating.
    """
    body = code_hex[2:] if code_hex.startswith("0x") else code_hex
    return hashlib.sha256(bytes.fromhex(body)).hexdigest() if body else ""


def _sweep(addresses: list[str], rpc_url: str, batch: int, sleep_s: float) -> dict[str, dict]:
    """One pass of eth_getCode against one endpoint."""
    out: dict[str, dict] = {}
    for i in range(0, len(addresses), batch):
        chunk = addresses[i:i + batch]
        calls = [{"jsonrpc": "2.0", "id": f"c{j}", "method": "eth_getCode",
                  "params": [a, "latest"]} for j, a in enumerate(chunk)]
        by_id = {r.get("id"): r for r in _rpc_batch(calls, rpc_url) if isinstance(r, dict)}
        for j, a in enumerate(chunk):
            r = by_id.get(f"c{j}", {})
            res, err = r.get("result"), (r.get("error") or {}).get("message")
            if isinstance(res, str) and res.startswith("0x"):
                rec = {"code_len_bytes": max(0, (len(res) - 2) // 2),
                       "code_hash": code_digest(res),
                       "empty": res == EMPTY_CODE, "endpoint": rpc_url}
                impl = detect_eip1167(res)
                if impl:
                    rec["proxy_standard"] = "EIP-1167"
                    rec["proxy_implementation"] = impl
                out[a.lower()] = rec
            else:
                out[a.lower()] = {RPC_ERROR: err or "no result", "endpoint": rpc_url}
        time.sleep(sleep_s)
    return out


def fetch_code(addresses: list[str], chain: str, sleep_s: float = 0.25,
               use_cache: bool = True, verify_empty: bool = True) -> dict[str, dict]:
    """address -> {code_len_bytes, code_hash, empty} or {rpc_error}. One record per input address.

    Endpoints are rotated on failure, and -- when `verify_empty` -- every "0x" answer is re-asked on
    a different endpoint. A rate-limited endpoint is not a slow one: it answers instantly and wrongly,
    which is why rotation rather than backoff alone is the remedy.
    """
    want = [a.lower() for a in dict.fromkeys(a.lower() for a in addresses)]
    cache = _cache_load().get(chain, {}) if use_cache else {}
    out = {a: dict(cache[a]) for a in want if a in cache}
    todo = [a for a in want if a not in out]

    endpoints = rpc_chain_for(chain)
    batch = rpc_batch_for(chain)
    for ep in endpoints:
        if not todo:
            break
        got = _sweep(todo, ep, batch, sleep_s)
        for a, rec in got.items():
            if RPC_ERROR not in rec:
                out[a] = rec
        todo = [a for a in todo if a not in out]

    # "0x" is a positive claim that the address holds no code. Confirm it somewhere else before
    # believing it, because a throttled endpoint produces the same string.
    if verify_empty:
        empties = [a for a, r in out.items() if r.get("empty")]
        for a in empties:
            first = out[a].get("endpoint")
            for ep in [e for e in endpoints if e != first]:
                rec = _sweep([a], ep, 1, sleep_s).get(a, {})
                if RPC_ERROR in rec:
                    continue
                out[a]["confirmed_empty_on"] = ep
                if not rec.get("empty"):
                    # The first endpoint lied. Take the answer that found code.
                    out[a] = rec | {"first_endpoint_reported_empty": first}
                break

    for a in todo:
        out[a] = {RPC_ERROR: "unresolved on every endpoint"}

    if use_cache:
        # Cache definite answers only. An "0x" is definite only once confirmed elsewhere.
        _cache_save(chain, {a: r for a, r in out.items()
                            if RPC_ERROR not in r
                            and (not r.get("empty") or r.get("confirmed_empty_on"))})
    return {a: out[a] for a in want}


def coverage(records: dict[str, dict]) -> float:
    """Share of addresses with a definite answer. Errors do not count as answers."""
    if not records:
        return 0.0
    return sum(1 for r in records.values() if RPC_ERROR not in r) / len(records)


def require_coverage(records: dict[str, dict], minimum: float, label: str) -> None:
    """Raise below `minimum`, matching the run_ftt --min-coverage idiom.

    A partially-resolved sweep must not reach an aggregate: a 3.6%-coverage Base metadata sweep once
    nearly published a one-token "fake-token ecosystem".
    """
    c = coverage(records)
    if c < minimum:
        raise RuntimeError(
            f"{label}: eth_getCode coverage {c:.4f} below the required {minimum:.4f} "
            f"({sum(1 for r in records.values() if RPC_ERROR in r)} of {len(records)} unresolved). "
            f"Refusing to characterise bytecode families from a partial sweep.")


def families(records: dict[str, dict]) -> dict:
    """Group resolved addresses by code digest.

    Returns the family sizes and the members of each, largest first. Addresses that returned an
    error are listed separately and never silently merged into a family, and addresses with NO code
    are their own reported category because for a token contract that answer should be impossible.
    """
    by_hash: dict[str, list[str]] = {}
    unresolved, empty = [], []
    for a, r in sorted(records.items()):
        if RPC_ERROR in r:
            unresolved.append(a)
        elif r.get("empty"):
            empty.append(a)
        else:
            by_hash.setdefault(r["code_hash"], []).append(a)
    ordered = sorted(by_hash.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    resolved = sum(len(v) for v in by_hash.values())
    proxies: dict[str, list[str]] = {}
    for a, r in sorted(records.items()):
        if r.get("proxy_implementation"):
            proxies.setdefault(r["proxy_implementation"], []).append(a)
    return {
        "resolved_with_code": resolved,
        "distinct_code_hashes": len(by_hash),
        "largest_family": len(ordered[0][1]) if ordered else 0,
        "largest_family_share": round(len(ordered[0][1]) / resolved, 6) if resolved else None,
        "singletons": sum(1 for _, v in ordered if len(v) == 1),
        "families": [{"code_hash": h, "n": len(v),
                      "code_len_bytes": records[v[0]]["code_len_bytes"], "members": v[:50],
                      "members_truncated": max(0, len(v) - 50)} for h, v in ordered],
        "proxies": {
            "n_proxy_contracts": sum(len(v) for v in proxies.values()),
            "distinct_implementations": len(proxies),
            "by_implementation": [{"implementation": k, "n": len(v), "members": v[:50]}
                                  for k, v in sorted(proxies.items(), key=lambda kv: -len(kv[1]))],
            "why_it_matters": ("a proxy's code hash describes the proxy, not the token logic, so a "
                              "proxy family shares no hash with a directly deployed token even when "
                              "behaviour is identical; the shared implementation is the real family"),
        },
        "no_code_addresses": empty,
        "unresolved_addresses": unresolved,
        "no_code_note": ("an address that emitted Transfer logs cannot hold no code; a non-empty "
                         "list here is a finding (self-destructed) or a fault, never a detail"),
    }
