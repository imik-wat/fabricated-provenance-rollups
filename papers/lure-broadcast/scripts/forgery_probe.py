"""Establish, contract by contract, that the lure channel's transfer events are fabricated.

The argument, and why it is THIS argument
-----------------------------------------
An earlier version of this analysis proposed a conservation test: an address cannot send a token it
never received or had minted. That test is INADMISSIBLE and is not implemented here. A control on
registry-verified tokens reproduces the same "violation": Base WETH shows thousands of addresses
with negative implied balance, because the corpus holds only `Transfer` events while bridging and
constructor minting move balances without emitting one. Legitimate tokens fail it too, so it
separates nothing. See docs/FINDINGS.md.

What does work is a positive statement about contract STATE, which needs no assumption about the
completeness of the event log:

  LEDGER TEST. Ask the contract for `balanceOf(a)` at two addresses generated at random. A contract
  holding a real per-address ledger returns zero for an address it has never seen. A contract that
  returns the SAME NON-ZERO value for two unrelated addresses exposes no per-address mapping at
  all -- so no event it emits can correspond to a balance movement, whoever is named in it.

  AUTHORISATION TEST. `allowance(owner, spender)` is the only mechanism by which a third party may
  legitimately appear in a transfer's sender field. If the call does not return an allowance, that
  mechanism is unusable and the sender field cannot be explained by a delegated transfer. We
  distinguish a REVERT (the function exists and refused) from an EMPTY return (no return data at
  all, which is what an address with no code produces), because only the former is evidence about
  an implemented function and the two support different sentences.

Controls are mandatory, not optional: the probe is only evidence if it discriminates. Registry-
verified tokens are probed identically and must return zero. A probe that flagged everything would
be measuring the probe, which is exactly the error the conservation test made.

Why the block height is pinned
------------------------------
An earlier version issued every call at `"latest"`. Contract state moves, so those results were not
reproducible: a reader re-running the probe would query a different world state and could legitimately
obtain different verdicts. Calls are now pinned to the LAST BLOCK OF THE CORPUS for the chain, derived
from the chunk directory names rather than hardcoded, so the state probed is the state as of the data
analysed elsewhere in the paper.

Pinning requires archive state. Measured on Base at the corpus end block: `mainnet.base.org` and
`base.drpc.org` serve it, `base-rpc.publicnode.com` does not respond and `1rpc.io/base` answers
"state at block #49743281 is pruned". Endpoints are therefore canaried for ARCHIVE capability at the
pinned block and the non-archive ones are excluded; if none qualifies the run raises rather than
silently falling back to `"latest"`, which would reintroduce the irreproducibility.

Why a revert must not be swallowed
----------------------------------
For the authorisation test a revert is the POSITIVE result, so it must be distinguished from a
transport failure. `token_meta._rpc_batch` returns the raw JSON-RPC list, in which each element
carries either `result` or `error`, and returns [] only when the transport itself failed after
retries. We therefore classify FOUR outcomes -- value, revert, empty, missing (see `_outcome`) --
and never conflate them. In particular a throttled endpoint yields `missing`, which can only ever
understate fabrication, and an address with no code yields `empty`, which is not a refusal.

Verdicts
--------
constant_nonzero  both probes return the same non-zero balance -> no per-address ledger; FABRICATED
address_derived   probes differ and are non-zero -> a pseudo-balance computed from the address;
                  also fabrication, but of a different shape, so it is counted separately
zero_ledger       both probes return zero -> behaves as a real ledger would (the control outcome)
no_response       the contract did not answer after retries across all endpoints
Unresolved contracts are RETAINED in the output and excluded from coverage, never silently dropped.

Inputs
------
analysis/config/token_metadata.json     (lure classification, read-only)
analysis/src/registry.py                (control tokens)

Outputs
-------
papers/lure-broadcast/results/forgery_probe_<chain>.json (+ .manifest.json)

Failure modes
-------------
- Zero lure contracts classified -> raises rather than emitting an empty probe.
- A control returning `constant_nonzero` -> raises. That would mean the probe does not discriminate
  and no verdict in the file is trustworthy.
- Every contract unreachable -> raises rather than reporting 0 fabricated.

Complexity: 4 eth_calls per contract, batched; ~145 contracts on one chain, seconds to a minute.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analysis"))

from src import metadata_lure as ml, registry, token_meta  # noqa: E402

OUT_ROOT = ROOT / "papers" / "lure-broadcast" / "results"

BALANCE_OF = "0x70a08231"      # balanceOf(address)
TOTAL_SUPPLY = "0x18160ddd"    # totalSupply()
ALLOWANCE = "0xdd62ed3e"       # allowance(address,address)

#: Fixed so the probe addresses are reproducible: the same seed yields the same two addresses on
#: every run and in every replication. The addresses are not secret and hold no funds.
PROBE_SEED = "balance-probe"

RAW = ROOT / "data" / "raw" / "hypersync"

#: Endpoints added specifically because they serve ARCHIVE state, which the configured defaults do
#: not do reliably at the corpus end block. Measured on Base at block 49,743,280: the configured
#: primaries answer a single archive call and then throttle, so a full run returned `no_response`
#: for 128 of 141 contracts; `base-pokt.nodies.app` answers eight calls per request repeatedly.
#: Kept HERE rather than in `token_meta.CHAIN_RPC_FALLBACKS` so the shared configuration used by the
#: other paper is untouched; these are tried first and the configured endpoints remain as fallbacks.
ARCHIVE_ENDPOINTS = {
    "base": ["https://base-pokt.nodies.app"],
    # Measured 2026-10-27 at Arbitrum's corpus end block 491,701,806. Neither configured endpoint
    # serves archive state there, nor do drpc, llamarpc, 1rpc or meowrpc; these two do.
    "arbitrum-one": ["https://arbitrum-one.public.blastapi.io", "https://arb-pokt.nodies.app"],
}


def corpus_end_block(chain: str) -> int:
    """Last block of the pulled corpus, from the chunk directory names.

    Derived rather than hardcoded so the pin cannot drift away from the data, and so a re-pull that
    extends the corpus moves the probe with it.
    """
    ends = []
    for d in (RAW / chain).glob("chunk_*"):
        if not (d / "_DONE").exists():
            continue
        try:
            ends.append(int(d.name.split("_")[2]))
        except (IndexError, ValueError):
            continue
    if not ends:
        raise SystemExit(f"no completed chunks for {chain}; cannot determine the corpus end block, "
                         f"and probing at 'latest' would not be reproducible")
    return max(ends)


def archive_capable(endpoints: list[str], block_hex: str, probe_addr: str) -> list[str]:
    """Keep only endpoints that serve historical state at `block_hex`.

    A pruning node answers `latest` perfectly well and fails only on the pinned block, so this cannot
    be inferred from ordinary reachability and must be tested at the block actually used.
    """
    ok = []
    for ep in endpoints:
        call = {"jsonrpc": "2.0", "id": 1, "method": "eth_call",
                "params": [{"to": probe_addr, "data": TOTAL_SUPPLY}, block_hex]}
        r = token_meta._rpc_batch([call], ep, retries=1)
        if r and isinstance(r[0], dict) and "result" in r[0] and r[0]["result"] not in (None, "0x"):
            ok.append(ep)
    return ok


def calibrate_batch(endpoint: str, block_hex: str, probe_addr: str, ladder=(8, 4, 2, 1)) -> int:
    """Largest contract-batch the endpoint answers in full at the pinned block.

    Archive `eth_call` is far more expensive than a call at `latest`, and public endpoints degrade
    on BATCH SIZE rather than refusing outright. Measured on Base at the corpus end block:
    `mainnet.base.org` answers 4 calls in full and returns nothing at 10 or above, while
    `base.drpc.org` already fails at 4. A single-call canary therefore certifies an endpoint that
    then answers nothing for the run -- which is exactly how the first pinned attempt returned
    `no_response` for 140 of 141 contracts. The workable size is measured here instead of assumed.

    Each contract costs four calls (two balances, a supply and an allowance), so a contract-batch of
    n issues 4n calls.
    """
    # Public endpoints throttle non-deterministically -- the project notes record three identical
    # Base requests resolving 5/6, 0/6, 0/6 -- so a single failed ladder is not evidence of a limit.
    # Each rung is retried with backoff before the size is ruled out.
    for n in ladder:
        calls = []
        for i in range(n):
            calls.append({"jsonrpc": "2.0", "id": f"c{i}", "method": "eth_call",
                          "params": [{"to": probe_addr, "data": TOTAL_SUPPLY}, block_hex]})
            calls.append({"jsonrpc": "2.0", "id": f"d{i}", "method": "eth_call",
                          "params": [{"to": probe_addr, "data": TOTAL_SUPPLY}, block_hex]})
            calls.append({"jsonrpc": "2.0", "id": f"e{i}", "method": "eth_call",
                          "params": [{"to": probe_addr, "data": TOTAL_SUPPLY}, block_hex]})
            calls.append({"jsonrpc": "2.0", "id": f"f{i}", "method": "eth_call",
                          "params": [{"to": probe_addr, "data": TOTAL_SUPPLY}, block_hex]})
        for attempt in range(3):
            r = token_meta._rpc_batch(calls, endpoint, retries=1)
            ok = sum(1 for x in r if isinstance(x, dict) and "result" in x)
            if ok == len(calls):
                return n
            time.sleep(2.0 * (attempt + 1))
    return 0


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def probe_addresses(seed: str, n: int = 2) -> list[str]:
    """Deterministic pseudo-random addresses derived from `seed`.

    Derivation is by SHA-256 of the seed and an index, truncated to 20 bytes. Two properties matter:
    it is reproducible from the seed alone (so a reviewer can regenerate the exact probes), and the
    addresses are not chosen with knowledge of any contract's state. Their absence from the chain is
    verified separately by forgery_chain_scan.py rather than assumed here.
    """
    out = []
    for i in range(n):
        h = hashlib.sha256(f"{seed}:{i}".encode()).hexdigest()
        out.append("0x" + h[:40])
    return out


def _word(addr: str) -> str:
    return addr[2:].lower().rjust(64, "0")


def _outcome(entry: dict | None) -> tuple[str, int | None]:
    """Map one JSON-RPC response element to (kind, value).

    kind is one of:
      'value'   a decodable return word
      'revert'  the node reported an execution error -- the function exists and refused
      'empty'   the call returned 0x, i.e. NO return data. This is what an address with no code,
                or a contract without that function, produces. It is NOT the same as a revert, and
                conflating the two would let "the approval path is not implemented" be reported for
                an address that simply has no code at all.
      'missing' no response element -- a transport failure, i.e. absence of evidence

    Keeping 'revert' and 'empty' apart is what lets the authorisation test be stated precisely.
    """
    if entry is None:
        return "missing", None
    if "error" in entry:
        return "revert", None
    r = entry.get("result")
    if r is None or r == "0x" or r == "":
        return "empty", None
    try:
        return "value", int(r, 16)
    except ValueError:
        return "missing", None


def classify_balances(kinds: list[str], values: list[int | None]) -> str:
    """The verdict rule, as a pure function.

    This is the paper's central inferential step, so it is kept separate from the network loop and
    unit-tested rather than exercised only against live endpoints.

    Truth table, given one balanceOf answer per probe address:
      any answer not a value        -> no_response   (cannot conclude; NOT evidence of a ledger)
      all answers 0                 -> zero_ledger   (behaves as a real ledger; the control outcome)
      all equal and non-zero        -> constant_nonzero  (no per-address mapping -> FABRICATED)
      unequal and non-zero          -> address_derived   (a pseudo-balance computed from the address)

    The asymmetry is deliberate: a missing answer never counts towards fabrication, so a throttled
    endpoint can only ever understate the result.
    """
    if not kinds or any(k != "value" for k in kinds):
        return "no_response"
    if any(v is None for v in values):
        return "no_response"
    if all(v == 0 for v in values):
        return "zero_ledger"
    if len(set(values)) == 1:
        return "constant_nonzero"
    return "address_derived"


def probe(contracts: list[str], endpoints: list[str], probes: list[str],
          batch: int, sleep_s: float, block_hex: str) -> dict[str, dict]:
    """Probe every contract, rotating endpoints for those that do not answer."""
    pending = list(contracts)
    results: dict[str, dict] = {}
    for ep in endpoints:
        if not pending:
            break
        still: list[str] = []
        for i in range(0, len(pending), batch):
            chunk = pending[i:i + batch]
            calls = []
            for j, a in enumerate(chunk):
                for k, p in enumerate(probes):
                    calls.append({"jsonrpc": "2.0", "id": f"b{j}_{k}", "method": "eth_call",
                                  "params": [{"to": a, "data": BALANCE_OF + _word(p)}, block_hex]})
                calls.append({"jsonrpc": "2.0", "id": f"t{j}", "method": "eth_call",
                              "params": [{"to": a, "data": TOTAL_SUPPLY}, block_hex]})
                calls.append({"jsonrpc": "2.0", "id": f"a{j}", "method": "eth_call",
                              "params": [{"to": a,
                                          "data": ALLOWANCE + _word(probes[0]) + _word(probes[1])},
                                         block_hex]})
            by_id = {r.get("id"): r for r in token_meta._rpc_batch(calls, ep)
                     if isinstance(r, dict)}
            for j, a in enumerate(chunk):
                bal_kinds, bal_vals = [], []
                for k in range(len(probes)):
                    kind, val = _outcome(by_id.get(f"b{j}_{k}"))
                    bal_kinds.append(kind)
                    bal_vals.append(val)
                if all(k == "missing" for k in bal_kinds):
                    still.append(a)
                    continue
                ts_kind, ts_val = _outcome(by_id.get(f"t{j}"))
                al_kind, _ = _outcome(by_id.get(f"a{j}"))

                verdict = classify_balances(bal_kinds, bal_vals)

                results[a] = {
                    "verdict": verdict,
                    "balances": bal_vals,
                    "balance_kinds": bal_kinds,
                    "total_supply": ts_val,
                    "balance_equals_total_supply": (verdict == "constant_nonzero"
                                                    and ts_val is not None
                                                    and bal_vals[0] == ts_val),
                    "allowance": al_kind,
                    "endpoint": ep,
                }
            time.sleep(sleep_s)
        pending = still
    for a in pending:
        results[a] = {"verdict": "no_response", "balances": [], "balance_kinds": [],
                      "total_supply": None, "balance_equals_total_supply": False,
                      "allowance": "missing", "endpoint": None}
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chain", default="base")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--sleep", type=float, default=0.6,
                    help="pause between batches; archive calls throttle more readily than latest")
    ap.add_argument("--block", type=int, default=0,
                    help="pin calls to this block; 0 = the corpus end block for the chain")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = OUT_ROOT / f"forgery_probe_{args.chain}.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out.relative_to(ROOT)} exists; pass --force")

    md = json.loads((ROOT / "analysis" / "config" /
                     "token_metadata.json").read_text())[args.chain]
    lure = sorted({a.lower() for a, m in md.items()
                   if ml.classify(m.get("name"), m.get("symbol")).strength == "strong"})
    if not lure:
        raise SystemExit("no lure contracts classified -- refusing to emit an empty probe")

    # Controls: registry-verified tokens. The probe must return zero for these or it discriminates
    # nothing. Registry addresses are stored WITHOUT the 0x prefix (INTEGRATION_NOTES 3b).
    reg = sorted("0x" + a.lower() for a in registry.load_token_registry(args.chain).legit_addresses)
    controls = [a for a in reg if a in md] or reg[:8]
    controls = [a for a in controls if a not in set(lure)][:8]
    if not controls:
        raise SystemExit("no control tokens available -- the probe would be uncontrolled")

    probes = probe_addresses(PROBE_SEED)
    block = args.block or corpus_end_block(args.chain)
    block_hex = hex(block)
    all_eps = ARCHIVE_ENDPOINTS.get(args.chain, []) + token_meta.rpc_chain_for(args.chain)
    eps = archive_capable(all_eps, block_hex, controls[0])
    if not eps:
        raise SystemExit(f"no endpoint serves archive state for {args.chain} at block {block:,}. "
                         f"Falling back to 'latest' would make the probe irreproducible, so this "
                         f"aborts instead. Supply an archive endpoint, or pass --block.")
    print(f"  {args.chain}: {len(lure)} lure contracts, {len(controls)} controls")
    print(f"  pinned at block {block:,} ({block_hex}), the corpus end; "
          f"{len(eps)} of {len(all_eps)} endpoints serve archive state there")
    cal = {e: calibrate_batch(e, block_hex, controls[0]) for e in eps}
    eps = [e for e in eps if cal[e] > 0]
    if not eps:
        raise SystemExit(f"no archive endpoint answers even a single-contract batch at block "
                         f"{block:,}; aborting rather than reporting no_response as evidence")
    eps.sort(key=lambda e: -cal[e])
    batch = min(args.batch, cal[eps[0]])
    print(f"  archive batch calibration {cal}; using contract-batch {batch}")
    print(f"  probe addresses (seed {PROBE_SEED!r}): {probes[0]}, {probes[1]}")

    t0 = time.time()
    res_lure = probe(lure, eps, probes, batch, args.sleep, block_hex)
    res_ctrl = probe(controls, eps, probes, batch, args.sleep, block_hex)
    el = time.time() - t0

    def tally(d: dict[str, dict]) -> dict[str, int]:
        t: dict[str, int] = {}
        for v in d.values():
            t[v["verdict"]] = t.get(v["verdict"], 0) + 1
        return dict(sorted(t.items()))

    tl, tc = tally(res_lure), tally(res_ctrl)

    # --- the guards that make the result admissible -----------------------------------------
    bad = [a for a, v in res_ctrl.items() if v["verdict"] in ("constant_nonzero", "address_derived")]
    if bad:
        raise SystemExit(f"CONTROL FAILURE: {len(bad)} registry-verified token(s) look fabricated "
                         f"({bad[:3]}). The probe does not discriminate, so no verdict in this run "
                         f"is trustworthy. Investigate before publishing anything from it.")
    answered = sum(n for k, n in tl.items() if k != "no_response")
    if answered == 0:
        raise SystemExit("no lure contract answered on any endpoint -- refusing to report 0 "
                         "fabricated, which would be indistinguishable from a network failure")

    fabricated = [a for a, v in res_lure.items() if v["verdict"] == "constant_nonzero"]
    allow_kinds: dict[str, int] = {}
    for v in res_lure.values():
        allow_kinds[v["allowance"]] = allow_kinds.get(v["allowance"], 0) + 1
    # "no usable approval path" = the call reverted OR returned no data. Reported together AND
    # separately, because only the first is evidence about an implemented-but-refusing function.
    reverting = allow_kinds.get("revert", 0)
    no_allowance = reverting + allow_kinds.get("empty", 0)
    eq_supply = sum(1 for v in res_lure.values() if v["balance_equals_total_supply"])

    doc = {
        "chain": args.chain,
        "argument": ("Ledger test: a contract returning the SAME NON-ZERO balanceOf for two "
                     "randomly generated addresses exposes no per-address ledger, so no event it "
                     "emits can correspond to a balance movement. Authorisation test: a reverting "
                     "allowance() means the delegated-transfer path that could legitimately place a "
                     "third party in the sender field is not implemented."),
        "inadmissible_test_not_used": ("conservation of supply: a reproduced control shows "
                                       "registry-verified tokens violate it too, because the corpus "
                                       "holds only Transfer events while bridging and constructor "
                                       "minting move balances silently. See docs/FINDINGS.md."),
        "block": block,
        "block_hex": block_hex,
        "block_note": ("calls are pinned to the last block of the corpus for this chain, so the "
                       "state probed is the state as of the analysed data; endpoints without "
                       "archive state at that block are excluded rather than downgraded to 'latest'"),
        "archive_endpoints": eps,
        "archive_batch_calibration": cal,
        "contract_batch_used": batch,
        "endpoints_without_archive_state": [e for e in all_eps if e not in eps],
        "selectors": {"balanceOf(address)": BALANCE_OF, "totalSupply()": TOTAL_SUPPLY,
                      "allowance(address,address)": ALLOWANCE},
        "probe_seed": PROBE_SEED,
        "probe_addresses": probes,
        "probe_absence_verified_by": ("scripts/forgery_chain_scan.py -- absence from the chain is "
                                      "measured there, not assumed here"),
        "wall_seconds": round(el, 1),
        "lure": {"n": len(lure), "verdicts": tl,
                 "fabricated_constant_nonzero": len(fabricated),
                 "balance_equals_total_supply": eq_supply,
                 "allowance_reverts": reverting,
                 "allowance_kinds": dict(sorted(allow_kinds.items())),
                 "allowance_unusable_revert_or_empty": no_allowance,
                 "ledger_and_authorisation_both": len(
                     [a for a, v in res_lure.items()
                      if v["verdict"] == "constant_nonzero"
                      and v["allowance"] in ("revert", "empty")]),
                 "ledger_only": len(
                     [a for a, v in res_lure.items()
                      if v["verdict"] == "constant_nonzero"
                      and v["allowance"] == "value"])},
        "controls": {"n": len(controls), "verdicts": tc, "addresses": controls},
        "per_contract": res_lure,
        "per_control": res_ctrl,
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    (OUT_ROOT / f"forgery_probe_{args.chain}.manifest.json").write_text(json.dumps({
        "producer": "papers/lure-broadcast/scripts/forgery_probe.py",
        "git_commit": git_commit(), "probe_seed": PROBE_SEED,
    }, indent=1) + "\n")

    print(f"  lure verdicts    {tl}")
    print(f"  control verdicts {tc}   <- must be zero_ledger only")
    print(f"  balanceOf == totalSupply on {eq_supply} contracts")
    print(f"  allowance kinds  {dict(sorted(allow_kinds.items()))} "
          f"(revert={reverting}, revert-or-empty={no_allowance})")
    print(f"  -> {out.relative_to(ROOT)}  ({el:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
