"""Compare collection-time contract metadata with the metadata reported at historical block heights.

Purpose
-------
The transfer corpus is a complete history, but `name()` and `symbol()` are read once, over RPC, at
collection time. Nothing in that arrangement prevents a contract having reported different values
when the events were emitted years earlier, and if it did, campaign assignment, detector status and
every per-campaign count would inherit the error. The manuscript previously stated the risk without
testing it.

The question is settled by measurement rather than by restricting the estimand. `name()` and `symbol()` are called
again at HISTORICAL block heights inside each contract's own emission window and compared with the
collection-time value. Where the answers agree, the attribution is not an assumption; where they
disagree, the contract is named and excluded from any historical reading.

Choice of reference value
-------------------------
The historical readings are compared against a reading taken by THIS code at the corpus end block,
not against the stored `token_metadata.json`. The first version compared against the stored file and
reported three contracts as having changed their metadata. None of them had. All three differed at
the end block too, which is the tell: two by a leading space the collection step had stripped, and
one by a symbol the collection step had recorded as empty while the contract returns a string. Those
are collection-time capture differences, and attributing them to the passage of time would have put
a false finding in the paper on the exact question a reviewer had raised.

Comparing like with like isolates the temporal question. The stored-versus-chain difference is a
real defect and is still reported, separately, as `registry_mismatch` -- it matters because the
detector reads the stored strings, so a stripped or missing one is a detector input, not a cosmetic
issue.

Heights probed, per contract, all derived and none hardcoded:
  FIRST  the first block of the contract's campaign, the earliest point the window reaches;
  LAST   the last block of that campaign, the latest point at which it was still emitting;
  END    the corpus end block, which is what `token_metadata.json` effectively reflects.

A contract deployed part-way through its campaign has no code at FIRST. That is recorded as
`no_code_at_height` and is NOT a disagreement -- absence of a contract is not a change of metadata,
and conflating the two would manufacture exactly the finding this script exists to test for.

Inputs
------
analysis/config/token_metadata.json                          (collection-time values)
papers/lure-broadcast/results/campaign_census_<chain>.json    (per-campaign block windows)
data/raw/hypersync/<chain>/chunk_*                            (corpus end block, derived)

Outputs
-------
papers/lure-broadcast/results/metadata_history_<chain>.json (+ .manifest.json)

Failure modes
-------------
- no archive-capable endpoint            -> RAISE. A pruning node answers `latest` perfectly and
                                            fails only at the pinned block, so this is tested at the
                                            height actually used, not inferred from reachability.
- an RPC error at a height               -> `rpc_error`, never counted as agreement OR disagreement.
- comparable coverage below --min-coverage -> RAISE, rather than publishing a reassuring ratio drawn
                                            from whichever handful of contracts happened to answer.
- zero contracts probed                  -> RAISE.

The asymmetry is deliberate: an absent answer is never recorded as agreement. A throttled endpoint
returning nothing would otherwise be indistinguishable from a corpus in which no metadata changed.

Complexity: three heights x two calls per contract, batched. No corpus pass.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import metadata_lure as ml  # noqa: E402
from src import registry, token_meta  # noqa: E402
from forgery_probe import ARCHIVE_ENDPOINTS, archive_capable, corpus_end_block  # noqa: E402

OUT_ROOT = ROOT / "papers" / "lure-broadcast" / "results"

NAME_SEL = "0x06fdde03"
SYMBOL_SEL = "0x95d89b41"
GET_CODE = "eth_getCode"


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def decode_string(hex_result: str | None) -> str | None:
    """Decode an ABI-encoded `string` return. Returns None on anything unexpected.

    Returning None rather than guessing matters: a half-decoded string compared against the
    collection-time value would read as a metadata CHANGE, which is the finding under test.
    """
    if not hex_result or hex_result in ("0x", "0X"):
        return None
    raw = bytes.fromhex(hex_result[2:])
    if len(raw) < 64:
        # Some non-conformant tokens return a bare bytes32 instead of a dynamic string.
        if len(raw) == 32:
            return raw.rstrip(b"\x00").decode("utf-8", errors="replace") or None
        return None
    offset = int.from_bytes(raw[:32], "big")
    if offset + 32 > len(raw):
        return None
    length = int.from_bytes(raw[offset:offset + 32], "big")
    body = raw[offset + 32:offset + 32 + length]
    if len(body) != length:
        return None
    return body.decode("utf-8", errors="replace")


def flagged_contracts(chain: str) -> dict[str, str]:
    md = json.loads((ROOT / "analysis" / "config" / "token_metadata.json").read_text())[chain]
    out = {}
    for a, m in md.items():
        v = ml.classify(m.get("name"), m.get("symbol"))
        if v.strength == "strong":
            k = ml.campaign_key(v)
            if k:
                out[a.lower()] = k
    return out


def probe(endpoint: str, addr: str, block_hex: str, batch: int) -> dict:
    """Fetch code presence, name and symbol for one contract at one height."""
    calls = [
        {"jsonrpc": "2.0", "id": "code", "method": GET_CODE, "params": [addr, block_hex]},
        {"jsonrpc": "2.0", "id": "name", "method": "eth_call",
         "params": [{"to": addr, "data": NAME_SEL}, block_hex]},
        {"jsonrpc": "2.0", "id": "symbol", "method": "eth_call",
         "params": [{"to": addr, "data": SYMBOL_SEL}, block_hex]},
    ]
    r = token_meta._rpc_batch(calls, endpoint, retries=2)
    if not r or not isinstance(r, list):
        return {"verdict": "rpc_error"}
    by = {}
    for el in r:
        if isinstance(el, dict) and "id" in el:
            by[el["id"]] = el
    if any(k not in by for k in ("code", "name", "symbol")):
        return {"verdict": "rpc_error"}
    if any("error" in by[k] for k in ("code", "name", "symbol")):
        return {"verdict": "rpc_error"}
    code = by["code"].get("result")
    if code in (None, "0x", "0X"):
        return {"verdict": "no_code_at_height"}
    return {"verdict": "answered",
            "name": decode_string(by["name"].get("result")),
            "symbol": decode_string(by["symbol"].get("result"))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chain", default="base")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--sleep", type=float, default=0.15)
    ap.add_argument("--min-coverage", type=float, default=0.80,
                    help="minimum share of contracts yielding at least one comparable height")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = OUT_ROOT / f"metadata_history_{args.chain}.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out.relative_to(ROOT)} exists; pass --force")

    md = json.loads((ROOT / "analysis" / "config" / "token_metadata.json").read_text())[args.chain]
    flagged = flagged_contracts(args.chain)
    if not flagged:
        raise SystemExit(f"no flagged contracts on {args.chain}")

    census = OUT_ROOT / f"campaign_census_{args.chain}.json"
    if not census.exists():
        raise SystemExit(f"missing {census.name}; the campaign block windows come from it")
    per_campaign = json.loads(census.read_text())["per_campaign"]

    end_block = corpus_end_block(args.chain)
    end_hex = hex(end_block)
    # The full configured list, primary included. CHAIN_RPC_FALLBACKS alone omits the primary, and
    # holds nothing at all for zkSync Era, whose only endpoint serves archive state.
    endpoints = list(dict.fromkeys(ARCHIVE_ENDPOINTS.get(args.chain, [])
                                   + token_meta.rpc_chain_for(args.chain)))
    # The canary is a registry token, as in frame_probe.py. A flagged contract may revert the canary
    # call, which would make a working archive endpoint look unusable.
    canaries = sorted("0x" + a.lower() for a in registry.load_token_registry(args.chain).legit_addresses)
    probe_addr = canaries[0] if canaries else next(iter(flagged))
    usable = archive_capable(endpoints, end_hex, probe_addr)
    if not usable:
        raise SystemExit(
            f"no endpoint serves archive state for {args.chain} at block {end_block}. Without one "
            f"this question cannot be answered, and reporting agreement measured only at `latest` "
            f"would restate the assumption rather than test it.")
    endpoint = usable[0]
    print(f"  {args.chain}: {len(flagged)} flagged contracts, archive endpoint {endpoint}")

    records: dict[str, dict] = {}
    t0 = time.time()
    for i, (addr, camp) in enumerate(sorted(flagged.items()), 1):
        window = per_campaign.get(camp) or {}
        heights = {"first": window.get("first_block"), "last": window.get("last_block"),
                   "end": end_block}
        stored = {"name": md[addr].get("name") if addr in md else None,
                  "symbol": md[addr].get("symbol") if addr in md else None}
        at = {}
        for label, blk in heights.items():
            if blk is None:
                at[label] = {"verdict": "no_window"}
                continue
            r = probe(endpoint, addr, hex(int(blk)), args.batch)
            r["block"] = int(blk)
            at[label] = r
            time.sleep(args.sleep)

        # The reference is the END-height reading taken by this same code, so that a decoding or
        # collection difference cannot masquerade as a change over time.
        ref = at.get("end") if at.get("end", {}).get("verdict") == "answered" else None
        historical = [v for k, v in at.items()
                      if k != "end" and v["verdict"] == "answered"]
        for v in historical:
            v["matches_end"] = (ref is not None and v["name"] == ref["name"]
                                and v["symbol"] == ref["symbol"])
        if ref is None:
            verdict = "unresolved" if not historical else "no_reference_height"
        elif not historical:
            verdict = "only_current_height"
        elif all(v["matches_end"] for v in historical):
            verdict = "stable"
        else:
            verdict = "changed"

        # Separate question, same data: does the stored registry match the chain today?
        if ref is None:
            reg = "unresolved"
        elif ref["name"] == stored["name"] and ref["symbol"] == stored["symbol"]:
            reg = "registry_matches"
        elif ((ref["name"] or "").strip() == (stored["name"] or "").strip()
              and (ref["symbol"] or "").strip() == (stored["symbol"] or "").strip()):
            reg = "registry_whitespace_only"
        else:
            reg = "registry_mismatch"
        records[addr] = {"campaign": camp, "stored": stored, "on_chain_now": ref,
                         "at": at, "verdict": verdict, "registry": reg}
        if i % 20 == 0:
            print(f"    {i}/{len(flagged)} probed, {time.time()-t0:.0f}s", flush=True)

    by_verdict: dict[str, int] = {}
    for r in records.values():
        by_verdict[r["verdict"]] = by_verdict.get(r["verdict"], 0) + 1
    if sum(by_verdict.values()) != len(flagged):
        raise SystemExit("verdicts do not sum to the probed population")

    by_registry: dict[str, int] = {}
    for r in records.values():
        by_registry[r["registry"]] = by_registry.get(r["registry"], 0) + 1
    comparable = by_verdict.get("stable", 0) + by_verdict.get("changed", 0)
    coverage = comparable / len(flagged)
    changed = [a for a, r in records.items() if r["verdict"] == "changed"]

    doc = {
        "chain": args.chain,
        "question": ("does the collection-time metadata match what the contract reported inside its "
                     "own emission window? Comparable contracts are those answering at a historical "
                     "height; the rest are reported, not assumed to agree."),
        "endpoint": endpoint,
        "corpus_end_block": end_block,
        "flagged_contracts": len(flagged),
        "by_verdict": by_verdict,
        "comparable_contracts": comparable,
        "comparable_coverage": coverage,
        "changed_contracts": sorted(changed),
        "by_registry": by_registry,
        "registry_note": ("a SEPARATE question from the temporal one: whether the stored metadata "
                          "the detector reads matches what the contract returns today. Whitespace-"
                          "only differences are separated out because the detector normalises "
                          "whitespace and they cannot change its verdict."),
        "registry_mismatch_contracts": sorted(
            a for a, r in records.items() if r["registry"] == "registry_mismatch"),
        "stable_share_of_comparable": (by_verdict.get("stable", 0) / comparable) if comparable else None,
        "records": records,
        "caveats": [
            "no_code_at_height means the contract did not yet exist at that height; it is not a "
            "metadata change and is excluded from the comparison rather than counted either way.",
            "rpc_error is neither agreement nor disagreement. Silence is never read as stability.",
            "The heights are the campaign's first and last block, which bracket the contract's "
            "emission window but need not be blocks at which this particular contract emitted.",
            "Historical readings are compared with an end-block reading taken by the same code, "
            "not with the stored registry, so that a collection-time capture difference cannot be "
            "reported as a change over time. The stored-versus-chain comparison is reported "
            "separately under by_registry.",
        ],
    }
    if coverage < args.min_coverage:
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        partial = OUT_ROOT / f"metadata_history_{args.chain}_PARTIAL.json"
        partial.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        raise SystemExit(
            f"comparable coverage {coverage:.4f} < {args.min_coverage}; wrote {partial.name} and "
            f"refusing to emit a headline stability figure from {comparable} of {len(flagged)} "
            f"contracts")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    (OUT_ROOT / f"metadata_history_{args.chain}.manifest.json").write_text(json.dumps({
        "producer": "papers/lure-broadcast/scripts/metadata_history.py",
        "git_commit": git_commit(), "endpoint": endpoint,
    }, indent=1) + "\n")

    print(f"\n  verdicts: {by_verdict}")
    print(f"  registry: {by_registry}")
    print(f"  comparable {comparable}/{len(flagged)} ({coverage:.4f}); "
          f"stable {doc['stable_share_of_comparable']}")
    if changed:
        print(f"  CHANGED: {changed}")
    print(f"  -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
