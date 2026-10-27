"""Do the historically unresolvable contracts differ from the ones that answered?

Purpose
-------
The historical-metadata check compares each flagged contract's response at heights inside its
emission window with an end-of-corpus reading. Nine contracts could not be resolved, and the result
is therefore conditioned on an endpoint's willingness to serve historical state. That is
response selection, and the manuscript already declines to put a confidence interval on it. What the
manuscript did not do, and a referee asked for, is say whether the unresolved contracts look like
the resolved ones.

This script builds that comparison descriptively. It does not fit a missing-data model and does not
test a hypothesis: with nine contracts on one side, any test would be theatre. It reports the
attributes a reader would want in order to judge whether the failures look incidental or systematic:

  VOLUME        transfers emitted, since a large unresolved contract would matter more
  AGE           first block of the contract's campaign window, since archive availability
                plausibly degrades with depth
  CODE SIZE     deployed bytecode length, a crude proxy for implementation complexity
  PROXY         whether the deployed code is an EIP-1167 minimal proxy, the referee's first
                hypothesis for why historical state might not resolve. The bytecode is fetched raw
                and matched against the 45-byte pattern; proxy status is never inferred from code
                length alone, which is a rule this project adopted after a length-based inference
                proved wrong elsewhere.

Inputs   papers/lure-broadcast/results/metadata_history_<chain>.json
         papers/lure-broadcast/results/{forgery_chain,campaign_census}_<chain>.json
Outputs  papers/lure-broadcast/results/unresolved_profile_<chain>.json (+ manifest)

Failure modes
-------------
- A contract missing from the volume table RAISES rather than being scored as zero volume, which
  would understate the unresolved group's weight, the direction that flatters the paper.
- Code coverage below --min-coverage RAISES rather than profiling whichever contracts answered,
  since that would reintroduce response selection into the check for response selection.

Complexity: one eth_getCode sweep over the flagged contracts. No corpus pass.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analysis"))

from src import codemeta  # noqa: E402
from src import token_meta  # noqa: E402

RES = ROOT / "papers" / "lure-broadcast" / "results"
RESOLVED = ("stable", "changed")


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def fetch_raw_code(addresses: list[str], chain: str, batch: int = 8) -> dict[str, str | None]:
    """address -> deployed bytecode hex, or None where no endpoint answered.

    None means unanswered and is counted as missing coverage, never as an empty contract: those are
    different facts and conflating them is how a throttled endpoint becomes a finding.
    """
    endpoints = token_meta.CHAIN_RPC_FALLBACKS.get(chain, [])
    out: dict[str, str | None] = {a: None for a in addresses}
    todo = list(addresses)
    for ep in endpoints:
        if not todo:
            break
        nxt = []
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            calls = [{"jsonrpc": "2.0", "id": a, "method": "eth_getCode",
                      "params": [a, "latest"]} for a in chunk]
            r = token_meta._rpc_batch(calls, ep, retries=2)
            got = {e["id"]: e for e in r if isinstance(e, dict) and "id" in e} if r else {}
            for a in chunk:
                e = got.get(a)
                if e and "result" in e and isinstance(e["result"], str):
                    out[a] = e["result"]
                else:
                    nxt.append(a)
        todo = nxt
    return out


def summarise(rows: list[dict]) -> dict:
    """Descriptive summary of one group. Medians, because nine observations have no mean worth
    quoting and the volume distribution is heavy-tailed."""
    if not rows:
        return {"n": 0}
    vols = sorted(r["transfers"] for r in rows)
    firsts = [r["first_block"] for r in rows if r["first_block"] is not None]
    sizes = [r["code_bytes"] for r in rows if r["code_bytes"] is not None]
    return {
        "n": len(rows),
        "transfers_total": sum(vols),
        "transfers_median": statistics.median(vols),
        "transfers_min": vols[0], "transfers_max": vols[-1],
        "first_block_median": statistics.median(firsts) if firsts else None,
        "code_bytes_median": statistics.median(sizes) if sizes else None,
        "eip1167_proxies": sum(1 for r in rows if r["is_proxy"]),
        "no_code": sum(1 for r in rows if r["code_bytes"] == 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chain", default="base")
    ap.add_argument("--min-coverage", type=float, default=0.95)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = RES / f"unresolved_profile_{args.chain}.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out.relative_to(ROOT)} exists; pass --force")

    mh = json.loads((RES / f"metadata_history_{args.chain}.json").read_text())
    vol = json.loads((RES / f"forgery_chain_{args.chain}.json").read_text())[
        "lure_transfers_by_contract"]
    camp = json.loads((RES / f"campaign_census_{args.chain}.json").read_text())["per_campaign"]

    addrs = sorted(mh["records"])
    print(f"  {args.chain}: fetching code for {len(addrs)} flagged contracts")
    # fetch_code stores only length and hash, so the raw bytecode is fetched here: the EIP-1167
    # pattern cannot be checked without it, and length alone must not stand in for the pattern.
    code = fetch_raw_code(addrs, args.chain)
    cov = sum(1 for v in code.values() if v is not None) / len(code)
    if cov < args.min_coverage:
        raise SystemExit(f"code coverage {cov:.4f} < {args.min_coverage}; profiling only the "
                         f"contracts that answered would reintroduce response selection into the "
                         f"very check meant to detect it")

    rows = []
    for a, rec in mh["records"].items():
        if a not in vol:
            raise SystemExit(f"{a} has no entry in the volume table; scoring it zero would "
                             f"understate the unresolved group's weight")
        hexcode = code.get(a)
        rows.append({
            "address": a,
            "resolved": rec["verdict"] in RESOLVED,
            "transfers": vol[a],
            "first_block": (camp.get(rec["campaign"]) or {}).get("first_block"),
            "code_bytes": (len(hexcode) - 2) // 2 if isinstance(hexcode, str)
                          and hexcode.startswith("0x") else None,
            "is_proxy": bool(codemeta.detect_eip1167(hexcode)) if isinstance(hexcode, str) else False,
        })

    res = [r for r in rows if r["resolved"]]
    unres = [r for r in rows if not r["resolved"]]
    total = sum(r["transfers"] for r in rows)

    doc = {
        "chain": args.chain,
        "question": ("do the contracts whose historical state no endpoint would serve differ "
                     "descriptively from those that answered?"),
        "not_a_test": ("with nine contracts on one side no hypothesis test is meaningful; this is "
                       "a description so the reader can judge whether the failures look incidental "
                       "or systematic, not evidence that they are ignorable"),
        "code_coverage": cov,
        "resolved": summarise(res),
        "unresolved": summarise(unres),
        "unresolved_share_of_volume": (sum(r["transfers"] for r in unres) / total) if total else None,
        "unresolved_contracts": sorted(r["address"] for r in unres),
    }
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    (RES / f"unresolved_profile_{args.chain}.manifest.json").write_text(json.dumps({
        "producer": "papers/lure-broadcast/scripts/unresolved_profile.py",
        "git_commit": git_commit(),
    }, indent=1) + "\n")

    for name, g in (("resolved", doc["resolved"]), ("unresolved", doc["unresolved"])):
        print(f"    {name:11s} n={g['n']:>3d}  median transfers {g['transfers_median']:>10,.0f}  "
              f"median first block {g['first_block_median']}  "
              f"median code {g['code_bytes_median']}  proxies {g['eip1167_proxies']}")
    print(f"  unresolved carry {doc['unresolved_share_of_volume']:.4%} of flagged volume")
    print(f"  -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
