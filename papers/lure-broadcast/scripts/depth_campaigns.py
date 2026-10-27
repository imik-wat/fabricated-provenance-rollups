"""Campaigns per rank-depth bucket, completing the depth table (reviewer round 2, item 11).

Why this is separate from the scan
-----------------------------------
`forgery_chain_scan.py` accumulates contracts, transfers and distinct recipients per depth bucket in
its single chain pass, because those quantities need the rows. A campaign count does not: a campaign
is an equivalence class of contracts under the normalised registrable domain, so it is a function of
the flagged contract set and its payloads alone. Computing it here avoids a second multi-hour pass
over 19 billion rows to learn something already determined by data in hand.

The bucket assignment is IMPORTED from the scan (`depth_buckets`) rather than reimplemented, so the
campaign counts partition exactly the same population as the contract, transfer and recipient counts
they sit beside in the table. A second implementation could drift from the first without either
being obviously wrong, and the two would then be describing different populations under one heading.

Campaigns do not add across buckets
-----------------------------------
A campaign can span buckets: two contracts sharing a domain may have very different volume ranks. The
cumulative campaign count at depth d is therefore the size of the union of campaign keys over buckets
0..d, which is NOT the sum of per-bucket counts, and the marginal is the number of campaign keys
appearing at depth d that appeared at no shallower depth. Both are computed as set operations for
that reason. This is why the marginal campaign count can be smaller than the marginal contract count
even when many contracts are added.

Inputs
------
analysis/config/token_metadata.json                      (payloads)
papers/lure-broadcast/results/<chain>_token_volumes.csv  (ranks, via depth_buckets)
papers/lure-broadcast/results/forgery_chain_<chain>.json (the contract totals to agree with)

Outputs
-------
papers/lure-broadcast/results/depth_campaigns.json (+ .manifest.json)

Failure modes
-------------
- If the per-bucket contract counts disagree with those the scan recorded, the run RAISES: the two
  are the same assignment over the same population and a divergence means the artefacts describe
  different things.
- A chain with no scan is skipped and named in the output.

Complexity: one classification per flagged contract; no chain pass, no network.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import metadata_lure as ml, registry  # noqa: E402
from forgery_chain_scan import DEPTH_CUTOFFS, bare, depth_buckets  # noqa: E402

OUT_ROOT = ROOT / "papers" / "lure-broadcast" / "results"
CHAINS = ["arbitrum-one", "base", "optimism", "polygon-zkevm", "zksync-era"]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = OUT_ROOT / "depth_campaigns.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out.relative_to(ROOT)} exists; pass --force")

    cache = json.loads((ROOT / "analysis" / "config" / "token_metadata.json").read_text())
    per_chain, missing = {}, []

    for chain in CHAINS:
        scan_p = OUT_ROOT / f"forgery_chain_{chain}.json"
        if not scan_p.exists():
            missing.append(chain)
            continue
        scan = json.loads(scan_p.read_text())
        md = cache[chain]
        flagged = sorted({a.lower() for a, m in md.items()
                          if ml.classify(m.get("name"), m.get("symbol")).strength == "strong"})
        targets = sorted("0x" + a.lower()
                         for a in registry.load_token_registry(chain).legit_addresses)
        buckets = depth_buckets(chain, flagged, targets)

        # campaign keys present in each bucket
        keys_in: dict[int, set[str]] = {i: set() for i in range(len(DEPTH_CUTOFFS))}
        counts_in: dict[int, int] = {i: 0 for i in range(len(DEPTH_CUTOFFS))}
        for a in flagged:
            bi = buckets.get(bare(a))
            if bi is None:
                continue
            counts_in[bi] += 1
            rec = md[a] if a in md else md[a.lower()]
            k = ml.campaign_key(ml.classify(rec.get("name"), rec.get("symbol")))
            if k:
                keys_in[bi].add(k)

        recorded = [r["marginal_contracts"] for r in scan["depth_table"]]
        mine = [counts_in[i] for i in range(len(DEPTH_CUTOFFS))]
        if mine != recorded:
            raise SystemExit(f"{chain}: per-bucket contract counts {mine} disagree with the scan's "
                             f"{recorded}; the campaign counts would describe a different population "
                             f"from the columns beside them")

        rows, seen = [], set()
        for i, cut in enumerate(DEPTH_CUTOFFS):
            new = keys_in[i] - seen
            seen |= keys_in[i]
            rows.append({
                "cutoff": None if cut >= 10**9 else cut,
                "cutoff_label": "all" if cut >= 10**9 else str(cut),
                "campaigns": len(seen),
                "marginal_campaigns": len(new),
                "marginal_contracts": counts_in[i],
            })
        per_chain[chain] = {"depth_table": rows, "campaigns_total": len(seen)}
        print(f"  {chain:14s} campaigns by depth "
              f"{[r['campaigns'] for r in rows]} (marginal {[r['marginal_campaigns'] for r in rows]})")

    if not per_chain:
        raise SystemExit("no chain scans found; nothing to report")

    doc = {
        "purpose": "campaign counts per rank-depth bucket, to complete the depth-sensitivity table",
        "depth_cutoffs": [None if c >= 10**9 else c for c in DEPTH_CUTOFFS],
        "per_chain": per_chain,
        "chains_without_a_scan": missing,
        "additivity_note": ("a campaign may span buckets, so cumulative counts are set unions and "
                            "the marginal is the number of keys new at that depth; per-bucket counts "
                            "do not sum to the total"),
        "assignment": ("bucket assignment is imported from forgery_chain_scan.depth_buckets, the "
                       "same function the scan used, and the per-bucket contract counts are asserted "
                       "equal to those the scan recorded"),
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    (OUT_ROOT / "depth_campaigns.manifest.json").write_text(json.dumps({
        "producer": "papers/lure-broadcast/scripts/depth_campaigns.py",
        "git_commit": git_commit(),
    }, indent=1) + "\n")
    if missing:
        print(f"  chains without a scan: {missing}")
    print(f"  -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
