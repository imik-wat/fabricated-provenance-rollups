"""Characterise the sampling frame, and sweep the rank cutoff (reviewer items 3 and 9).

Why this exists
---------------
The manuscript described a "census" without consistently separating two different things, and the
reviewer was right to object:

  * the census of transfer EVENTS is complete -- every row of every chain is read; and
  * the detector POPULATION is not, being the volume head of the contract universe.

Conflating them lets headline counts read as complete-chain totals when they are lower bounds. This
script produces the numbers needed to state the distinction per chain, and sweeps the one free
parameter of the population definition so the reader can see what the cutoff costs.

Two coverage measures, because they disagree violently
------------------------------------------------------
Coverage by CONTRACT and coverage by TRANSFER VOLUME are not interchangeable here. On Base the top
1,200 contracts are 0.005% of the universe yet carry 35% of all transfer volume; on Polygon zkEVM
the same depth is 14.6% of contracts and 27%. Reporting only the first understates the frame and
only the second overstates it, so both are emitted and the paper quotes both.

The volume split is reported in three parts that sum to one: the examined head, the
registry-verified tokens excluded by design, and the UNEXAMINED remainder -- non-registry contracts
below the cutoff. That last figure is the honest measure of what the cutoff costs, and it ranges
from 0.05% on zkSync Era to 18.94% on Base.

The rank sweep
--------------
Depths are evaluated as prefixes of the SAME cached metadata, so no additional RPC is needed and the
arms differ only in depth. A depth deeper than the cached population cannot be evaluated and is
reported as such rather than silently truncated -- otherwise the deepest rung would look identical
to the cache size by construction.

Inputs
------
results/<chain>_token_volumes.csv  (or this paper's tree for Polygon zkEVM)
analysis/config/token_metadata.json
analysis/src/registry.py

Outputs
-------
papers/lure-broadcast/results/sampling_frame.json (+ .manifest.json)

Failure modes
-------------
- A missing volume table raises rather than reporting an unenumerated universe.
- A depth exceeding the cached metadata is recorded as `evaluable: false`, never silently clipped.

Complexity: one pass over each volume table plus two classifications per cached contract.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analysis"))

from src import metadata_lure as ml, registry  # noqa: E402

OUT_ROOT = ROOT / "papers" / "lure-broadcast" / "results"
VOLUMES = ROOT / "results"
CHAINS = ["arbitrum-one", "base", "optimism", "polygon-zkevm", "zksync-era"]
DEPTHS = (300, 600, 1200)
SWEPT_DEPTH = 1200


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def volume_table(chain: str) -> tuple[list[tuple[str, int]], int, int]:
    """Returns (non-registry rows descending, total volume over ALL contracts, registry volume).

    Two denominators matter and they are not interchangeable. The detector population excludes
    registry-verified tokens by design, so ranking must exclude them; but a reader asking what share
    of chain activity was examined means share of ALL transfer volume. Reporting the head against a
    registry-excluded denominator silently inflates coverage -- on Base the same cutoff reads 65% of
    non-registry volume and 81% of all volume. Both are returned so the paper can state which.
    """
    cands = [OUT_ROOT / f"{chain}_token_volumes.csv", VOLUMES / f"{chain}_token_volumes.csv"]
    path = next((p for p in cands if p.exists()), None)
    if path is None:
        raise SystemExit("missing volume table, looked in: "
                         + ", ".join(str(p.relative_to(ROOT)) for p in cands))
    reg = {("0x" + a.lower()) for a in registry.load_token_registry(chain).legit_addresses}
    rows, total, regvol = [], 0, 0
    for r in csv.DictReader(path.open()):
        a, n = r["token"].lower(), int(r["count"])
        total += n
        if a in reg:
            regvol += n
        else:
            rows.append((a, n))
    rows.sort(key=lambda kv: -kv[1])
    return rows, total, regvol


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = OUT_ROOT / "sampling_frame.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out.relative_to(ROOT)} exists; pass --force")

    cache = json.loads((ROOT / "analysis" / "config" / "token_metadata.json").read_text())
    per_chain: dict[str, dict] = {}

    for chain in CHAINS:
        rows, total_volume, registry_volume = volume_table(chain)
        total_contracts = len(rows)
        md = cache.get(chain, {})
        if not md:
            raise SystemExit(f"{chain} has no cached metadata; run run_metadata_sweep.py")

        # Deepest rung = the chain's actual cached depth, since unresolved contracts are not cached
        # and a fixed 1200 would be unevaluable on three chains.
        head = rows[:SWEPT_DEPTH]
        cut_volume = sum(n for _, n in head)
        frame = {
            "total_contracts": total_contracts,
            "total_transfers": total_volume,
            "rank_cutoff": SWEPT_DEPTH,
            "contracts_covered": len(head),
            "contract_fraction": len(head) / total_contracts,
            "transfer_fraction_all": cut_volume / total_volume,
            "registry_fraction": registry_volume / total_volume,
            "unexamined_fraction": 1 - (cut_volume + registry_volume) / total_volume,
            "transfers_at_cutoff": head[-1][1] if head else None,
            "metadata_resolved": len(md),
        }

        # Rank sweep: prefixes of the same cached population.
        sweep = []
        vol_rank = {a: i for i, (a, _) in enumerate(rows)}
        cached = sorted((a for a in (x.lower() for x in md) if a in vol_rank),
                        key=lambda a: vol_rank[a])
        # The deepest rung is the chain's ACTUAL cached depth: unresolved contracts are never
        # cached, so a fixed 1200 is unevaluable on three of the five chains and would silently
        # drop the rung that matters most.
        depths = tuple(sorted({d for d in DEPTHS if d < len(cached)} | {len(cached)}))
        for d in depths:
            if d > len(cached):
                sweep.append({"depth": d, "evaluable": False,
                              "reason": f"cached metadata holds only {len(cached)} ranked contracts"})
                continue
            sub = cached[:d]
            strong = camps = 0
            keys = set()
            for a in sub:
                v = ml.classify(md[a].get("name"), md[a].get("symbol"))
                if v.strength == "strong":
                    strong += 1
                    k = ml.campaign_key(v)
                    if k:
                        keys.add(k)
            camps = len(keys)
            sweep.append({
                "depth": d, "evaluable": True,
                "flagged": strong, "campaigns": camps,
                "flagged_rate": strong / d,
                "transfers_at_depth": rows[d - 1][1],
                "transfer_fraction": sum(n for _, n in rows[:d]) / total_volume,
            })
        per_chain[chain] = {"frame": frame, "rank_sweep": sweep}
        f = frame
        print(f"  {chain:15s} universe {f['total_contracts']:>10,} | cutoff {f['rank_cutoff']} "
              f"= {f['contract_fraction']*100:6.3f}% of contracts, "
              f"{f['transfer_fraction_all']*100:6.2f}% of all transfers "
              f"(registry {f['registry_fraction']*100:5.2f}%, unexamined "
              f"{f['unexamined_fraction']*100:5.2f}%), "
              f"floor {f['transfers_at_cutoff']:,}")
        for s in sweep:
            if s["evaluable"]:
                print(f"      depth {s['depth']:>4d}: flagged {s['flagged']:>4d} "
                      f"({s['flagged_rate']*100:5.2f}%), campaigns {s['campaigns']:>3d}, "
                      f"volume {s['transfer_fraction']*100:6.2f}%")

    doc = {
        "purpose": ("separate the COMPLETE census of transfer events from the RANK-LIMITED detector "
                    "population, and show what the cutoff costs"),
        "event_census": ("complete: every transfer row of every chain is read by the full-chain "
                         "scans; no sampling is involved in the event counts"),
        "detector_population": ("the volume head of the contract universe, registry tokens excluded; "
                                "contracts below the cutoff are not examined, so every count derived "
                                "from it is a LOWER BOUND"),
        "coverage_note": ("contract coverage and transfer-volume coverage differ by orders of "
                          "magnitude and both must be quoted; neither alone describes the frame"),
        "depths_requested": list(DEPTHS),
        "denominator_note": ("transfer_fraction_all is share of ALL transfer volume including "
                             "registry-verified tokens, which the detector population excludes by "
                             "design; registry_fraction and unexamined_fraction complete the split"),
        "per_chain": per_chain,
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    (OUT_ROOT / "sampling_frame.manifest.json").write_text(json.dumps({
        "producer": "papers/lure-broadcast/scripts/sampling_frame.py",
        "git_commit": git_commit(),
    }, indent=1) + "\n")
    print(f"\n  -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
