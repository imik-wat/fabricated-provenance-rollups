"""Sensitivity of the reported counts to the remaining free choices (reviewer items 12, 13, 9).

Three knobs are swept here; the rank cutoff is swept in `sampling_frame.py` and the confusable fold
in `normalisation_ablation.py`.

1. UNRESOLVED METADATA. A contract whose name and symbol could not be fetched is excluded from the
   detector population. Excluding it is not neutral: if unresolved contracts were flagged at the
   same rate as resolved ones, the counts would be higher. The number and share of unresolved
   contracts is reported per chain, together with the count that would obtain under two explicit
   assumptions -- none of them a payload, and all of them flagged at the observed rate. Those
   bracket the truth without pretending to locate it.

2. CAMPAIGN KEY. Grouping by normalised registrable domain is a choice, and the reviewer is right
   that a single domain may host unrelated campaigns while a shortener host is shared by many. Four
   keys are compared: host only, host with path (the reported key), the full normalised destination,
   and the payload string itself. The spread between them is the sensitivity of every campaign count
   in the paper.

3. DETECTOR BAR. `strong` requires a destination and a call to action; `weak` requires one. The
   paper reports `strong`. Both are given so a reader can see what the stricter bar costs.

Outputs
-------
papers/lure-broadcast/results/sensitivity_battery.json (+ .manifest.json)

Failure modes
-------------
- A chain missing from the metadata cache raises rather than being skipped.
- The host-only key is computed but flagged as UNSAFE for reporting: it merges every campaign that
  shares a shortener, which is why the paper does not use it. The magnitude of that merging is the
  point of computing it.

Complexity: a few classifications per contract; no network access.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analysis"))

from src import metadata_lure as ml  # noqa: E402

OUT_ROOT = ROOT / "papers" / "lure-broadcast" / "results"
CHAINS = ["arbitrum-one", "base", "optimism", "polygon-zkevm", "zksync-era"]
REQUESTED_DEPTH = 1200


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def keys_for(v) -> dict[str, str | None]:
    """The four candidate campaign keys for one verdict."""
    reported = ml.campaign_key(v)                     # host, plus path for aggregator hosts
    host = None
    if reported:
        host = reported.split("/")[0]                 # strip any retained slug
    dests = tuple(v.domains) + tuple(v.aggregator_hosts)
    full = "|".join(sorted(dests)) or None
    return {"host_only": host, "host_plus_path": reported, "full_destination": full}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = OUT_ROOT / "sensitivity_battery.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out.relative_to(ROOT)} exists; pass --force")

    cache = json.loads((ROOT / "analysis" / "config" / "token_metadata.json").read_text())
    sweeps = json.loads((OUT_ROOT / "sampling_frame.json").read_text())["per_chain"]
    per_chain = {}

    for chain in CHAINS:
        md = cache.get(chain)
        if not md:
            raise SystemExit(f"{chain} absent from the metadata cache")
        resolved = len(md)
        requested = REQUESTED_DEPTH
        unresolved = max(0, requested - resolved)

        bands = {"strong": 0, "weak": 0, "none": 0}
        key_sets: dict[str, set] = {"host_only": set(), "host_plus_path": set(),
                                    "full_destination": set(), "payload": set()}
        for a, rec in md.items():
            v = ml.classify(rec.get("name"), rec.get("symbol"))
            bands[v.strength] += 1
            if v.strength != "strong":
                continue
            ks = keys_for(v)
            for k, val in ks.items():
                if val:
                    key_sets[k].add(val)
            payload = " ".join(x for x in (rec.get("name"), rec.get("symbol")) if x)
            key_sets["payload"].add(payload.lower())

        flagged = bands["strong"]
        rate = flagged / resolved if resolved else 0.0
        per_chain[chain] = {
            "unresolved_metadata": {
                "requested_depth": requested, "resolved": resolved, "unresolved": unresolved,
                "unresolved_share": unresolved / requested if requested else 0.0,
                "flagged_observed": flagged,
                "flagged_if_all_unresolved_are_payloads": flagged + unresolved,
                "flagged_if_unresolved_match_observed_rate": flagged + round(unresolved * rate),
                "note": ("the reported figure assumes NOTHING about unresolved contracts -- they are "
                         "simply absent. The two alternatives bracket the effect of that exclusion "
                         "and are not estimates."),
            },
            "campaign_key": {k: len(s) for k, s in key_sets.items()},
            "campaign_key_note": ("host_only merges every campaign sharing a shortener and is UNSAFE "
                                  "to report; host_plus_path is the key used throughout"),
            "detector_bar": {"strong": bands["strong"], "weak": bands["weak"],
                             "strong_plus_weak": bands["strong"] + bands["weak"],
                             "none": bands["none"]},
        }
        u = per_chain[chain]["campaign_key"]
        print(f"  {chain:15s} unresolved {unresolved:>3d}/{requested} "
              f"({unresolved/requested*100:5.2f}%) | flagged {flagged:>4d} "
              f"(+weak -> {bands['strong']+bands['weak']:>4d}) | campaigns "
              f"host {u['host_only']:>3d}, host+path {u['host_plus_path']:>3d}, "
              f"dest {u['full_destination']:>3d}, payload {u['payload']:>4d}")

    # Cross-chain summary of the campaign-key spread, which is the figure the paper needs.
    spread = {}
    for k in ("host_only", "host_plus_path", "full_destination", "payload"):
        spread[k] = sum(per_chain[c]["campaign_key"][k] for c in CHAINS)
    doc = {
        "purpose": "sensitivity of reported counts to unresolved metadata, campaign key and detector bar",
        "per_chain": per_chain,
        "campaign_key_totals": spread,
        "campaign_key_ratio_host_only_to_reported": (
            spread["host_only"] / spread["host_plus_path"] if spread["host_plus_path"] else None),
        "rank_cutoff": "swept separately in sampling_frame.json",
        "normalisation": "ablated separately in normalisation_ablation.json",
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    (OUT_ROOT / "sensitivity_battery.manifest.json").write_text(json.dumps({
        "producer": "papers/lure-broadcast/scripts/sensitivity_battery.py",
        "git_commit": git_commit(),
    }, indent=1) + "\n")
    print(f"\n  campaign totals across chains: {spread}")
    print(f"  -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
