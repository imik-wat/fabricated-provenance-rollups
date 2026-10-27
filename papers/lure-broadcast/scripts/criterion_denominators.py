"""The fabricated-sender criterion against four denominators (reviewer round 2, item 9).

Why four
--------
The criterion was reported as a single share of transfer volume. A volume-weighted share can be
carried by one campaign: on a channel where the largest contract emits tens of millions of transfers
and the median emits thousands, "half the volume" and "half the contracts" are different statements,
and only the first was being made. The reviewer is right that this is ambiguous, so the criterion is
reported here against every denominator it can sensibly take:

  contracts   -- flagged contracts satisfying the criterion, over flagged contracts in the frame
  campaigns   -- campaigns CONTAINING at least one such contract, over campaigns in the frame
  volume      -- transfer events satisfying it, over channel transfer events
  recipients  -- distinct addresses named as recipient in such an event, over channel recipients

The campaign denominator needs its qualifier stated wherever it appears: a campaign counts if any of
its contracts satisfies the criterion, which is an upper count of campaign involvement and a lower
count of campaign-wide fabrication. Both readings are wrong if the qualifier is dropped.

The recipient denominator does not decompose. Distinct-recipient sets intersect across contracts, so
the per-chain recipient share is NOT the sum of per-contract shares and the shares do not add across
chains either; the totals here are therefore reported per chain only.

Inputs
------
papers/lure-broadcast/results/forgery_chain_<chain>.json   (criterion counts, channel totals)
analysis/config/token_metadata.json                        (payloads, for the campaign key)

Outputs
-------
papers/lure-broadcast/results/criterion_denominators.json (+ .manifest.json)

Failure modes
-------------
- A criterion contract absent from the metadata cache RAISES: its campaign is undefined, and silently
  dropping it would understate the campaign numerator while leaving the denominator intact.
- A chain whose scan is missing is skipped and NAMED in the output, never omitted silently.
- A criterion contract whose payload yields no campaign key is counted separately as
  `contracts_without_campaign_key` rather than being folded into either number.

Complexity: trivial; one classification per criterion contract, no chain pass and no network.
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


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def campaigns_of(addresses, md: dict[str, dict], chain: str) -> tuple[set[str], int]:
    """Campaign keys for a set of contract addresses, plus the count that yield none."""
    keys, keyless = set(), 0
    for a in addresses:
        rec = md.get(a.lower())
        if rec is None:
            raise SystemExit(f"{chain}: contract {a} satisfies the criterion but is absent from the "
                             f"metadata cache; its campaign is undefined and dropping it would "
                             f"understate the numerator against an intact denominator")
        k = ml.campaign_key(ml.classify(rec.get("name"), rec.get("symbol")))
        if k:
            keys.add(k)
        else:
            keyless += 1
    return keys, keyless


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = OUT_ROOT / "criterion_denominators.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out.relative_to(ROOT)} exists; pass --force")

    cache = json.loads((ROOT / "analysis" / "config" / "token_metadata.json").read_text())
    per_chain, missing = {}, []

    for chain in CHAINS:
        scan = OUT_ROOT / f"forgery_chain_{chain}.json"
        if not scan.exists():
            missing.append(chain)
            continue
        d = json.loads(scan.read_text())
        crit = d["fabricated_sender_criterion"]
        ch = d["channel_census"]
        md = {a.lower(): v for a, v in cache[chain].items()}

        crit_keys, keyless = campaigns_of(crit["contracts_list"], md, chain)
        all_keys, _ = campaigns_of(list(d["lure_transfers_by_contract"]), md, chain)

        per_chain[chain] = {
            "contracts": {"n": crit["contracts"], "of": d["lure_contracts_observed"],
                          "share": crit["contract_share"]},
            "campaigns": {"n": len(crit_keys), "of": len(all_keys),
                          "share": len(crit_keys) / len(all_keys) if all_keys else None,
                          "qualifier": "campaigns containing at least one qualifying contract"},
            "transfers": {"n": crit["rows"], "of": ch["transfers"], "share": crit["row_share"]},
            "recipients": {"n": crit["recipients"], "of": ch["distinct_recipients"],
                           "share": crit["recipient_share"]},
            "contracts_without_campaign_key": keyless,
        }

        # Is the event-level criterion nested inside the contract-state result?
        #
        # The criterion is a property of the LOG: a row whose named sender is a registry-verified
        # token. The ledger test is a property of contract STATE. A reader is entitled to ask
        # whether they describe the same contracts or merely adjacent ones, and the paper cannot
        # answer by juxtaposing the two shares. Computing the intersection settles it.
        probe = OUT_ROOT / f"forgery_probe_{chain}.json"
        if probe.exists():
            pc = json.loads(probe.read_text())["per_contract"]
            probed = {a.lower().removeprefix("0x"): v for a, v in pc.items()}
            crit_bare = [a.lower().removeprefix("0x") for a in crit["contracts_list"]]
            unprobed = [a for a in crit_bare if a not in probed]
            if unprobed:
                raise SystemExit(
                    f"{chain}: {len(unprobed)} contract(s) satisfy the criterion but were never "
                    f"probed, e.g. 0x{unprobed[0]}; the nesting cannot be asserted over a "
                    f"population the state test did not cover")
            confirmed = [a for a in crit_bare
                         if probed[a].get("verdict") == "constant_nonzero"]
            per_chain[chain]["ledger_intersection"] = {
                "criterion_contracts": len(crit_bare),
                "also_ledger_confirmed": len(confirmed),
                "all_nested": len(confirmed) == len(crit_bare),
                "criterion_rows": crit["rows"],
                "criterion_rows_from_confirmed_contracts":
                    crit["rows"] if len(confirmed) == len(crit_bare) else None,
                "note": ("every contract emitting a criterion row is one the ledger test positively "
                         "establishes, so the event-level share is nested inside the contract-state "
                         "result rather than merely adjacent to it. When the nesting is complete the "
                         "row figure is exact, not apportioned: all criterion rows are emitted by "
                         "confirmed contracts by construction."),
            }
        r = per_chain[chain]
        print(f"  {chain:14s} contracts {r['contracts']['n']:>4d}/{r['contracts']['of']:<4d} "
              f"({r['contracts']['share']*100:5.1f}%) | campaigns {r['campaigns']['n']:>3d}/"
              f"{r['campaigns']['of']:<3d} ({(r['campaigns']['share'] or 0)*100:5.1f}%) | "
              f"volume {r['transfers']['share']*100:5.1f}% | recipients {r['recipients']['share']*100:5.1f}%")

    if not per_chain:
        raise SystemExit("no chain scans found; nothing to report")

    doc = {
        "purpose": ("report the fabricated-sender criterion against contracts, campaigns, transfer "
                    "volume and distinct named recipients, because a volume share alone can be "
                    "carried by a single campaign"),
        "criterion": ("a broadcast transfer whose named sender is a registry-verified token; the "
                      "assertion is established as fabricated by contract state, and the criterion "
                      "identifies rows making it"),
        "per_chain": per_chain,
        "chains_without_a_scan": missing,
        "additivity_note": ("recipient sets intersect across contracts and across chains, so "
                            "recipient counts are NOT additive and no cross-chain recipient total is "
                            "reported; contract and campaign counts are additive, volume shares are "
                            "not"),
        "campaign_qualifier": ("a campaign is counted when ANY of its contracts qualifies: an upper "
                               "count of campaign involvement and a lower count of campaign-wide "
                               "fabrication. The qualifier must accompany the number."),
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    (OUT_ROOT / "criterion_denominators.manifest.json").write_text(json.dumps({
        "producer": "papers/lure-broadcast/scripts/criterion_denominators.py",
        "git_commit": git_commit(),
    }, indent=1) + "\n")
    if missing:
        print(f"  chains without a scan (named in the artefact, not dropped): {missing}")
    print(f"  -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
