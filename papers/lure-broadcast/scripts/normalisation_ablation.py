"""Ablate the normalisation step, and test Proposition 1 empirically on all five chains.

What Proposition 1 predicts
---------------------------
Campaigns are equivalence classes of contracts under `dom . sigma . pl`. Proposition 1 states that,
for a delimiter-preserving sigma, the partition induced WITHOUT the fold REFINES the partition
induced WITH it: omitting sigma can only split a campaign into several, never merge two distinct
campaigns into one. Two consequences are checkable against data rather than argued:

  (P1a) COUNT.       The unnormalised partition has at least as many blocks as the normalised one.
  (P1b) REFINEMENT.  Every unnormalised block lies inside a single normalised block. This is the
                     strong form; (P1a) follows from it but not conversely, so a run that satisfies
                     only the count has NOT confirmed the proposition.

A violation of (P1b) would mean sigma had separated two contracts that the raw strings identified,
which the proof forbids. Any violation is reported with its witnesses rather than summarised, since
it would falsify a proved statement and therefore indicate a defect in the character tables or in
the delimiter-preservation hypothesis of Definition 5.

Design
------
The ablation runs `classify(..., normalise=False)`, a keyword whose default preserves the original
behaviour exactly, so the ablated arm differs from the reported arm in the fold and in nothing else.
Both arms are computed over the identical contract population.

Contracts that are unflagged in an arm have no campaign in that arm and are excluded from that arm's
partition; the counts of flagged contracts are reported per arm, because the fold also changes how
many payloads are detected at all, and conflating "fewer campaigns" with "fewer detections" would
misattribute the effect.

Inputs
------
analysis/config/token_metadata.json (all chains present)

Outputs
-------
papers/lure-broadcast/results/normalisation_ablation.json (+ .manifest.json)

Failure modes
-------------
- A chain absent from the metadata cache raises rather than being skipped, since a silently missing
  chain would make the ablation look narrower than claimed.
- Refinement violations do NOT raise: they are the scientifically interesting outcome and must be
  reported, not suppressed. They are surfaced in the summary line and in the JSON.

Complexity: two classifications per contract; a few thousand contracts per chain.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
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


def partition(md: dict, normalise: bool) -> tuple[dict[str, str], int]:
    """contract -> campaign key, plus the number of contracts flagged `strong` in this arm."""
    keys: dict[str, str] = {}
    flagged = 0
    for addr, rec in md.items():
        v = ml.classify(rec.get("name"), rec.get("symbol"), normalise=normalise)
        if v.strength != "strong":
            continue
        flagged += 1
        k = ml.campaign_key(v)
        if k:
            keys[addr.lower()] = k
    return keys, flagged


#: Constructed payloads exercising each folding stage. These are POSITIVE CONTROLS for the ablation
#: itself: if the measured population shows no effect, the reader must be able to distinguish
#: "the fold does nothing" from "the ablation switch does nothing". Each case must differ between
#: the two arms, and the pure-ASCII case must NOT differ.
WITNESSES = [
    ("cyrillic_homoglyph_in_host", "www.fomo\u0445.club claim now"),
    ("cyrillic_c_in_tld",          "visit www.fomox.\u0441lub now"),
    ("bracket_dot_obfuscation",    "claim at bonus-zk[.]com"),
    ("zero_width_in_host",         "claim at bonus\u200bzk.com"),
    ("ascii_control_case",         "claim at bonus-zk.com"),
]


def witness_table() -> list[dict]:
    rows = []
    for label, s in WITNESSES:
        a = ml.classify(s, None, normalise=True)
        b = ml.classify(s, None, normalise=False)
        rows.append({
            "case": label,
            "normalised_strength": a.strength,
            "raw_strength": b.strength,
            "normalised_campaign": ml.campaign_key(a),
            "raw_campaign": ml.campaign_key(b),
            "differs": (ml.campaign_key(a) != ml.campaign_key(b)) or (a.strength != b.strength),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = OUT_ROOT / "normalisation_ablation.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out.relative_to(ROOT)} exists; pass --force")

    wit = witness_table()
    control = next(w for w in wit if w["case"] == "ascii_control_case")
    if control["differs"]:
        raise SystemExit("the pure-ASCII control differs between arms; the ablation is changing "
                         "more than the fold and its result cannot be interpreted")
    if not all(w["differs"] for w in wit if w["case"] != "ascii_control_case"):
        raise SystemExit("a constructed confusable case did NOT differ between arms; the ablation "
                         "switch is a no-op and a null result would be meaningless")
    print(f"  ablation verified on {len(wit) - 1} constructed cases plus 1 ASCII control")

    cache = json.loads((ROOT / "analysis" / "config" / "token_metadata.json").read_text())
    per_chain: dict[str, dict] = {}
    tot_viol = 0

    for chain in CHAINS:
        if chain not in cache:
            raise SystemExit(f"{chain} absent from the metadata cache; run run_metadata_sweep.py "
                             f"for it. Skipping would make the ablation look broader than it is.")
        md = cache[chain]
        with_keys, with_flagged = partition(md, normalise=True)
        without_keys, without_flagged = partition(md, normalise=False)

        with_blocks = len(set(with_keys.values()))
        without_blocks = len(set(without_keys.values()))

        # (P1b) every unnormalised block must lie inside ONE normalised block. Checked only over
        # contracts present in both arms, since a contract absent from an arm has no block there.
        both = set(with_keys) & set(without_keys)
        spans: dict[str, set[str]] = defaultdict(set)
        for a in both:
            spans[without_keys[a]].add(with_keys[a])
        violations = {k: sorted(v) for k, v in spans.items() if len(v) > 1}
        tot_viol += len(violations)

        # Campaigns the fold MERGES: a normalised block covering several unnormalised ones. This is
        # the predicted direction, and its size is the ablation's effect.
        merged: dict[str, set[str]] = defaultdict(set)
        for a in both:
            merged[with_keys[a]].add(without_keys[a])
        merges = {k: sorted(v) for k, v in merged.items() if len(v) > 1}

        per_chain[chain] = {
            "contracts_in_metadata": len(md),
            "flagged_with_normalisation": with_flagged,
            "flagged_without_normalisation": without_flagged,
            "campaigns_with_normalisation": with_blocks,
            "campaigns_without_normalisation": without_blocks,
            "campaign_delta": without_blocks - with_blocks,
            "count_prediction_holds": without_blocks >= with_blocks,
            "refinement_violations": violations,
            "refinement_holds": not violations,
            "campaigns_merged_by_normalisation": merges,
            "n_campaigns_merged": len(merges),
        }
        print(f"  {chain:15s} flagged {with_flagged:>4d} -> {without_flagged:<4d} | "
              f"campaigns {with_blocks:>3d} -> {without_blocks:<3d} "
              f"(delta {without_blocks - with_blocks:+d}) | "
              f"merges {len(merges):>2d} | refinement "
              f"{'HOLDS' if not violations else f'VIOLATED x{len(violations)}'}")

    doc = {
        "proposition": ("Proposition 1: for delimiter-preserving sigma, the partition induced "
                        "WITHOUT the fold refines the partition induced WITH it -- omitting sigma "
                        "can only split a campaign, never merge two."),
        "checks": {
            "P1a_count": "campaigns_without_normalisation >= campaigns_with_normalisation",
            "P1b_refinement": ("every unnormalised block lies inside a single normalised block; "
                               "this is the strong form and is what actually tests the proposition"),
        },
        "ablation": ("classify(..., normalise=False) skips the confusable/invisible/emoji fold and "
                     "nothing else; the default preserves the reported behaviour exactly"),
        "witnesses": wit,
        "witness_note": ("positive controls for the ablation itself: each constructed case must "
                         "differ between arms and the ASCII control must not, so that a null result "
                         "on real data cannot be confused with an inert switch"),
        "chains": CHAINS,
        "per_chain": per_chain,
        "refinement_holds_on_all_chains": tot_viol == 0,
        "total_violations": tot_viol,
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    (OUT_ROOT / "normalisation_ablation.manifest.json").write_text(json.dumps({
        "producer": "papers/lure-broadcast/scripts/normalisation_ablation.py",
        "git_commit": git_commit(),
    }, indent=1) + "\n")

    print(f"\n  refinement holds on all chains: {tot_viol == 0} "
          f"({tot_viol} violation(s) in total)")
    print(f"  -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
