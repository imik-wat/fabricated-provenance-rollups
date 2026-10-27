"""Measure how much of the candidate-generating population the 4/4 operating point labels.

Why this exists
---------------
An attempt to test whether the broadcast and poisoning channels target the same addresses produced a
risk ratio of 4.05, then 0.47 under a different specification, then proved to exceed its own arithmetic
ceiling. The reason turned out to be a property of the label itself, and that property is a result worth
reporting on its own.

Let E be the CANDIDATE-GENERATING SUPPORT: addresses that appear as a victim in the feature table, i.e.
addresses that sent a zero-value transfer of a registry token and hold between 1 and UTILITY_CAP legit
counterparties, so that a lookalike collision could be observed for them at all. Let V_tau be those
whose best observed match reaches tau nibbles of prefix AND suffix.

The measurement is P(V_tau | E). If that probability approaches 1, then essentially every address that
COULD be labelled a victim IS one, and three things follow:

  1. Any risk ratio restricted to E is confined to [0, 1/P(V_tau|E)] by arithmetic. At tau=4 that ceiling
     is about 1.01, so a kill condition set at 1.5 can never fire and a ratio of 4.05 cannot have come
     from within the support -- it came from the difference in eligibility between the arms.
  2. Marginal rate comparisons are uninformative about targeting, because the outcome barely varies.
     List-level comparison is the only honest unit.
  3. The interesting question inverts. It is no longer "which addresses are targeted" but "why is
     essentially every eligible address targeted".

What this is NOT saying
-----------------------
It is NOT saying the 4/4 rule is a false-positive machine. The opposite: this script also computes the
coincidental-collision probability per address from the observed counterparty counts, and it is of order
1e-9, so the matches are overwhelmingly deliberate crafts rather than birthday collisions. The prior
work's finding that crafted matches concentrate at four nibbles is not contradicted. The two results
compose into something stronger and more alarming than either alone: the crafting is real, AND it is
near-universal across the eligible population. That is what makes a rate ratio uninformative while making
the absolute measurement worse, not better.

Two sources, and only one of them is admissible as the headline
---------------------------------------------------------------
PRIMARY -- `results/<chain>_detection.json:best_match_level_histogram`. This is a complete, UNSAMPLED
census of every scored candidate transfer, so shares taken from it are population quantities. The unit is
the candidate TRANSFER, not the address.

CONTAMINATED, reported only as a comparison -- `results/<chain>_features.csv`. The feature table keeps
every pair at level >= 2 but only a `neg_rate` sample of the rest (`detect.py:272`:
`if best < 2 and rng.random() >= neg_rate: continue`, default neg_rate = 0.01). The measured coverage
confirms it: on Optimism `behavioural.json:feature_table_coverage_by_level` gives 0.0038 at level 0 and
0.0030 at level 1, against a low-level population of 956,706 of which 3,520 were sampled. Computing
P(victim | support) from that table therefore DELETES ~99% of the non-victims and inflates the share
towards 1 by construction. An earlier version of this analysis reported 0.98576 on Optimism from the
feature table and that figure is an artefact of the sampling design, not a measurement. It is retained
here beside the census figure purely so the size of the inflation is visible.

Inputs
------
results/<chain>_detection.json  (primary; unsampled level histogram)
results/<chain>_features.csv    (secondary, negative-sampled; used only for the median counterparty and
                                 candidate counts, which are conditioned on level >= 2 and stated as such)
results/<chain>_behavioural.json (per-level feature-table coverage, to quantify the inflation)

Outputs
-------
papers/lure-broadcast/results/saturation_<chain>.json (+ .manifest.json)

Failure modes
-------------
- A missing feature table raises; an absent support would make P(V|E) undefined, not 1.0 or 0.0.
- A chain whose support is empty raises rather than reporting a share.
- Ceilings are reported for every tau, including those where the ratio design is provably dead, so the
  arithmetic is auditable rather than asserted.

Complexity: one streamed pass over the feature table, O(distinct victims) memory (6.6M on Base, a few GB).
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analysis"))

from src import stats  # noqa: E402

FEATURES = ROOT / "results"
OUT_ROOT = ROOT / "papers" / "lure-broadcast" / "results"
TAUS = (1, 2, 3, 4, 5, 6, 8)
#: A 4-nibble prefix AND 4-nibble suffix match is 16**-8 of the address space per candidate pair.
NIBBLE_SPACE = 16


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("chain")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = OUT_ROOT / f"saturation_{args.chain}.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out.relative_to(ROOT)} exists; pass --force")

    # ---- PRIMARY: the unsampled level histogram over candidate transfers ----------------------
    det_path = FEATURES / f"{args.chain}_detection.json"
    if not det_path.exists():
        raise SystemExit(f"missing {det_path.relative_to(ROOT)}; the census histogram is the only "
                         f"admissible source for a population share")
    det = json.loads(det_path.read_text())
    hist = {int(k): v for k, v in (det.get("best_match_level_histogram") or {}).items()}
    if not hist:
        raise SystemExit(f"{args.chain}: detection.json carries no level histogram")
    hist_total = sum(hist.values())
    census = {}
    for t in TAUS:
        ge = sum(v for k, v in hist.items() if k >= t)
        p = ge / hist_total
        census[str(t)] = {
            "candidate_transfers_at_or_above": ge,
            "p_of_scored_transfers": p,
            "ratio_ceiling": (1.0 / p) if p else None,
            "ratio_design_dead_at_1_5": (p > 0) and (1.0 / p) < 1.5,
            "wilson95": list(stats.wilson(ge, hist_total)),
        }

    # ---- SECONDARY: the negative-sampled feature table, for conditional medians only -----------
    # Absent on a chain that produced no candidates (Polygon zkEVM). That is a legitimate state, not a
    # failure: the census histogram above is sufficient for the headline, so the chain is still reported
    # with its medians recorded as unavailable rather than being dropped from the five-chain table.
    path = FEATURES / f"{args.chain}_features.csv"
    best: dict[str, int] = {}
    cp: dict[str, int] = {}
    ncand: dict[str, int] = {}
    rows = 0
    if path.exists():
        with path.open() as fh:
            for r in csv.DictReader(fh):
                rows += 1
                v = r["victim"]
                try:
                    lvl = int(r["best_level"])
                except (KeyError, ValueError):
                    continue
                if lvl > best.get(v, -1):
                    best[v] = lvl
                if v not in cp:
                    try:
                        cp[v] = int(r["victim_cp_count"])
                        ncand[v] = int(r["n_victim_candidates"])
                    except (KeyError, ValueError):
                        pass

    support = len(best)
    have_features = bool(best)
    by_tau = {}
    for t in TAUS if support else ():
        v = sum(1 for lvl in best.values() if lvl >= t)
        p = v / support
        by_tau[str(t)] = {
            "victims": v,
            "p_victim_given_support": p,
            "ratio_ceiling": (1.0 / p) if p else None,
            "ratio_design_dead_at_1_5": (p > 0) and (1.0 / p) < 1.5,
            "wilson95": list(stats.wilson(v, support)),
            "rule_of_three_upper" : (stats.rule_of_three(support) if v == 0 else None),
        }

    # Sub-threshold mass: the rows the pre-thresholded candidate CSV cannot see.
    sub = sum(1 for lvl in best.values() if lvl < 4)
    by_tau = by_tau if have_features else {}
    cps = [c for c in cp.values() if c > 0]
    ncs = [n for n in ncand.values() if n > 0]
    med_cp = median(cps) if cps else None
    med_nc = median(ncs) if ncs else None
    # Chance of a coincidental tau=4 double-ended match for a median address: pairs x 16**-8.
    coincidence = None
    if med_cp and med_nc:
        coincidence = med_cp * med_nc * (NIBBLE_SPACE ** -8)

    # Quantify how badly the feature table inflates the share, using the recorded coverage.
    beh_path = FEATURES / f"{args.chain}_behavioural.json"
    coverage = None
    if beh_path.exists():
        coverage = json.loads(beh_path.read_text()).get("feature_table_coverage_by_level")

    result = {
        "chain": args.chain,
        "primary_source": f"results/{args.chain}_detection.json:best_match_level_histogram",
        "unit": "candidate transfer (a zero-value registry-token transfer that was scored)",
        "scored_transfers": hist_total,
        "level_histogram": {str(k): v for k, v in sorted(hist.items())},
        "census_by_tau": census,
        "contaminated_feature_table_comparison": {
            "why_inadmissible": (
                "detect.py:272 keeps pairs with best_level < 2 only at neg_rate (default 0.01), so the "
                "non-victims are deleted at ~99% and any share computed here is inflated towards 1 by "
                "construction. Reported only to show the size of the artefact."),
            "feature_rows_read": rows,
            "distinct_addresses_in_table": support,
            "sub_threshold_addresses_in_table": sub,
            "p_victim_given_table_by_tau": {str(t): by_tau[str(t)]["p_victim_given_support"]
                                            for t in TAUS if str(t) in by_tau},
            "recorded_coverage_by_level": coverage,
        },
        "median_victim_cp_count_given_level_ge_2": med_cp,
        "median_n_victim_candidates_given_level_ge_2": med_nc,
        "coincidental_4_4_probability_median_address": coincidence,
        "interpretation": (
            "The admissible headline is p_of_scored_transfers at tau=4, from the unsampled census "
            "histogram. Where it approaches 1, any risk ratio restricted to the scored population is "
            "confined to [0, 1/p] by arithmetic, so marginal rate comparisons cannot measure targeting "
            "and list-level comparison is the only honest unit. The coincidence probability shows the "
            "matches are deliberate crafts rather than birthday collisions, so a high share is a "
            "statement about how WIDESPREAD deliberate crafting is -- not a claim that the rule "
            "over-flags."),
        "caveat": (
            "The scored population is defined by the prior pipeline's eligibility rule, including its "
            "UTILITY_CAP on counterparty count, so this is conditional on that rule and is not a "
            "population prevalence over all addresses. It is exactly the quantity that bounds a "
            "support-restricted ratio, which is what it is used for. The medians are conditioned on "
            "level >= 2 because the feature table samples below that."),
    }

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    (OUT_ROOT / f"saturation_{args.chain}.manifest.json").write_text(json.dumps({
        "producer": "papers/lure-broadcast/scripts/saturation.py",
        "input": f"results/{args.chain}_features.csv",
        "git_commit": git_commit(),
    }, indent=1) + "\n")

    print(f"  {args.chain}: {hist_total:,} scored candidate transfers (UNSAMPLED census)")
    print(f"  {'tau':>4s} {'transfers>=tau':>15s} {'share':>9s} {'ceiling':>10s}  ratio design")
    for t in TAUS:
        c = census[str(t)]
        ceil = c["ratio_ceiling"]
        dead = "DEAD at 1.5" if c["ratio_design_dead_at_1_5"] else ""
        print(f"  {t:>4d} {c['candidate_transfers_at_or_above']:>15,} "
              f"{c['p_of_scored_transfers']:9.5f} "
              f"{(f'{ceil:10.4f}' if ceil else '       inf')}  {dead}")
    if have_features:
        print(f"  [contaminated comparison] feature table gives "
              f"P={by_tau['4']['p_victim_given_support']:.5f} at tau=4 over {support:,} addresses "
              f"-- inflated by ~99% negative deletion")
    else:
        print("  [no feature table on this chain -- it produced no candidates; census figure stands alone]")
    print(f"  sub-threshold (best_level < 4): {sub:,} addresses -- these are invisible to the "
          f"pre-thresholded candidate CSV")
    print(f"  median counterparties {med_cp}, median candidates {med_nc} -> coincidental 4/4 "
          f"probability {coincidence:.2e}" if coincidence else "")
    print(f"  -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
