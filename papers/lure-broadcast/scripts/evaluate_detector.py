"""Evaluate the lure detector against human labels, weighted back to the population.

Why the weighting is not optional
---------------------------------
The annotation sample is STRATIFIED, and deliberately far from proportional: the detector's `strong`
strata were sampled at 100%, 69% and 55%, while the `none` strata were sampled at 35%, 26% and 21%.
Computing precision or recall by counting sample rows would therefore report a detector evaluated on a
population that does not exist -- flagged contracts are over-represented by roughly a factor of three.
Every rate here is a stratified estimate: per-stratum sample rates re-weighted by per-stratum population
counts, with a finite-population correction because the strata are sampled at appreciable fractions.

Precision and recall need DIFFERENT denominators, and only one of them is confined to the flagged strata:
  precision = P(lure | flagged)   -- denominator is the population of flagged contracts, i.e. the
                                     `strong` strata only.
  recall    = P(flagged | lure)   -- denominator is ALL lures in the population, which is why the sample
                                     has a negative stratum. Without it, recall is not estimable at all
                                     and only precision could be reported.

What this script will NOT do
---------------------------
It will not emit Cohen's kappa. Kappa measures agreement between two independent raters; the input
carries ONE label column. With a single rater there is no agreement to measure, and a number computed by
comparing the rater against the detector would be a measure of the detector, not of label reliability --
reporting that as kappa would be a fabrication. The script records `kappa: null` with the reason, and
kill condition K3's third leg stays open until a second rater exists.

Handling of `unclear` and unlabelled rows
----------------------------------------
Neither is folded into a class. The primary estimate excludes them, and a sensitivity analysis bounds the
result by counting every ambiguous row first as a lure and then as a non-lure. If those bounds straddle a
claim, the claim is not supported by this sample.

Inputs
------
papers/lure-broadcast/results/annotation/<chain>_sample_blinded.csv   (with `label` filled in)
papers/lure-broadcast/results/annotation/<chain>_answer_key.json      (strata, populations, verdicts)

Outputs
-------
papers/lure-broadcast/results/detector_eval_<chain>.json (+ .manifest.json)

Failure modes
-------------
- An off-vocabulary label raises, rather than being silently dropped from a denominator.
- A row_id in the annotations that is absent from the key raises: the two files must describe the same
  draw, or the weights are wrong.
- A stratum with zero labelled rows is reported as unestimable rather than contributing a zero rate.

Complexity: trivial. Milliseconds.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analysis"))

from src import stats  # noqa: E402

ANN = ROOT / "papers" / "lure-broadcast" / "results" / "annotation"
OUT_ROOT = ROOT / "papers" / "lure-broadcast" / "results"

VOCAB = {"lure", "lure-nodest", "dest-only", "legitimate", "unclear"}
#: Only `lure` is a positive for the binary task the detector performs. `lure-nodest` and `dest-only`
#: are deliberately negatives: the detector's `strong` bar requires BOTH a destination and a call to
#: action, so a string with only one of them is correctly not flagged.
POSITIVE = {"lure"}
AMBIGUOUS = {"unclear", ""}


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def stratified(cells: list[tuple[int, int, int]]) -> tuple[float | None, float | None]:
    """(estimate, se) for a proportion over strata given [(N_h, k_h, n_h), ...].

    Weights are N_h / sum(N_h) over the strata SUPPLIED, so the caller controls the denominator (that is
    what makes precision and recall different calls). Returns (None, None) if no stratum has samples.
    """
    usable = [(N, k, n) for N, k, n in cells if n > 0 and N > 0]
    if not usable:
        return None, None
    total = sum(N for N, _, _ in usable)
    quads = [(N / total, k, n, N) for N, k, n in usable]
    return stats.stratified_proportion(quads)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("chain")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--reclassify", action="store_true",
                    help="re-run the CURRENT detector instead of the verdict frozen in the answer key; "
                         "use to measure a detector change against the same frozen labels")
    args = ap.parse_args()

    suffix = "_reclassified" if "--reclassify" in sys.argv else ""
    out_path = OUT_ROOT / f"detector_eval_{args.chain}{suffix}.json"
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path.relative_to(ROOT)} exists; pass --force")

    key = json.loads((ANN / f"{args.chain}_answer_key.json").read_text())
    rows = list(csv.DictReader((ANN / f"{args.chain}_sample_blinded.csv").open()))

    bad = {r["label"].strip() for r in rows} - VOCAB - {""}
    if bad:
        raise SystemExit(f"off-vocabulary labels {sorted(bad)}; fix the annotation rather than "
                         f"dropping them from a denominator")

    # Optionally re-run the CURRENT detector instead of trusting the verdict frozen into the key at
    # sample-draw time. The frozen verdict is the right default (it is what the strata were built from,
    # and it preserves provenance), but comparing a re-classified run against the SAME frozen labels is
    # how a detector change is measured without re-annotating.
    reclassified = {}
    if args.reclassify:
        md = json.loads((ROOT / "analysis" / "config" / "token_metadata.json").read_text())
        chain_md = md.get(args.chain) or {}
        lookup = {a.lower(): m for a, m in chain_md.items()}
        from src import metadata_lure as ml  # local import: only needed on this path
        for rid, meta in key["rows"].items():
            m = lookup.get(meta["address"], {})
            reclassified[rid] = ml.classify(m.get("name"), m.get("symbol")).strength == "strong"

    # Join annotations to the FIXED sampling strata. The strata define the weights and are immutable --
    # they were drawn with known N_h and n_h. But flagged-ness must NOT be treated as a stratum
    # property: under --reclassify a single stratum can contain both flagged and unflagged contracts, so
    # the confusion matrix is accumulated per (stratum, flagged, human_label) CELL and each cell is
    # expanded to the population by its own stratum's N_h/n_h. That expansion is correct whether or not
    # flagged-ness happens to align with the strata.
    per_stratum: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "pos": 0, "ambiguous": 0, "flagged": None,
                 "tp": 0, "fp": 0, "fn": 0, "tn": 0, "amb_flagged": 0, "amb_unflagged": 0})
    for r in rows:
        rid = r["row_id"]
        if rid not in key["rows"]:
            raise SystemExit(f"{rid} is not in the answer key; the annotation and the draw disagree")
        meta = key["rows"][rid]
        s = per_stratum[meta["stratum"]]
        flagged = reclassified[rid] if args.reclassify else (meta["strength"] == "strong")
        # Retained for the as-drawn report; under --reclassify a stratum may be mixed, recorded below.
        if s["flagged"] is None:
            s["flagged"] = flagged
        elif s["flagged"] != flagged:
            s["flagged"] = "mixed"
        lab = r["label"].strip()
        if lab in AMBIGUOUS:
            s["ambiguous"] += 1
            s["amb_flagged" if flagged else "amb_unflagged"] += 1
            continue
        s["n"] += 1
        is_pos = lab in POSITIVE
        s["pos"] += 1 if is_pos else 0
        if flagged and is_pos:
            s["tp"] += 1
        elif flagged and not is_pos:
            s["fp"] += 1
        elif not flagged and is_pos:
            s["fn"] += 1
        else:
            s["tn"] += 1

    pop = key["strata_population"]
    detail = {}
    for st, s in sorted(per_stratum.items()):
        detail[st] = {
            "population": pop[st], "labelled": s["n"], "ambiguous": s["ambiguous"],
            "human_lure": s["pos"],
            "human_lure_rate": (s["pos"] / s["n"]) if s["n"] else None,
            "detector_flagged": s["flagged"],
        }

    # Horvitz-Thompson expansion of each confusion cell by its own stratum's sampling fraction.
    TP = FP = FN = TN = 0.0
    for st, s in per_stratum.items():
        if not s["n"]:
            continue
        w = pop[st] / s["n"]                 # inverse sampling fraction for this stratum
        TP += w * s["tp"]; FP += w * s["fp"]; FN += w * s["fn"]; TN += w * s["tn"]

    prec = (TP / (TP + FP)) if (TP + FP) else None
    recall = (TP / (TP + FN)) if (TP + FN) else None
    f1 = (2 * prec * recall / (prec + recall)) if (prec and recall and prec + recall) else None
    lures_flagged, lures_unflagged = TP, FN
    lures_total = TP + FN
    n_flagged_pop = TP + FP

    # The stratified SE is only meaningful when flagged-ness aligns with the strata (the as-drawn case);
    # under --reclassify the flagged set cuts across strata and this SE is not the right variance, so it
    # is suppressed rather than reported misleadingly.
    flagged_st = [st for st, s in per_stratum.items() if s["flagged"] is True]
    unflagged_st = [st for st, s in per_stratum.items() if s["flagged"] is False]
    mixed_st = [st for st, s in per_stratum.items() if s["flagged"] == "mixed"]
    if mixed_st:
        prec_se = None
    else:
        _, prec_se = stratified([(pop[st], per_stratum[st]["pos"], per_stratum[st]["n"])
                                 for st in flagged_st])

    # sensitivity: bound the estimate by resolving every ambiguous row both ways
    def bound(as_lure: bool):
        cs_f, cs_u = [], []
        for st in flagged_st:
            s = per_stratum[st]
            n = s["n"] + s["ambiguous"]
            k = s["pos"] + (s["ambiguous"] if as_lure else 0)
            cs_f.append((pop[st], k, n))
        for st in unflagged_st:
            s = per_stratum[st]
            n = s["n"] + s["ambiguous"]
            k = s["pos"] + (s["ambiguous"] if as_lure else 0)
            cs_u.append((pop[st], k, n))
        p, _ = stratified(cs_f)
        lf = sum(N * k / n for N, k, n in cs_f if n)
        lu = sum(N * k / n for N, k, n in cs_u if n)
        return p, (lf / (lf + lu) if (lf + lu) else None)

    # Counting ambiguous rows as lures RAISES precision (more positives among the flagged) but LOWERS
    # recall (it adds lures to the denominator, and most ambiguous rows sit in unflagged strata). So the
    # two bounds run in opposite directions and each range must be sorted before it is reported --
    # an unsorted [0.9345, 0.8013] reads as a malformed interval.
    p_a, r_a = bound(False)
    p_b, r_b = bound(True)
    prec_range = sorted(x for x in (p_a, p_b) if x is not None)
    rec_range = sorted(x for x in (r_a, r_b) if x is not None)

    # Where do the misses fall? This is more informative than the recall number alone: a miss in the
    # detector's own `weak` band is a threshold question, whereas a miss in `none` is a blind spot.
    miss_by_band: dict[str, dict] = {}
    for st in unflagged_st:
        s = per_stratum[st]
        band = st.split("|", 1)[0]
        b = miss_by_band.setdefault(band, {"labelled": 0, "human_lure": 0, "population": 0})
        b["labelled"] += s["n"]
        b["human_lure"] += s["pos"]
        b["population"] += pop[st]

    # A saturated estimate (every sampled row positive) yields a stratified SE of exactly 0, because
    # p(1-p) vanishes in every stratum. That is arithmetically right and epistemically wrong: 80/80 is
    # not infinite precision. Report a Wilson interval on the pooled flagged sample, plus the
    # rule-of-three bound on the unobserved false-positive rate.
    # Count from the confusion CELLS, not from whole strata: under --reclassify the flagged set cuts
    # across strata, so summing stratum totals would silently omit the newly-promoted contracts and
    # report a Wilson interval on the wrong sample size.
    flagged_pos = sum(s["tp"] for s in per_stratum.values())
    flagged_n = flagged_pos + sum(s["fp"] for s in per_stratum.values())
    prec_wilson = stats.wilson(flagged_pos, flagged_n) if flagged_n else None
    fp_upper = stats.rule_of_three(flagged_n) if (flagged_n and flagged_pos == flagged_n) else None

    result = {
        "chain": args.chain,
        "raters": 1,
        "detector_source": ("recomputed with the current detector" if args.reclassify
                            else "frozen in the answer key at sample-draw time"),
        "strata_mixed_under_reclassification": sorted(mixed_st),
        "kappa": None,
        "kappa_reason": ("Cohen's kappa requires two independent raters; the annotation carries one "
                         "label column. Comparing the single rater against the detector would measure "
                         "the detector, not label reliability. K3's third leg remains open."),
        "sample": {"rows": len(rows), "labelled": sum(s["n"] for s in per_stratum.values()),
                   "ambiguous_or_blank": sum(s["ambiguous"] for s in per_stratum.values())},
        "population": {"total": key["population_total"], "flagged": n_flagged_pop},
        "estimates_population_weighted": {
            "precision": prec, "precision_se": prec_se,
            "precision_se_is_degenerate": (prec_se == 0.0),
            "precision_se_caveat": (
                "A stratified SE of exactly 0 arises because every sampled row in every flagged "
                "stratum was judged a lure, so p(1-p) vanishes. It is NOT a claim of infinite "
                "precision; use precision_wilson95 and false_positive_rate_upper95 instead."),
            "precision_wilson95": prec_wilson,
            "false_positive_rate_upper95": fp_upper,
            "precision_sample": [flagged_pos, flagged_n],
            "recall": recall,
            "f1": f1,
            "estimated_population_lures": lures_total,
            "estimated_lures_flagged": lures_flagged,
            "estimated_lures_missed": lures_unflagged,
        },
        "sensitivity_ambiguous_resolved": {
            "precision_range": prec_range, "recall_range": rec_range,
            "note": ("Each range is sorted. Resolving every unclear/blank row as a lure RAISES precision "
                     "but LOWERS recall, because it adds lures to the recall denominator and most "
                     "ambiguous rows fall in unflagged strata -- the two bounds run in opposite "
                     "directions."),
        },
        "miss_location": {
            "by_detector_band": miss_by_band,
            "interpretation": (
                "A miss in the detector's own `weak` band is a THRESHOLD question -- the string was seen "
                "and downgraded. A miss in `none` is a blind spot. Which band the misses fall in "
                "determines whether recall is recoverable by re-tuning or requires a new signal."),
        },
        "unweighted_for_comparison_only": {
            "precision": (sum(per_stratum[st]["pos"] for st in flagged_st) /
                          max(sum(per_stratum[st]["n"] for st in flagged_st), 1)),
            "warning": ("Sample rates differ ~3x between flagged and unflagged strata, so this figure "
                        "describes no real population. Recorded only so the weighted/unweighted gap is "
                        "visible; it must never be published."),
        },
        "per_stratum": detail,
        "positive_class": sorted(POSITIVE),
        "negative_class_note": ("lure-nodest and dest-only count as NEGATIVES: the detector's strong bar "
                                "requires both a destination and a call to action, so a string with only "
                                "one is correctly unflagged."),
    }

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    (OUT_ROOT / f"detector_eval_{args.chain}{suffix}.manifest.json").write_text(json.dumps({
        "producer": "papers/lure-broadcast/scripts/evaluate_detector.py",
        "git_commit": git_commit(),
        "codebook": "docs/ANNOTATION_CODEBOOK.md v1.0",
        "inputs": [f"annotation/{args.chain}_sample_blinded.csv", f"annotation/{args.chain}_answer_key.json"],
    }, indent=1) + "\n")

    w = result["estimates_population_weighted"]
    print(f"  raters: 1  ->  kappa NOT computed (needs two)")
    print(f"  labelled {result['sample']['labelled']} of {len(rows)} "
          f"({result['sample']['ambiguous_or_blank']} unclear/blank)")
    print(f"\n  POPULATION-WEIGHTED (n={key['population_total']} contracts, "
          f"{n_flagged_pop} flagged)")
    pw = w["precision_wilson95"]
    se_txt = ("   (stratified SE is 0 and DEGENERATE -- saturated estimate)"
              if w.get("precision_se_is_degenerate") else
              ("   (SE suppressed: flagged set cuts across strata)" if w["precision_se"] is None else ""))
    print(f"    precision {w['precision']:.4f}  "
          f"Wilson95 [{pw[0]:.4f}, {pw[1]:.4f}] on {flagged_pos}/{flagged_n}"
          + se_txt)
    if w["false_positive_rate_upper95"] is not None:
        print(f"              zero false positives observed; FP rate < "
              f"{w['false_positive_rate_upper95']:.4f} at 95%")
    print(f"    recall    {w['recall']:.4f}  sensitivity range "
          f"[{rec_range[0]:.4f}, {rec_range[-1]:.4f}]")
    print(f"    F1        {w['f1']:.4f}")
    print(f"    estimated lures in population {w['estimated_population_lures']:.0f} "
          f"({w['estimated_lures_flagged']:.0f} flagged, {w['estimated_lures_missed']:.0f} missed)")
    print(f"\n  WHERE THE MISSES ARE (this decides if recall is recoverable by re-tuning)")
    for band, b in sorted(miss_by_band.items()):
        rate = (b["human_lure"] / b["labelled"]) if b["labelled"] else 0.0
        print(f"    band {band:6s} population {b['population']:5d}  labelled {b['labelled']:4d}  "
              f"human lures {b['human_lure']:3d}  ({rate:.1%})")
    print(f"\n  unweighted precision would be "
          f"{result['unweighted_for_comparison_only']['precision']:.4f} -- do not publish")
    print(f"\n  -> {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
