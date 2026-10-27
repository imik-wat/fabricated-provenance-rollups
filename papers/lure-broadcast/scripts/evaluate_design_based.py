"""Detector evaluation with intervals consistent with the sampling design (reviewer items 5 and 6).

What was wrong before
---------------------
The point estimates were correct: per-stratum rates expanded by N_h/n_h, which is the right estimator
for a stratified sample in which flagged contracts are deliberately over-represented. The INTERVALS
were not. A Wilson interval was reported on pooled counts, and a Wilson interval is exact for a
single binomial proportion drawn with replacement -- not for a ratio of Horvitz--Thompson expansions
over a weighted, without-replacement design. Pooling discards the weights and the finite-population
correction together.

Two further presentational faults are fixed here. Expanded counts are fractional by construction
(an expected 146.5 flagged contracts), which reads as a mistake unless the raw sample counts are
given alongside; and the full confusion matrix was never published, so recall could not be checked.

What this produces
------------------
* the per-stratum table the design rests on: definition, N_h, n_h, sampling fraction, and the four
  cell counts;
* a confusion matrix in BOTH forms -- raw sampled rows, and population-expanded expectations;
* precision, recall and F1 as point estimates with finite-population bootstrap intervals
  (`stats.stratified_bootstrap_ci`), which reproduce a fully sampled stratum exactly and therefore
  handle the unanimous strata that drove the plug-in standard error to zero.

Definitions, stated because the reviewer is right that they were implicit
-------------------------------------------------------------------------
flagged  -- the CURRENT detector's `strong` band, recomputed at evaluation time. The STRATA are
            frozen at sample-draw time because they define the weights, but flagged-ness is the
            thing under test and must not be read from the frozen key: doing so evaluates whichever
            detector drew the sample. The two differ here, and materially -- recall reads 0.8934 on
            the frozen band against 0.9915 on the current one -- so the distinction is not academic.
            Once flagged-ness is recomputed it is no longer a stratum property, which is why the
            cells are expanded per row rather than per stratum.
positive -- both raters independently judged the payload a full lure; the agreed set only.
TP/FP/FN/TN follow from the two indicators. precision = P(positive | flagged),
recall = P(flagged | positive). Recall is estimable only because the sample includes unflagged
strata; without them its denominator does not exist.

Inputs
------
papers/lure-broadcast/results/annotation/<chain>_answer_key.json   (strata, N_h, verdicts)
papers/lure-broadcast/results/annotation/<chain>_sample_blinded.csv        (rater 1)
papers/lure-broadcast/results/annotation/<chain>_sample_blinded_aw.csv     (rater 2)

Outputs
-------
papers/lure-broadcast/results/detector_design_based_<chain>.json (+ .manifest.json)

Failure modes
-------------
- A stratum present in the sample but absent from the population table raises: the weights would be
  wrong and every downstream figure with them.
- Rows the two raters disagree on, or either left blank, are EXCLUDED and counted, never imputed.
- A stratum with no agreed rows is reported as unestimable rather than contributing a zero rate.

Complexity: trivial; the bootstrap dominates at n_boot x strata draws.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analysis"))

from src import metadata_lure as ml, stats  # noqa: E402

ANN = ROOT / "papers" / "lure-broadcast" / "results" / "annotation"
OUT_ROOT = ROOT / "papers" / "lure-broadcast" / "results"

NORMALISE = {"legit": "legitimate"}
POSITIVE = {"lure"}
N_BOOT = 5000


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def load_labels(path: Path) -> dict[str, str]:
    out = {}
    for r in csv.DictReader(path.open()):
        lab = r["label"].strip()
        out[r["row_id"]] = NORMALISE.get(lab, lab)
    return out


# --- estimators over [(N_h, [(flagged, positive), ...]), ...] --------------------------------
def _cells(st):
    tp = fp = fn = tn = 0.0
    for Nh, obs in st:
        w = Nh / len(obs)
        for fl, po in obs:
            if fl and po:
                tp += w
            elif fl and not po:
                fp += w
            elif (not fl) and po:
                fn += w
            else:
                tn += w
    return tp, fp, fn, tn


def precision(st):
    tp, fp, _, _ = _cells(st)
    return tp / (tp + fp) if (tp + fp) else None


def recall(st):
    tp, _, fn, _ = _cells(st)
    return tp / (tp + fn) if (tp + fn) else None


def f1(st):
    p, r = precision(st), recall(st)
    # `if (p and r ...)` would map a genuine precision of 0.0 to None, and an undefined replicate is
    # DROPPED by the bootstrap rather than counted -- which would condition the interval on
    # "precision > 0" and bias it upward. Only a truly undefined arm yields None.
    if p is None or r is None or (p + r) == 0:
        return None
    return 2 * p * r / (p + r)


def flagged_population(pop: dict[str, int], rows: dict, md: dict, band_of) -> dict[str, int]:
    """Flagged contracts per FROZEN stratum, as a census rather than an estimate.

    Why this is knowable exactly. Metadata is cached for every contract in the frame and the
    detector is a deterministic function of it, so which contracts are flagged is not sampled at
    all -- only their labels are. The precision denominator is therefore a known constant, which is
    what makes an exact finite-population bound available in the first place.

    Two facts let the per-stratum split be derived from the drawn rows alone, and BOTH are checked
    here rather than assumed:
      * no contract has left the `strong` band since the draw, so every frozen-strong stratum is
        flagged in full and M_h = N_h;
      * every `weak` stratum was drawn in full, so its flagged members can simply be counted.
    A `none` stratum contributes nothing only because no `none -> strong` transition exists; that
    too is asserted, since such a transition would hide flagged contracts in strata this function
    reports as empty.

    Failure modes: raises if any of the three facts fails, or if the per-stratum counts do not sum
    to the census count of flagged contracts in the frame. A silent disagreement here would put a
    wrong denominator under every precision figure in the paper.
    """
    frozen_of = {r["address"].lower(): (r["stratum"], r["strength"]) for r in rows.values()}
    for addr, (st, frozen) in frozen_of.items():
        now = band_of(addr)
        if frozen == "strong" and now != "strong":
            raise SystemExit(f"{addr} was frozen `strong` in {st} but is now `{now}`; frozen-strong "
                             f"strata can no longer be taken as flagged in full")
        if frozen == "none" and now == "strong":
            raise SystemExit(f"{addr} was frozen `none` in {st} but is now `strong`; flagged "
                             f"contracts would then hide in `none` strata this function reports "
                             f"as empty")

    out: dict[str, int] = {}
    for st, n_h in pop.items():
        band = st.split("|", 1)[0]
        drawn = [a for a, (s, _) in frozen_of.items() if s == st]
        if band == "strong":
            out[st] = n_h                                  # flagged in full, checked above
        elif band == "weak":
            if len(drawn) != n_h:
                raise SystemExit(f"{st} has {n_h} contracts but only {len(drawn)} were drawn; its "
                                 f"flagged count cannot be derived by counting")
            out[st] = sum(1 for a in drawn if band_of(a) == "strong")
        else:
            out[st] = 0
    census = sum(1 for a in md if band_of(a) == "strong")
    if sum(out.values()) != census:
        raise SystemExit(f"per-stratum flagged counts sum to {sum(out.values())} but the frame "
                         f"contains {census} flagged contracts; the denominator of every precision "
                         f"figure would be wrong")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chain", default="base")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = OUT_ROOT / f"detector_design_based_{args.chain}.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out.relative_to(ROOT)} exists; pass --force")

    key = json.loads((ANN / f"{args.chain}_answer_key.json").read_text())
    md = json.loads((ROOT / "analysis" / "config" /
                     "token_metadata.json").read_text())[args.chain]
    md = {a.lower(): v for a, v in md.items()}
    pop = key["strata_population"]
    rows = key["rows"]
    r1 = load_labels(ANN / f"{args.chain}_sample_blinded.csv")
    r2 = load_labels(ANN / f"{args.chain}_sample_blinded_aw.csv")

    def band_of(address: str) -> str:
        rec = md.get(address.lower())
        if rec is None:
            raise SystemExit(f"{address} is in the sample but absent from the metadata cache; "
                             f"its current detector verdict cannot be recomputed")
        return ml.classify(rec.get("name"), rec.get("symbol")).strength

    by_stratum: dict[str, list[tuple[int, int]]] = {}
    excluded = {"blank": 0, "disagreed": 0}
    reclassified: list[dict] = []
    # Bookkeeping the excluded rows too. The bound below needs the number of flagged contracts
    # actually DRAWN, not merely the number that survived adjudication, because the drawn set is
    # the one the sampling design produced; the agreed set is a further, outcome-dependent filter.
    drawn_by_stratum: Counter[str] = Counter()
    flagged_drawn: Counter[str] = Counter()
    flagged_agreed: Counter[str] = Counter()
    excluded_flagged: dict[str, list[str]] = {}
    for rid, meta in rows.items():
        st = meta["stratum"]
        if st not in pop:
            raise SystemExit(f"stratum {st!r} is in the sample but not in the population table; "
                             f"its weight is undefined and every figure using it would be wrong")
        drawn_by_stratum[st] += 1
        flagged = 1 if band_of(meta["address"]) == "strong" else 0
        if flagged:
            flagged_drawn[st] += 1
        a, b = r1.get(rid, ""), r2.get(rid, "")
        if not a or not b or a != b:
            excluded["blank" if (not a or not b) else "disagreed"] += 1
            if flagged:
                excluded_flagged.setdefault(st, []).append(rid)
            continue
        frozen = 1 if meta["strength"] == "strong" else 0
        if flagged != frozen:
            reclassified.append({"row": rid, "stratum": st, "frozen": meta["strength"],
                                 "now": "strong" if flagged else "not-strong"})
        if flagged:
            flagged_agreed[st] += 1
        positive = 1 if a in POSITIVE else 0
        by_stratum.setdefault(st, []).append((flagged, positive))

    strata = [(pop[s], obs) for s, obs in sorted(by_stratum.items())]
    unestimable = [s for s in pop if s not in by_stratum]
    flagged_pop = flagged_population(pop, rows, md, band_of)

    tp, fp, fn, tn = _cells(strata)
    raw = {"TP": sum(1 for _, o in strata for f, p in o if f and p),
           "FP": sum(1 for _, o in strata for f, p in o if f and not p),
           "FN": sum(1 for _, o in strata for f, p in o if not f and p),
           "TN": sum(1 for _, o in strata for f, p in o if not f and not p)}

    res = {}
    for name, fn_ in (("precision", precision), ("recall", recall), ("f1", f1)):
        pt, lo, hi = stats.stratified_bootstrap_ci(strata, fn_, n_boot=args.n_boot)
        res[name] = {"point": pt,
                     "ci95": [None if math.isnan(lo) else lo, None if math.isnan(hi) else hi],
                     "method": "finite-population stratified bootstrap"}

    # The finite-population bootstrap resamples OBSERVED rows, so with zero observed false positives
    # no replicate can contain one and the precision interval collapses to a point. That is a
    # property of the resampling scheme, not evidence of certainty. The replacement is an exact
    # DESIGN-BASED bound: the flagged population is a census, so the false-positive count can be
    # bounded by inverting the hypergeometric test stratum by stratum, jointly.
    #
    # A pooled rule of three was reported here previously. It is a binomial zero-event bound -- it
    # assumes independent draws from an infinite population and discards the stratum weights and the
    # finite-population correction, which is precisely what the rest of this script argues cannot be
    # done for this estimand. It is retained below, relabelled, for comparability only.
    flag_strata = [s for s in sorted(pop) if flagged_pop[s] > 0]
    m_pop = [flagged_pop[s] for s in flag_strata]
    m_agreed = [flagged_agreed[s] for s in flag_strata]
    m_drawn = [flagged_drawn[s] for s in flag_strata]
    n_excl = [len(excluded_flagged.get(s, [])) for s in flag_strata]

    k_a, th_a, alloc_a = stats.stratified_exact_upper_bound(list(zip(m_pop, m_agreed, [0] * len(m_pop))))
    k_w, th_w, _ = stats.stratified_exact_upper_bound(list(zip(m_pop, m_drawn, n_excl)))
    k_b, th_b, _ = stats.stratified_exact_upper_bound(list(zip(m_pop, m_drawn, [0] * len(m_pop))))
    h = len(m_pop)
    k_bonf = sum(stats.stratified_exact_upper_bound([(M, m, 0)], conf=1 - (1 - 0.95) / h)[0]
                 for M, m in zip(m_pop, m_agreed))

    res["precision"]["degenerate_no_observed_fp"] = (raw["FP"] == 0)
    res["precision"]["flagged_population_is_a_census"] = sum(m_pop)
    res["precision"]["false_positive_upper95_design_based"] = {
        "theta": th_a, "k_max": k_a, "allocation": dict(zip(flag_strata, alloc_a)),
        "flagged_population": sum(m_pop), "flagged_sampled_agreed": sum(m_agreed),
        "method": ("inversion of the exact hypergeometric test on the total, over the frozen "
                   "strata, conditional on the realised flagged-sample sizes"),
        "assumes": ("that excluding the disagreed and blank rows is ignorable within stratum; the "
                    "worst-case figure below assumes nothing about them"),
    }
    res["precision"]["false_positive_upper95_worst_case_exclusions"] = {
        "theta": th_w, "k_max": k_w, "flagged_sampled_drawn": sum(m_drawn),
        "excluded_flagged_treated_as_false_positive": sum(n_excl),
        "observed_precision": raw["TP"] / (raw["TP"] + sum(n_excl)) if raw["TP"] else None,
        "note": ("makes no assumption about the excluded rows, so it is unconditional with respect "
                 "to the unresolved-label question. Like every figure here it remains conditional "
                 "on the stated sampling design, on the detector frame as the population, and on "
                 "the operational codebook as the reference definition."),
    }
    res["precision"]["false_positive_upper95_if_exclusions_are_positive"] = {
        "theta": th_b, "k_max": k_b,
        "note": "what adjudicating the excluded rows as positives would buy",
    }
    res["precision"]["false_positive_upper95_bonferroni_crosscheck"] = {
        "theta": k_bonf / sum(m_pop), "k_max": k_bonf,
        "note": ("per-stratum bounds at alpha/H, summed. A joint region tighter than this would "
                 "indicate the region is too small, so this is a guard rather than a competitor"),
    }
    res["precision"]["false_positive_upper95_simple_binomial_diagnostic"] = {
        "theta": stats.rule_of_three(raw["TP"] + raw["FP"]),
        "note": ("NOT design-based: the infinite-population limit of the bound above, reported for "
                 "comparability with the rule-of-three literature"),
    }
    res["precision"]["degeneracy_note"] = (
        "the bootstrap cannot produce an outcome absent from the sample, so with no observed false "
        "positive its interval is a point; the design-based bound replaces it")

    per_stratum = []
    for s, obs in sorted(by_stratum.items()):
        n = len(obs)
        per_stratum.append({
            "stratum": s, "N_h": pop[s], "n_h": n, "sampling_fraction": n / pop[s],
            # Was `bool(obs[0][0])`, the flag of the FIRST row only. Flagged-ness is recomputed,
            # so a stratum can be mixed; `weak|200-600` holds 2 flagged of 4 and was published as
            # `false` purely because of which row happened to sort first.
            "flagged_sampled": sum(f for f, _ in obs),
            "flagged_population": flagged_pop.get(s, 0),
            "mixed": 0 < sum(f for f, _ in obs) < len(obs),
            "positives": sum(p for _, p in obs),
            "TP": sum(1 for f, p in obs if f and p), "FP": sum(1 for f, p in obs if f and not p),
            "FN": sum(1 for f, p in obs if not f and p), "TN": sum(1 for f, p in obs if not f and not p),
            "fully_sampled": n == pop[s],
        })

    doc = {
        "chain": args.chain,
        "estimand": {"precision": "P(positive | flagged)", "recall": "P(flagged | positive)"},
        "positive_class": sorted(POSITIVE),
        "agreed_set_only": ("rows are used only where both raters gave the same non-blank label; "
                            "disagreements and blanks are excluded and counted, never imputed"),
        "excluded": excluded,
        "agreed_rows": sum(len(o) for _, o in strata),
        "interval_method": ("finite-population stratified bootstrap: resample within each stratum "
                            "without replacement from a pseudo-population of size N_h. A Wilson "
                            "interval on pooled counts is NOT valid for this design and is no "
                            "longer reported."),
        "n_boot": args.n_boot,
        "confusion_matrix_sampled_rows": raw,
        "confusion_matrix_population_expanded": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        # The expanded flagged row, TP + FP, comes to 146.5 against a flagged population of exactly
        # 141. The excess is not noise: a RECOMPUTED indicator is being expanded with FROZEN stratum
        # weights, and two weak strata were drawn in full (5/5 and 12/12) yet are divided by their
        # smaller AGREED counts, inflating them by 0.5 and 5.0 respectively. Since flagged-ness is a
        # census, the flagged arm needs no expansion at all, and the calibrated matrix below uses it.
        "confusion_matrix_population_calibrated": {
            "TP": sum(m_pop) * (tp / (tp + fp)) if (tp + fp) else 0.0,
            "FP": sum(m_pop) * (fp / (tp + fp)) if (tp + fp) else 0.0,
            "flagged_total": sum(m_pop),
            "note": ("the flagged arm is the census count, not an expansion; the expanded matrix "
                     "above re-inflates the fully enumerated weak strata"),
        },
        "flagged_census": {"total": sum(m_pop), "drawn": sum(m_drawn), "agreed": sum(m_agreed),
                           "per_stratum": {s: flagged_pop[s] for s in flag_strata}},
        "excluded_flagged": {"total": sum(n_excl),
                             "rows": {s: v for s, v in excluded_flagged.items()}},
        "population_total": sum(pop.values()),
        "expanded_counts_note": ("expanded cells are EXPECTED counts under Horvitz--Thompson "
                                 "weighting and are fractional by construction; the sampled-row "
                                 "matrix above gives the integer counts actually observed"),
        "reclassified_since_draw": {
            "n": len(reclassified),
            "note": ("contracts whose CURRENT detector band differs from the band frozen when the "
                     "sample was drawn. Weights come from the frozen strata; flagged-ness comes "
                     "from the current detector, because that is what is under evaluation."),
            "rows": reclassified[:20],
        },
        "estimates": res,
        "per_stratum": per_stratum,
        "strata_without_agreed_rows": unestimable,
        "fully_sampled_strata": [s["stratum"] for s in per_stratum if s["fully_sampled"]],
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    (OUT_ROOT / f"detector_design_based_{args.chain}.manifest.json").write_text(json.dumps({
        "producer": "papers/lure-broadcast/scripts/evaluate_design_based.py",
        "git_commit": git_commit(), "n_boot": args.n_boot,
    }, indent=1) + "\n")

    print(f"  agreed rows {doc['agreed_rows']} over {len(strata)} strata "
          f"(excluded: {excluded}); fully sampled strata: {len(doc['fully_sampled_strata'])}")
    print(f"  sampled-row matrix   {raw}")
    print(f"  expanded matrix      TP={tp:.1f} FP={fp:.1f} FN={fn:.1f} TN={tn:.1f}")
    for k, v in res.items():
        lo, hi = v["ci95"]
        s = f"[{lo:.4f}, {hi:.4f}]" if lo is not None else "[undefined]"
        print(f"  {k:9s} {v['point']:.4f}  95% {s}")
    print(f"  -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
