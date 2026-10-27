"""Inter-rater reliability for the lure annotation, and the adjudication queue.

Why this exists
---------------
Kill condition K3's third leg requires Cohen's kappa from two INDEPENDENT raters. Until now the annotation
carried one label column, so no kappa existed and the detector's precision and recall carried no
inter-rater validity bound. A second rater's file closes that.

Two tasks are reported, because they answer different questions:
  5-class -- how reliably the codebook's full vocabulary is applied. Diagnoses the codebook.
  binary  -- `lure` versus everything else. This is the task the DETECTOR performs, so it is the kappa the
             detector's precision and recall should be qualified by.

Label normalisation is explicit, never silent
---------------------------------------------
The second rater wrote `legit` where the codebook says `legitimate`. That is an unambiguous abbreviation --
no other vocabulary term it could denote -- so it is mapped, but the mapping is declared in NORMALISE
below, applied to both raters identically, and recorded in the output. Silently coercing an
off-vocabulary label would hide a codebook-compliance problem; refusing to map it would discard 259 of 372
labels over a spelling. Any label NOT in the vocabulary or this table still raises.

Blanks
------
Kappa needs both raters to have judged an item, so rows blank for either rater are excluded from kappa and
their count is reported. They are NOT imputed: an unjudged item carries no information about agreement.

Inputs
------
papers/lure-broadcast/results/annotation/base_sample_blinded.csv      (rater 1)
papers/lure-broadcast/results/annotation/base_sample_blinded_aw.csv   (rater 2)
papers/lure-broadcast/results/annotation/base_answer_key.json         (strata; NOT shown to raters)

Outputs
-------
papers/lure-broadcast/results/interrater_<chain>.json (+ .manifest.json)
papers/lure-broadcast/results/annotation/<chain>_adjudication_queue.csv

Failure modes
-------------
- Differing row_id sets raise: the two raters must have judged the same draw or agreement is meaningless.
- Differing name/symbol for any row raises: that would mean the files are not the same sample.
- An off-vocabulary label outside NORMALISE raises.
- kappa below K3_THRESHOLD is reported as firing the kill condition, not smoothed over.

Complexity: trivial.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analysis"))

from src import stats  # noqa: E402

ANN = ROOT / "papers" / "lure-broadcast" / "results" / "annotation"
OUT_ROOT = ROOT / "papers" / "lure-broadcast" / "results"

VOCAB = {"lure", "lure-nodest", "dest-only", "legitimate", "unclear"}
#: Declared, auditable normalisations. Applied to BOTH raters identically.
NORMALISE = {"legit": "legitimate"}
POSITIVE = {"lure"}
#: Below this, K3's third leg fires and prevalence claims must be dropped.
K3_THRESHOLD = 0.40


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def load(path: Path) -> dict[str, dict]:
    rows = {}
    for r in csv.DictReader(path.open()):
        lab = r["label"].strip()
        lab = NORMALISE.get(lab, lab)
        if lab and lab not in VOCAB:
            raise SystemExit(f"{path.name}: off-vocabulary label {lab!r} on {r['row_id']}; add it to "
                             f"NORMALISE with a justification or fix the annotation")
        rows[r["row_id"]] = {"label": lab, "name": r["name"], "symbol": r["symbol"],
                            "notes": r.get("notes", ""), "rank": r.get("volume_rank")}
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chain", default="base")
    ap.add_argument("--rater1", default="base_sample_blinded.csv")
    ap.add_argument("--rater2", default="base_sample_blinded_aw.csv")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = OUT_ROOT / f"interrater_{args.chain}.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out.relative_to(ROOT)} exists; pass --force")

    A = load(ANN / args.rater1)
    B = load(ANN / args.rater2)
    if set(A) != set(B):
        raise SystemExit(f"row_id sets differ (A-only {len(set(A)-set(B))}, B-only {len(set(B)-set(A))}); "
                         f"agreement is meaningless across different draws")
    mism = [k for k in A if (A[k]["name"], A[k]["symbol"]) != (B[k]["name"], B[k]["symbol"])]
    if mism:
        raise SystemExit(f"{len(mism)} rows differ in name/symbol -- these are not the same sample")

    ids = sorted(A)
    both = [k for k in ids if A[k]["label"] and B[k]["label"]]
    blank_a = sum(1 for k in ids if not A[k]["label"])
    blank_b = sum(1 for k in ids if not B[k]["label"])

    a5 = [A[k]["label"] for k in both]
    b5 = [B[k]["label"] for k in both]
    a2 = ["lure" if x in POSITIVE else "other" for x in a5]
    b2 = ["lure" if x in POSITIVE else "other" for x in b5]

    k5, lo5, hi5, po5 = stats.cohen_kappa(a5, b5)
    k2, lo2, hi2, po2 = stats.cohen_kappa(a2, b2)

    conf = Counter(zip(a5, b5))
    labels = sorted({*a5, *b5})
    disagree = [k for k in both if A[k]["label"] != B[k]["label"]]

    # Adjudication queue: every disagreement, plus rows either rater called `unclear`.
    queue = sorted(set(disagree) | {k for k in both
                                    if "unclear" in (A[k]["label"], B[k]["label"])})
    qpath = ANN / f"{args.chain}_adjudication_queue.csv"
    with qpath.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row_id", "volume_rank", "name", "symbol",
                    "rater1", "rater2", "adjudicated", "notes"])
        for k in queue:
            w.writerow([k, A[k]["rank"], A[k]["name"], A[k]["symbol"],
                        A[k]["label"], B[k]["label"], "", ""])

    def interpret(k: float) -> str:
        if k < 0.40: return "poor -- K3's third leg FIRES"
        if k < 0.60: return "moderate"
        if k < 0.80: return "substantial"
        return "strong"

    doc = {
        "chain": args.chain,
        "raters": 2,
        "rater_files": [args.rater1, args.rater2],
        "normalisations_applied": NORMALISE,
        "rows": len(ids), "judged_by_both": len(both),
        "blank_rater1": blank_a, "blank_rater2": blank_b,
        "label_distribution": {"rater1": dict(Counter(a5)), "rater2": dict(Counter(b5))},
        "five_class": {"kappa": k5, "ci95": [lo5, hi5], "observed_agreement": po5,
                       "interpretation": interpret(k5)},
        "binary_lure_vs_rest": {"kappa": k2, "ci95": [lo2, hi2], "observed_agreement": po2,
                                "interpretation": interpret(k2),
                                "note": "this is the task the detector performs, so it is the kappa its "
                                        "precision and recall must be qualified by"},
        "k3_threshold": K3_THRESHOLD,
        "k3_third_leg_fires": k2 < K3_THRESHOLD,
        "confusion_5class": {f"{x}|{y}": conf[(x, y)] for x in labels for y in labels if conf[(x, y)]},
        "disagreements": len(disagree),
        "adjudication_queue": {"path": str(qpath.relative_to(ROOT)), "rows": len(queue)},
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    (OUT_ROOT / f"interrater_{args.chain}.manifest.json").write_text(json.dumps({
        "producer": "papers/lure-broadcast/scripts/interrater.py",
        "git_commit": git_commit(),
        "codebook": "docs/ANNOTATION_CODEBOOK.md v1.0",
    }, indent=1) + "\n")

    print(f"  rows {len(ids)}; judged by both {len(both)} "
          f"(blanks: rater1 {blank_a}, rater2 {blank_b})")
    print(f"  normalisations applied: {NORMALISE}")
    print(f"  5-class  kappa {k5:.4f}  CI[{lo5:.4f},{hi5:.4f}]  agreement {po5:.4f}  -> {interpret(k5)}")
    print(f"  binary   kappa {k2:.4f}  CI[{lo2:.4f},{hi2:.4f}]  agreement {po2:.4f}  -> {interpret(k2)}")
    print(f"  K3 third leg fires: {doc['k3_third_leg_fires']}")
    print(f"  disagreements {len(disagree)}; adjudication queue {len(queue)} rows -> "
          f"{qpath.relative_to(ROOT)}")
    print("  5-class confusion (rater1 | rater2 : n):")
    for kk, v in sorted(doc["confusion_5class"].items(), key=lambda kv: -kv[1]):
        print(f"    {kk:32s} {v:4d}")
    print(f"  -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
