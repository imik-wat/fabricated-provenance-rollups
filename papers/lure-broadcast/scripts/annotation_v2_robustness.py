"""Robustness and author-comparison analyses declared in deviation 4 (PREREG_ANNOTATION_V2.md §9).

Four sheets were returned against a design registered for two raters plus an adjudicator. The
registered analyses (evaluate_annotation_v2.py) are unchanged and run on the registered pair. This
script computes only what deviation 4 declared before any of the extra sheets was opened, reusing
the registered functions so that nothing is computed two ways:
  R1  P1, P2 and Cohen's kappa for every pair of non-author raters;
  R2  Fleiss' kappa over the non-author raters, per field and for the derived conjunction;
  R3  P1 and P2 on majority labels (at least two non-author raters give the same resolved value);
  A1  Cohen's kappa between the author rater and each non-author, and the non-author majority;
  A2  over rows with a non-author majority conjunction, how often the author's conjunction matches
      it, and how often the detector does.
All are descriptive. No figure from A1 or A2 enters a tier A result.

Guards (each raises): tier.json committed and unmodified; the frozen detector unchanged; every sheet
read through the registered reader (vocabulary, row set, header); the sheets given are exactly the
raters tier.json names, with author status taken from it; an author never enters R1-R3.

Inputs:  results/annotation_v2/{answer_key.json, sample_blinded.csv, sample.manifest.json, tier.json}
         every sheet: --sheet RATER_ID=PATH, once per rater named in tier.json
Output:  results/annotation_v2/robustness.json (+ .manifest.json)
"""
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import evaluate_annotation_v2 as ev
from src import stats

OUT = ev.DIR / "robustness.json"
FIELDS = ("destination", "call_to_action", "purpose")


def roles(tier: dict) -> tuple[list[str], list[str]]:
    """(non-author rater IDs, author rater IDs) from tier.json, in the order it lists them."""
    people = list(tier["raters"])
    if tier.get("adjudicator"):
        people.append(tier["adjudicator"])
    people += tier.get("additional_raters", [])
    ids = [p["id"] for p in people]
    if len(set(ids)) != len(ids):
        raise ValueError(f"tier.json names a rater twice: {ids}")
    return [p["id"] for p in people if not p["author"]], [p["id"] for p in people if p["author"]]


def majority_labels(derived: dict[str, dict[str, dict]], raters: list[str]) -> dict[str, dict]:
    """Per row and derived label, the resolved value at least two raters give, else None."""
    out = {}
    for rid in derived[raters[0]]:
        out[rid] = {}
        for lab in ev.POSITIVE:
            vals = [derived[r][rid][lab] for r in raters if derived[r][rid][lab] is not None]
            top = max(set(vals), key=vals.count) if vals else None
            out[rid][lab] = top if top is not None and vals.count(top) >= 2 else None
    return out


def majority_fields(sheets: dict[str, dict], raters: list[str]) -> dict[str, dict]:
    """Per row and raw field, the answer at least two raters give (unclear included), else blank."""
    out = {}
    for rid in sheets[raters[0]]:
        out[rid] = {}
        for f in FIELDS:
            vals = [sheets[r][rid][f] for r in raters if sheets[r][rid][f]]
            top = max(set(vals), key=vals.count) if vals else ""
            out[rid][f] = top if top and vals.count(top) >= 2 else ""
    return out


def fleiss(sheets: dict, derived: dict, raters: list[str]) -> dict:
    out = {}
    for f in FIELDS + ("conj",):
        if f == "conj":
            ratings = [[derived[r][rid]["conj"] or "unresolved" for r in raters] for rid in sheets[raters[0]]]
        else:
            ratings = [[sheets[r][rid][f] for r in raters] for rid in sheets[raters[0]]
                       if all(sheets[r][rid][f] for r in raters)]
        if not ratings:
            out[f] = {"n": 0}
            continue
        k, p_bar, p_e = stats.fleiss_kappa(ratings)
        out[f] = {"n": len(ratings), "kappa": k, "observed_agreement": p_bar, "chance_agreement": p_e}
    return out


def analyse(key: dict, sheets: dict[str, dict], nonauthors: list[str], authors: list[str],
            n_boot: int) -> dict:
    rows = key["rows"]
    populations = {c: info["population_by_stratum"] for c, info in key["chains"].items()}
    derived = {r: {rid: ev.derive(rec) for rid, rec in sheets[r].items()} for r in sheets}

    def p1p2(raters: list[str], d: dict) -> dict:
        return {name: ev.primary(label, rows, d, raters, populations, None, n_boot)
                for name, label in ev.LABELS.items()}

    out = {"nonauthor_raters": nonauthors, "author_raters": authors,
           "R1_pairs": {f"{a}-{b}": {"kappa": ev.kappas(sheets, [a, b], derived), **p1p2([a, b], derived)}
                        for a, b in combinations(nonauthors, 2)},
           "R2_fleiss": fleiss(sheets, derived, nonauthors)}
    maj = majority_labels(derived, nonauthors)
    out["R3_majority"] = p1p2(["MAJ", "MAJ"], {"MAJ": maj})
    out["R3_majority"]["rows_with_majority"] = {lab: sum(1 for v in maj.values() if v[lab] is not None)
                                               for lab in ev.POSITIVE}
    sheets_m = {**sheets, "MAJ": majority_fields(sheets, nonauthors)}
    derived_m = {**derived, "MAJ": maj}
    for au in authors:
        a1 = {x: ev.kappas(sheets, [au, x], derived) for x in nonauthors}
        a1["majority"] = ev.kappas(sheets_m, [au, "MAJ"], derived_m)
        with_maj = [rid for rid in rows if maj[rid]["conj"] is not None]
        out.setdefault("A1_author_kappa", {})[au] = a1
        out.setdefault("A2_author_vs_majority", {})[au] = {
            "rows_with_majority_conjunction": len(with_maj),
            "author_matches_majority": sum(derived[au][rid]["conj"] == maj[rid]["conj"] for rid in with_maj),
            "detector_matches_majority": sum((rows[rid]["strength"] == "strong") == (maj[rid]["conj"] == "yes")
                                             for rid in with_maj),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sheet", action="append", required=True, metavar="RATER_ID=PATH")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if OUT.exists() and not args.force:
        raise SystemExit(f"{OUT.relative_to(ev.ROOT)} exists; pass --force")
    pairs = dict(s.split("=", 1) for s in args.sheet)

    ev.check_tier_committed()
    tier = json.loads(ev.TIER.read_text())
    nonauthors, authors = roles(tier)
    if sorted(pairs) != sorted(nonauthors + authors):
        raise SystemExit(f"tier.json names {sorted(nonauthors + authors)}, sheets were given for {sorted(pairs)}")
    if len(nonauthors) < 3:
        raise SystemExit("deviation 4's robustness analyses need at least three non-author raters")
    ev.check_frozen_detector(json.loads(ev.DRAW_MANIFEST.read_text()))
    key = json.loads(ev.KEY.read_text())
    with ev.SHEET.open(encoding="utf-8-sig", newline="") as fh:
        sample = {r["row_id"]: r for r in csv.DictReader(fh)}
    sheets = {r: ev.read_sheet(Path(p), sample)[0] for r, p in pairs.items()}

    result = analyse(key, sheets, nonauthors, authors, args.n_boot)
    result["sheets"] = {r: {"path": p, "sha256": ev.sha256(Path(p))} for r, p in pairs.items()}
    OUT.write_text(json.dumps(result, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    OUT.with_suffix(".manifest.json").write_text(json.dumps({
        "producer": "papers/lure-broadcast/scripts/annotation_v2_robustness.py",
        "git_commit": ev.git_commit(), "n_boot": args.n_boot,
        "registration": "papers/lure-broadcast/docs/PREREG_ANNOTATION_V2.md (deviation 4)",
    }, indent=1) + "\n")
    print(f"  -> {OUT.relative_to(ev.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
