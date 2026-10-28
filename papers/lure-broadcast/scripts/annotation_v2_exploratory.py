"""An exploratory check on the v2 annotation, NOT declared in the pre-registration or in deviation 4.

Deviation 4's A2 counts how often the author's conjunction, and how often the detector, match the
non-author majority. Both counts came out equal, which raises a question A2 does not answer: does the
author's reading sit closer to the detector than an outsider's does? This script answers it
descriptively and is reported as exploratory, apart from every registered figure:
  E1  for each rater, over the rows where that rater's conjunction is resolved, how often it agrees
      with the detector (flagged <-> "yes");
  E2  how many rows the author's and the detector's mismatches with the non-author majority share.
It was written after robustness.json was committed and changes nothing in it.

Guards (each raises): tier.json committed and unmodified; the frozen detector unchanged; every sheet
read through the registered reader; the sheets given are exactly the raters tier.json names.

Inputs:  results/annotation_v2/{answer_key.json, sample_blinded.csv, sample.manifest.json, tier.json}
         every sheet: --sheet RATER_ID=PATH, once per rater named in tier.json
Output:  results/annotation_v2/exploratory.json (+ .manifest.json)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import annotation_v2_robustness as rb
import evaluate_annotation_v2 as ev

OUT = ev.DIR / "exploratory.json"


def detector_agreement(rows: dict, derived: dict[str, dict]) -> dict:
    """One rater's resolved conjunctions against the detector band: rows resolved, rows agreeing."""
    resolved = [rid for rid, d in derived.items() if d["conj"] is not None]
    agree = sum((rows[rid]["strength"] == "strong") == (derived[rid]["conj"] == "yes") for rid in resolved)
    return {"resolved": len(resolved), "agrees_with_detector": agree,
            "share": agree / len(resolved) if resolved else None}


def mismatch_overlap(rows: dict, author: dict, majority: dict) -> dict:
    """Rows where the author, and where the detector, differ from a resolved majority conjunction."""
    settled = [rid for rid, v in majority.items() if v["conj"] is not None]
    by_author = {rid for rid in settled if author[rid]["conj"] != majority[rid]["conj"]}
    by_detector = {rid for rid in settled
                   if (rows[rid]["strength"] == "strong") != (majority[rid]["conj"] == "yes")}
    return {"rows_with_majority_conjunction": len(settled), "author_mismatches": len(by_author),
            "detector_mismatches": len(by_detector), "shared": len(by_author & by_detector)}


def analyse(key: dict, sheets: dict[str, dict], nonauthors: list[str], authors: list[str]) -> dict:
    rows = key["rows"]
    derived = {r: {rid: ev.derive(rec) for rid, rec in sheets[r].items()} for r in sheets}
    majority = rb.majority_labels(derived, nonauthors)
    return {"status": "exploratory, not declared in PREREG_ANNOTATION_V2.md or its deviations",
            "nonauthor_raters": nonauthors, "author_raters": authors,
            "E1_detector_agreement": {r: detector_agreement(rows, derived[r]) for r in nonauthors + authors},
            "E2_mismatch_overlap": {au: mismatch_overlap(rows, derived[au], majority) for au in authors}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sheet", action="append", required=True, metavar="RATER_ID=PATH")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if OUT.exists() and not args.force:
        raise SystemExit(f"{OUT.relative_to(ev.ROOT)} exists; pass --force")
    pairs = dict(s.split("=", 1) for s in args.sheet)

    ev.check_tier_committed()
    nonauthors, authors = rb.roles(json.loads(ev.TIER.read_text()))
    if sorted(pairs) != sorted(nonauthors + authors):
        raise SystemExit(f"tier.json names {sorted(nonauthors + authors)}, sheets were given for {sorted(pairs)}")
    if not authors:
        raise SystemExit("tier.json names no author rater; there is nothing to compare")
    ev.check_frozen_detector(json.loads(ev.DRAW_MANIFEST.read_text()))
    key = json.loads(ev.KEY.read_text())
    with ev.SHEET.open(encoding="utf-8-sig", newline="") as fh:
        sample = {r["row_id"]: r for r in csv.DictReader(fh)}
    sheets = {r: ev.read_sheet(Path(p), sample)[0] for r, p in pairs.items()}

    result = analyse(key, sheets, nonauthors, authors)
    result["sheets"] = {r: {"path": p, "sha256": ev.sha256(Path(p))} for r, p in pairs.items()}
    OUT.write_text(json.dumps(result, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    OUT.with_suffix(".manifest.json").write_text(json.dumps({
        "producer": "papers/lure-broadcast/scripts/annotation_v2_exploratory.py",
        "git_commit": ev.git_commit(),
        "registration": "none: exploratory, reported apart from the registered analyses",
    }, indent=1) + "\n")
    print(f"  -> {OUT.relative_to(ev.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
