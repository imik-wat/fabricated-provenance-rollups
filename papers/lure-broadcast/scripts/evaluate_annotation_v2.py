"""Evaluate the frozen detector against the v2 ratings, exactly as registered (PREREG_ANNOTATION_V2.md §5-7).

Written and tested BEFORE any rating sheet was returned, so that the analysis cannot be shaped by the
labels. Every quantity here is named in the pre-registration. Anything else is out of scope.

Guards (each raises rather than producing a number):
  * TIER. results/annotation_v2/tier.json must exist, be tracked by git and be unmodified. It names
    the tier and each rater's author status, which the registration requires to be fixed before any
    returned sheet is opened. The raters it names must match the sheets given.
  * FROZEN DETECTOR. The SHA-256 of every detector input must equal the digest in the draw manifest.
    Every recomputed verdict must equal the strength frozen in the answer key.
  * SHEETS. The header must be the blinded sheet's, and the row_ids must be exactly the sample's
    (none missing, duplicated or extra). Any answer value outside the protocol vocabulary raises,
    because the registration declares no normalisation. Names or symbols that differ from the sheet,
    which a spreadsheet program can introduce, are counted and listed, not fatal, since rows are
    matched by row_id.

Primary analyses (§6), per chain and pooled with design weights:
  P1  detector strong  vs  rater conjunction (destination = yes AND call_to_action = yes)
  P2  detector strong  vs  rater purpose = solicit
  Each reports: the sampled confusion matrix on the agreed set; precision, recall and F1 with
  finite-population stratified bootstrap intervals; and the exact 95% upper bound on disagreeing
  flagged contracts, both under ignorability and in the worst case.
  Cohen's kappa for destination, call_to_action, purpose and the derived conjunction.

Secondary analyses (§7), descriptive: feature-level agreement, the language gap with translated
counts, the interaction cases (shortcut paths, allowlisted domains) and, for tier A, adjudication,
reported separately.

Inputs:  results/annotation_v2/{answer_key.json, sample_blinded.csv, sample.manifest.json, tier.json}
         two rater sheets (--sheet RATER_ID=PATH, twice); optional --adjudication PATH (tier A)
Output:  results/annotation_v2/evaluation.json (+ .manifest.json). Refuses to overwrite without
         --force.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "papers" / "lure-broadcast"
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import metadata_lure as ml, stats  # noqa: E402
import build_annotation_sample_v2 as draw  # noqa: E402
import rule_paths as rp  # noqa: E402
from evaluate_design_based import precision, recall, f1, _cells  # noqa: E402

DIR = PAPER / "results" / "annotation_v2"
KEY = DIR / "answer_key.json"
SHEET = DIR / "sample_blinded.csv"
DRAW_MANIFEST = DIR / "sample.manifest.json"
TIER = DIR / "tier.json"
OUT = DIR / "evaluation.json"

COLUMNS = draw.BLINDED_COLUMNS
VOCAB = {"destination": {"yes", "no", "unclear"}, "call_to_action": {"yes", "no", "unclear"},
         "purpose": {"solicit", "name", "unclear"}}
LANG_CODES = {"en", "zh", "ru", "ko", "ja", "es", "tr", "vi"}
LABELS = {"P1_conjunction": "conj", "P2_purpose": "purpose"}
POSITIVE = {"conj": "yes", "purpose": "solicit"}


# ------------------------------------------------------------------------------------------ guards
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def check_tier_committed(path: Path = TIER) -> None:
    """tier.json must be tracked and clean, so it provably predates the analysis run."""
    rel = str(path.relative_to(ROOT))
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", rel],
                             capture_output=True, text=True)
    if tracked.returncode != 0:
        raise ValueError(f"{rel} is not committed. Commit the tier before opening any sheet.")
    dirty = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", rel])
    if dirty.returncode != 0:
        raise ValueError(f"{rel} has uncommitted changes; the tier must not change after registration")


def validate_tier(tier: dict, rater_ids: list[str]) -> None:
    """The tier's rule on author status, and agreement between named raters and supplied sheets."""
    if tier.get("tier") not in {"A", "B", "C"}:
        raise ValueError(f"tier must be A, B or C, got {tier.get('tier')!r}")
    raters = {r["id"]: bool(r["author"]) for r in tier.get("raters", [])}
    if sorted(raters) != sorted(rater_ids):
        raise ValueError(f"tier.json names raters {sorted(raters)} but sheets were given for {sorted(rater_ids)}")
    authors = sum(raters.values())
    rule = {"A": authors == 0, "B": authors == 1, "C": authors == len(raters)}[tier["tier"]]
    if len(raters) != 2 or not rule:
        raise ValueError(f"tier {tier['tier']} does not match the raters' author status {raters}")
    adj = tier.get("adjudicator")
    if adj is not None and (tier["tier"] != "A" or adj.get("author")):
        raise ValueError("an adjudicator is registered only for tier A, and must not be an author")


def check_frozen_detector(manifest: dict) -> None:
    for rel, digest in manifest["frozen_inputs_sha256"].items():
        if rel.endswith("ANNOTATION_PROTOCOL_V2.md"):
            continue            # the protocol moved to v2.2 by logged deviations; the detector did not
        if sha256(ROOT / rel) != digest:
            raise ValueError(f"{rel} differs from the frozen digest; v2 results are defined only "
                             f"against the frozen detector (PREREG §2)")


# ------------------------------------------------------------------------------------------ sheets
def read_sheet(path: Path, sample: dict[str, dict]) -> tuple[dict[str, dict], list[str]]:
    """One rater's answers keyed by row_id, plus the row_ids whose strings differ from the sample."""
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"{path.name}: columns {reader.fieldnames} are not the sheet's {COLUMNS}")
        recs = list(reader)
    ids = [r["row_id"] for r in recs]
    dupes = sorted(k for k, n in Counter(ids).items() if n > 1)
    if dupes:
        raise ValueError(f"{path.name}: duplicated rows {dupes[:5]}")
    if set(ids) != set(sample):
        missing, extra = sorted(set(sample) - set(ids)), sorted(set(ids) - set(sample))
        raise ValueError(f"{path.name}: rows do not match the sample (missing {missing[:5]}, extra {extra[:5]})")
    bad: list[str] = []
    for r in recs:
        for field, allowed in VOCAB.items():
            if r[field] not in allowed and r[field] != "":
                bad.append(f"{r['row_id']}:{field}={r[field]!r}")
        lang = r["cta_language"]
        if lang and lang not in LANG_CODES and not (lang.startswith("other:") and lang[6:].strip()):
            bad.append(f"{r['row_id']}:cta_language={lang!r}")
    if bad:
        raise ValueError(f"{path.name}: values outside the protocol vocabulary: {bad[:10]}"
                         + (f" and {len(bad) - 10} more" if len(bad) > 10 else ""))
    altered = [r["row_id"] for r in recs
               if (r["name"], r["symbol"]) != (sample[r["row_id"]]["name"], sample[r["row_id"]]["symbol"])]
    return {r["row_id"]: r for r in recs}, sorted(altered)


def derive(rec: dict) -> dict:
    """The registered derived labels (§5): None marks an unresolved label."""
    d, c = rec.get("destination", ""), rec.get("call_to_action", "")
    if d == "yes" and c == "yes":
        conj = "yes"
    elif d == "no" or c == "no":
        conj = "no"
    else:
        conj = None
    p = rec.get("purpose", "")
    return {"conj": conj, "purpose": p if p in ("solicit", "name") else None}


# ----------------------------------------------------------------------------------------- analysis
def _estimates(strata: list[tuple[int, list]], n_boot: int) -> dict:
    usable = [(n, obs) for n, obs in strata if obs]
    out = {"strata_used": len(usable), "strata_without_agreed_rows": len(strata) - len(usable)}
    for name, fn in (("precision", precision), ("recall", recall), ("f1", f1)):
        if not usable:
            out[name] = {"point": None, "ci95": [None, None]}
            continue
        pt, lo, hi = stats.stratified_bootstrap_ci(usable, fn, n_boot=n_boot)
        out[name] = {"point": pt, "ci95": [lo, hi]}
    return out


def _bounds(design: dict[str, dict]) -> dict:
    """Exact upper bounds on disagreeing flagged contracts, under ignorability and in the worst case."""
    flagged = {s: d for s, d in design.items() if d["flagged"]}
    if not flagged:
        return {"flagged_population": 0, "ignorability": None, "worst_case": None}
    total = sum(d["N"] for d in flagged.values())
    ign = stats.stratified_exact_upper_bound([(d["N"], d["agreed_n"], d["fp"]) for d in flagged.values()])
    worst = stats.stratified_exact_upper_bound([(d["N"], d["drawn"], d["fp"] + d["excluded"])
                                                for d in flagged.values()])
    return {"flagged_population": total,
            "ignorability": {"k_max": ign[0], "theta": ign[1]},
            "worst_case": {"k_max": worst[0], "theta": worst[1]}}


def primary(label: str, rows: dict, derived: dict[str, dict[str, dict]], raters: list[str],
            populations: dict[str, dict[str, int]], chains: list[str] | None, n_boot: int) -> dict:
    """One registered comparison over the chains given (one chain, or all of them pooled)."""
    design: dict[str, dict] = {}
    excluded = Counter()
    for rid, meta in rows.items():
        if chains is not None and meta["chain"] not in chains:
            continue
        key = f"{meta['chain']}|{meta['stratum']}"
        d = design.setdefault(key, {"N": populations[meta["chain"]][meta["stratum"]], "obs": [],
                                    "flagged": meta["strength"] == "strong", "agreed_n": 0,
                                    "drawn": 0, "fp": 0, "excluded": 0})
        d["drawn"] += 1
        a, b = (derived[r][rid][label] for r in raters)
        if a is None or b is None or a != b:
            excluded["unresolved" if (a is None or b is None) else "disagreed"] += 1
            if d["flagged"]:
                d["excluded"] += 1
            continue
        positive = a == POSITIVE[label]
        d["obs"].append((d["flagged"], positive))
        d["agreed_n"] += 1
        if d["flagged"] and not positive:
            d["fp"] += 1
    raw = Counter()
    for d in design.values():
        for fl, po in d["obs"]:
            raw[("TP" if po else "FP") if fl else ("FN" if po else "TN")] += 1
    strata = [(d["N"], d["obs"]) for _, d in sorted(design.items())]
    return {
        "agreed_rows": sum(d["agreed_n"] for d in design.values()),
        "excluded": dict(excluded),
        "excluded_flagged": sum(d["excluded"] for d in design.values()),
        "sampled_confusion": {k: raw.get(k, 0) for k in ("TP", "FP", "FN", "TN")},
        "estimates": _estimates(strata, n_boot),
        "upper_bound_disagreeing_flagged": _bounds(design),
    }


def kappas(sheets: dict[str, dict], raters: list[str], derived: dict) -> dict:
    """Cohen's kappa over rows both raters answered (blank answers excluded and counted)."""
    a_id, b_id = raters
    out = {}
    for field in ("destination", "call_to_action", "purpose", "conj"):
        if field == "conj":
            pairs = [(derived[a_id][r]["conj"] or "unresolved", derived[b_id][r]["conj"] or "unresolved")
                     for r in sheets[a_id]]
        else:
            pairs = [(sheets[a_id][r][field], sheets[b_id][r][field]) for r in sheets[a_id]
                     if sheets[a_id][r][field] and sheets[b_id][r][field]]
        if not pairs:
            out[field] = {"n": 0}
            continue
        k, lo, hi, po = stats.cohen_kappa([p[0] for p in pairs], [p[1] for p in pairs])
        out[field] = {"n": len(pairs), "kappa": k, "ci95": [lo, hi], "observed_agreement": po}
    return out


def secondary(rows: dict, sheets: dict, derived: dict, raters: list[str], verdicts: dict) -> dict:
    a_id, b_id = raters
    feature = {}
    for field, det_key in (("destination", "destination"), ("call_to_action", "call_to_action")):
        table = Counter()
        for rid, meta in rows.items():
            va, vb = sheets[a_id][rid][field], sheets[b_id][rid][field]
            if va == vb and va in ("yes", "no"):
                table[f"detector_{'yes' if meta['detector'][det_key] else 'no'}|raters_{va}"] += 1
        feature[field] = dict(sorted(table.items()))

    gap_rows = [rid for rid in rows
                if sheets[a_id][rid]["call_to_action"] == sheets[b_id][rid]["call_to_action"] == "yes"
                and all(sheets[r][rid]["cta_language"] not in ("", "en") for r in raters)]
    language_gap = {
        "rows_non_english_cta_both_raters": len(gap_rows),
        "detector_found_no_cta": sum(1 for rid in gap_rows if not rows[rid]["detector"]["call_to_action"]),
        "translated_notes": {r: sum(1 for rid in gap_rows if sheets[r][rid]["notes"].startswith("translated"))
                             for r in raters},
    }

    def agreed(rid, label):
        a, b = derived[a_id][rid][label], derived[b_id][rid][label]
        return a if (a is not None and a == b) else "not_agreed"

    shortcuts: dict[str, Counter] = {}
    for rid, v in verdicts.items():
        kind = rp.shortcut_kind(v)
        if kind is None:
            continue
        c = shortcuts.setdefault(kind, Counter())
        c["rows"] += 1
        c[f"conj_{agreed(rid, 'conj')}"] += 1
        c[f"purpose_{agreed(rid, 'purpose')}"] += 1
    allow = Counter()
    for rid, v in verdicts.items():
        if v.benign_domains:
            allow["rows"] += 1
            allow[f"destination_{sheets[a_id][rid]['destination']}|{sheets[b_id][rid]['destination']}"] += 1
            allow[f"purpose_{agreed(rid, 'purpose')}"] += 1
    return {"feature_agreement": feature, "language_gap": language_gap,
            "interaction_shortcuts": {k: dict(v) for k, v in sorted(shortcuts.items())},
            "allowlisted_domain_rows": dict(allow)}


def evaluate(key: dict, sample: dict[str, dict], sheets: dict[str, dict], raters: list[str],
             n_boot: int = 5000, adjudication: dict[str, dict] | None = None) -> dict:
    rows = key["rows"]
    populations = {c: info["population_by_stratum"] for c, info in key["chains"].items()}
    derived = {r: {rid: derive(rec) for rid, rec in sheets[r].items()} for r in raters}
    verdicts = {rid: ml.classify(sample[rid]["name"], sample[rid]["symbol"]) for rid in rows}
    drift = [rid for rid, v in verdicts.items() if v.strength != rows[rid]["strength"]]
    if drift:
        raise ValueError(f"recomputed verdicts differ from the frozen key on {drift[:5]}")
    chains = sorted(populations)
    result = {"kappa": kappas(sheets, raters, derived), "primary": {}, "secondary": {}}
    for name, label in LABELS.items():
        result["primary"][name] = {
            "pooled": primary(label, rows, derived, raters, populations, None, n_boot),
            "per_chain": {c: primary(label, rows, derived, raters, populations, [c], n_boot) for c in chains},
        }
    result["secondary"] = secondary(rows, sheets, derived, raters, verdicts)
    if adjudication is not None:
        adj = {"adj": {rid: derive(rec) for rid, rec in adjudication.items()}}
        merged = {r: {} for r in raters}
        for r in raters:
            for rid in rows:
                a, b = derived[raters[0]][rid], derived[raters[1]][rid]
                merged[r][rid] = {lab: (a[lab] if a[lab] is not None and a[lab] == b[lab]
                                        else adj["adj"].get(rid, {}).get(lab)) for lab in POSITIVE}
        result["adjudicated_reported_separately"] = {
            name: primary(label, rows, merged, raters, populations, None, n_boot)
            for name, label in LABELS.items()}
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sheet", action="append", required=True, metavar="RATER_ID=PATH")
    ap.add_argument("--adjudication", type=Path)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if OUT.exists() and not args.force:
        raise SystemExit(f"{OUT.relative_to(ROOT)} exists; pass --force")
    pairs = [s.split("=", 1) for s in args.sheet]
    if len(pairs) != 2 or any(len(p) != 2 for p in pairs):
        raise SystemExit("give exactly two --sheet RATER_ID=PATH arguments")
    raters = [p[0] for p in pairs]

    check_tier_committed()
    tier = json.loads(TIER.read_text())
    validate_tier(tier, raters)
    check_frozen_detector(json.loads(DRAW_MANIFEST.read_text()))
    key = json.loads(KEY.read_text())
    with SHEET.open(encoding="utf-8-sig", newline="") as fh:
        sample = {r["row_id"]: r for r in csv.DictReader(fh)}
    sheets, altered = {}, {}
    for rid_, path in pairs:
        sheets[rid_], altered[rid_] = read_sheet(Path(path), sample)
    adjudication = None
    if args.adjudication is not None:
        if tier["tier"] != "A":
            raise SystemExit("adjudication is registered for tier A only")
        adjudication, _ = read_sheet(args.adjudication, sample)

    result = evaluate(key, sample, sheets, raters, args.n_boot, adjudication)
    result["tier"] = tier
    result["sheets"] = {r: {"path": str(Path(p)), "sha256": sha256(Path(p)), "strings_altered": altered[r]}
                        for r, p in pairs}
    OUT.write_text(json.dumps(result, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    OUT.with_suffix(".manifest.json").write_text(json.dumps({
        "producer": "papers/lure-broadcast/scripts/evaluate_annotation_v2.py",
        "git_commit": git_commit(), "n_boot": args.n_boot,
        "registration": "papers/lure-broadcast/docs/PREREG_ANNOTATION_V2.md",
    }, indent=1) + "\n")
    p1 = result["primary"]["P1_conjunction"]["pooled"]
    print(f"  P1 pooled: agreed {p1['agreed_rows']}, confusion {p1['sampled_confusion']}, "
          f"worst-case bound {p1['upper_bound_disagreeing_flagged']['worst_case']}")
    print(f"  -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
