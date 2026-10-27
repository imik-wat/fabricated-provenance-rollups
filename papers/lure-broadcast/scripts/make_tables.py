"""Emit tables whose bodies are data rather than a fixed layout.

Why only some tables
--------------------
Most tables in this paper have a fixed set of rows, one per chain, and their cells are already
generated macros; writing those rows out by hand in the `.tex` is clear and the numbers are still
traceable. This script exists for the tables whose ROW SET comes from a results file, where typing
the body means typing data.

The stratified design table is the case in point. Its ten rows are the sampling strata, which are a
property of how the sample was drawn, and every cell is a count from
`detector_design_based_<chain>.json`. It was previously typed by hand. The values happened to be
correct, but nothing kept them correct: the strata are recomputed whenever the sample or the
detector changes, and a hand-typed body would go stale silently, which is exactly the failure that
produced the reviewer's stale-artefact finding elsewhere in this revision.

Outputs
-------
papers/lure-broadcast/paper/generated/tab_strata.tex
papers/lure-broadcast/paper/generated/tab_unresolved.tex   (ledger-probe exceptions)
papers/lure-broadcast/paper/generated/tab_residue.tex      (residue-class replicates)
papers/lure-broadcast/paper/generated/tab_unresolved_profile.tex  (response-selection check)
papers/lure-broadcast/paper/generated/tab_bound.tex        (worked disagreement bound, appendix)

Failure modes
-------------
- A stratum name that does not parse into band and rank range RAISES rather than being printed
  unformatted, since an unrecognised name means the stratum scheme changed.
- The emitted row counts are asserted against the confusion matrix in the same artefact; if the
  per-stratum cells do not sum to it, the table and the text would disagree and the run RAISES.
- The FLAGGED column is asserted to sum to `flagged_census.total`, the number the text quotes. This
  column exists because of a review in which a reader summed $N_h$ over the weak and strong bands,
  got 152, and reported it as an inconsistency against the quoted 141. The reader's arithmetic was
  wrong -- $N_h$ is the stratum population, not the flagged count, and the weak band contributes
  0+2+7 flagged out of 3+5+12 -- but the table gave them no way to see that, because the flagged
  census was not among its columns. A quantity the text quotes must be reconstructible from the
  table that is supposed to show the design.

Complexity: trivial.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "papers" / "lure-broadcast" / "results"
GEN = ROOT / "papers" / "lure-broadcast" / "paper" / "generated"


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def stratum_label(name: str) -> tuple[str, str, tuple[int, int]]:
    """'strong|600-1200' -> ('strong', '600--1200', sort key). RAISES on an unknown shape."""
    if "|" not in name:
        raise SystemExit(f"stratum {name!r} is not 'band|range'; the stratum scheme changed and the "
                         f"table layout must be revisited rather than printing the raw name")
    band, rng = name.split("|", 1)
    order = {"none": 0, "weak": 1, "strong": 2}
    if band not in order:
        raise SystemExit(f"unknown detector band {band!r} in stratum {name!r}")
    lo = rng.split("-")[0].rstrip("+")
    try:
        lo_i = int(lo)
    except ValueError:
        raise SystemExit(f"cannot order stratum {name!r}: {lo!r} is not a rank")
    return band, rng.replace("-", "--"), (order[band], lo_i)


def render_strata(doc: dict) -> str:
    rows = sorted(doc["per_stratum"], key=lambda r: stratum_label(r["stratum"])[2])
    tot = {k: 0 for k in ("TP", "FP", "FN", "TN")}
    flagged = 0
    out = []
    for r in rows:
        band, rng, _ = stratum_label(r["stratum"])
        for k in tot:
            tot[k] += r[k]
        flagged += r["flagged_population"]
        out.append(f"{band}$\\vert${rng} & {r['N_h']} & {r['flagged_population']} & {r['n_h']} & "
                   f"{r['TP']} & {r['FP']} & {r['FN']} & {r['TN']}\\\\")
    raw = doc["confusion_matrix_sampled_rows"]
    if tot != {k: raw[k] for k in tot}:
        raise SystemExit(f"per-stratum cells sum to {tot} but the artefact's sampled confusion "
                         f"matrix is {raw}; the table would contradict the text")
    census = doc["flagged_census"]["total"]
    if flagged != census:
        raise SystemExit(f"the flagged column sums to {flagged} but the flagged census is {census}; "
                         f"the table would license the very miscount it exists to prevent")
    return "\n".join(out)


def render_bound(doc: dict) -> str:
    """The inputs of the exact disagreement bound and an allocation attaining it, per flagged stratum.

    Both variants are rendered side by side: under ignorability (unlabelled flagged rows dropped) and
    in the worst case (each counted as an observed disagreement). Raises if M_h differs between the
    variants, if the strata do not sum to the flagged population, if an allocation does not sum to
    its k_max, or if an allocation leaves the data-consistent box of Theorem 1.
    """
    ign, worst = doc["ignorability"], doc["worst_case"]
    out, total_m = [], 0
    for name, (mi, si, xi), (mw, sw, xw), ki, kw in zip(
            doc["strata"], ign["inputs_M_m_x"], worst["inputs_M_m_x"], ign["allocation"], worst["allocation"]):
        if mi != mw:
            raise SystemExit(f"{name}: M_h is {mi} under ignorability but {mw} in the worst case")
        for m_h, x_h, k_h, tag in ((si, xi, ki, "ignorability"), (sw, xw, kw, "worst case")):
            if not x_h <= k_h <= mi - m_h + x_h:
                raise SystemExit(f"{name} ({tag}): k_h={k_h} outside [{x_h}, {mi - m_h + x_h}]")
        band, rng, _ = stratum_label(name)
        total_m += mi
        out.append(f"{band}$\\vert${rng} & {mi} & {si} & {xi} & {ki} & {sw} & {xw} & {kw}\\\\")
    if total_m != doc["flagged_population"]:
        raise SystemExit(f"strata sum to {total_m} flagged contracts, not {doc['flagged_population']}")
    for tag, v in (("ignorability", ign), ("worst case", worst)):
        if sum(v["allocation"]) != v["k_max"]:
            raise SystemExit(f"{tag}: allocation sums to {sum(v['allocation'])}, k_max is {v['k_max']}")
    return "\n".join(out)


def render_unresolved(doc: dict) -> str:
    """The flagged contracts the ledger probe does not positively establish, and why.

    Reported individually because an aggregate cannot distinguish a transport failure from a
    contract that answered and behaved differently. The two mean different things: the first is an
    absence of evidence, the second is evidence of something else.
    """
    reasons = {
        "no_response": "no answer",
        "address_derived": "balances differ",
    }
    out = []
    for addr, v in sorted(doc["per_contract"].items()):
        if v["verdict"] == "constant_nonzero":
            continue
        why = reasons.get(v["verdict"], v["verdict"].replace("_", " "))
        out.append(f"\\texttt{{{addr[:10]}}} & {why} & {v.get('allowance', 'n/a')}\\\\")
    if not out:
        raise SystemExit("no unresolved contracts; the table would be empty and the prose that "
                         "refers to it would be wrong")
    return "\n".join(out)


def render_residue(doc: dict) -> str:
    """One row per residue class, so the robustness diagnostic is independently inspectable."""
    out = []
    for c in doc["per_class"]:
        n = f"{c['sampled_recipients']:,}".replace(",", "{,}")
        out.append(f"{c['residue_class']} & {n} & {100 * c['active']:.2f}\\\\")
    return "\n".join(out)


def render_unresolved_profile(doc: dict) -> str:
    """Resolved against unresolved contracts, so response selection can be judged descriptively."""
    def row(label: str, g: dict) -> str:
        fb = f"{g['first_block_median']:,.0f}".replace(",", "{,}")
        tv = f"{g['transfers_median']:,.0f}".replace(",", "{,}")
        cb = f"{g['code_bytes_median']:,.0f}".replace(",", "{,}")
        return f"{label} & {g['n']} & {tv} & {fb} & {cb} & {g['eip1167_proxies']}\\\\"
    return row("Resolved", doc["resolved"]) + "\n" + row("Unresolved", doc["unresolved"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chain", default="base")
    args = ap.parse_args()

    src = RES / f"detector_design_based_{args.chain}.json"
    if not src.exists():
        raise SystemExit(f"missing {src.relative_to(ROOT)}; run evaluate_design_based.py first")
    body = render_strata(json.loads(src.read_text()))

    GEN.mkdir(parents=True, exist_ok=True)
    out = GEN / "tab_strata.tex"
    out.write_text(
        "% GENERATED FILE -- DO NOT EDIT.\n"
        "% Producer: papers/lure-broadcast/scripts/make_tables.py\n"
        f"% Source:   results/detector_design_based_{args.chain}.json\n"
        f"% git commit: {git_commit()}\n" + body + "\n")
    print(f"  {len(body.splitlines())} strata -> {out.relative_to(ROOT)}")

    fp = RES / f"forgery_probe_{args.chain}.json"
    if fp.exists():
        b = render_unresolved(json.loads(fp.read_text()))
        o = GEN / "tab_unresolved.tex"
        o.write_text(f"% GENERATED FILE -- DO NOT EDIT.\n% Source: results/forgery_probe_"
                     f"{args.chain}.json\n% git commit: {git_commit()}\n" + b + "\n")
        print(f"  {len(b.splitlines())} unresolved -> {o.relative_to(ROOT)}")

    bw = RES / "bound_worked_example.json"
    if bw.exists():
        b = render_bound(json.loads(bw.read_text()))
        o = GEN / "tab_bound.tex"
        o.write_text("% GENERATED FILE -- DO NOT EDIT.\n% Source: results/bound_worked_example.json\n"
                     f"% git commit: {git_commit()}\n" + b + "\n")
        print(f"  {len(b.splitlines())} flagged strata -> {o.relative_to(ROOT)}")

    rr = RES / f"residue_robustness_{args.chain}.json"
    if rr.exists():
        b = render_residue(json.loads(rr.read_text()))
        o = GEN / "tab_residue.tex"
        o.write_text(f"% GENERATED FILE -- DO NOT EDIT.\n% Source: results/residue_robustness_"
                     f"{args.chain}.json\n% git commit: {git_commit()}\n" + b + "\n")
        print(f"  {len(b.splitlines())} residue classes -> {o.relative_to(ROOT)}")

    up = RES / f"unresolved_profile_{args.chain}.json"
    if up.exists():
        b = render_unresolved_profile(json.loads(up.read_text()))
        o = GEN / "tab_unresolved_profile.tex"
        o.write_text(f"% GENERATED FILE -- DO NOT EDIT.\n% Source: results/unresolved_profile_"
                     f"{args.chain}.json\n% git commit: {git_commit()}\n" + b + "\n")
        print(f"  response-selection profile -> {o.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
