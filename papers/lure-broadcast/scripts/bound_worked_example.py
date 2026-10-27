"""A hand-checkable worked calculation of the exact disagreement bound (IEEE review R2.3; plan WP9.3).

Reviewer 2 asked for enough information to verify the reported bound: its estimand, design and
calculation. For each of the two published variants (ignorability, worst case) this script writes:
  * the per-stratum inputs (M_h flagged population, m_h flagged rows in the sample, x_h observed
    disagreements), read from the detector evaluation artefact;
  * the allocation K = (K_1..K_H) attaining the maximum total, with its tail probability
    P_K(sum X_h <= sum x_h), which must exceed alpha;
  * the LARGEST such tail over every data-consistent allocation of total k_max + 1, which must be
    at most alpha. That is the evidence that k_max + 1 lies outside the confidence region.
Everything is recomputed by exhaustive enumeration over the box x_h <= K_h <= M_h - m_h + x_h,
independently of stats.stratified_exact_upper_bound, and must agree with the value it reproduces.

Input:  papers/lure-broadcast/results/detector_design_based_base.json
Output: papers/lure-broadcast/results/bound_worked_example.json (+ .manifest.json)
Complexity: prod_h (M_h - m_h + 1) allocations. A few thousand here; milliseconds.
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom

ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "papers" / "lure-broadcast"
sys.path.insert(0, str(ROOT / "analysis"))

from src import stats  # noqa: E402

SRC = PAPER / "results" / "detector_design_based_base.json"
OUT = PAPER / "results" / "bound_worked_example.json"
ALPHA = 0.05


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def tail(strata: list[tuple[int, int, int]], ks: tuple[int, ...]) -> float:
    """P_K(sum_h X_h <= sum_h x_h) for X_h ~ Hyp(M_h, K_h, m_h), by full convolution."""
    t = sum(x for _, _, x in strata)
    pmf = np.array([1.0])
    for (big_m, m, _), k in zip(strata, ks):
        pmf = np.convolve(pmf, hypergeom.pmf(np.arange(m + 1), big_m, k, m))
    return float(pmf[: t + 1].sum())


def worked(strata: list[tuple[int, int, int]], alpha: float = ALPHA) -> dict:
    """The admitted maximum and the strongest rejected allocation one above it."""
    box = [range(x, big_m - m + x + 1) for big_m, m, x in strata]
    best: tuple[int, tuple[int, ...], float] | None = None
    by_total: dict[int, float] = {}
    for ks in itertools.product(*box):
        p = tail(strata, ks)
        s = sum(ks)
        by_total[s] = max(by_total.get(s, 0.0), p)
        if p > alpha and (best is None or s > best[0] or (s == best[0] and p > best[2])):
            best = (s, ks, p)
    assert best is not None, "K = x always reproduces the observation"
    k_max, alloc, p_at = best
    return {"k_max": k_max, "allocation": list(alloc),
            "allocation_note": ("one allocation attaining k_max, the one with the largest tail; "
                                "others may attain it too, so it need not match the library's"),
            "tail_at_allocation": p_at,
            "max_tail_at_k_max_plus_1": by_total.get(k_max + 1, 0.0),
            "allocations_enumerated": int(np.prod([len(r) for r in box]))}


def designs(doc: dict) -> dict[str, dict]:
    """The two published inputs, per flagged stratum, in a fixed stratum order."""
    census = doc["flagged_census"]["per_stratum"]
    per = {s["stratum"]: s for s in doc["per_stratum"]}
    excl = {k: len(v) for k, v in doc["excluded_flagged"]["rows"].items()}
    names = sorted(census)
    agreed = [(census[s], per[s]["TP"] + per[s]["FP"], per[s]["FP"]) for s in names]
    worst = [(census[s], per[s]["TP"] + per[s]["FP"] + excl.get(s, 0), per[s]["FP"] + excl.get(s, 0))
             for s in names]
    return {"strata": names,
            "ignorability": agreed,
            "worst_case": worst}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if OUT.exists() and not args.force:
        raise SystemExit(f"{OUT.relative_to(ROOT)} exists; pass --force")
    doc = json.loads(SRC.read_text())
    d = designs(doc)
    total = sum(m for m, _, _ in d["ignorability"])
    out = {"alpha": ALPHA, "flagged_population": total, "strata": d["strata"],
           "estimand": ("the number of flagged contracts, within the examined Base frame, whose "
                        "operational-codebook label disagrees with the rule; an upper 95% confidence "
                        "bound on that count, not a false-positive rate against abuse"),
           "design": ("flagged contracts are a census per stratum (M_h known); labels are a "
                      "stratified sample without replacement; the region is the inversion of the "
                      "exact test on the total, intersected with the data-consistent box")}
    for variant in ("ignorability", "worst_case"):
        strata = d[variant]
        w = worked(strata)
        lib = stats.stratified_exact_upper_bound(strata)[0]
        if lib != w["k_max"]:
            raise ValueError(f"{variant}: enumeration gives {w['k_max']} but the library gives {lib}")
        if not (w["tail_at_allocation"] > ALPHA >= w["max_tail_at_k_max_plus_1"]):
            raise ValueError(f"{variant}: the admitted/rejected pair does not straddle alpha")
        out[variant] = {"inputs_M_m_x": [list(s) for s in strata], **w, "theta": w["k_max"] / total}
        print(f"  {variant:13s} k_max {w['k_max']:3d}/{total} = {w['k_max'] / total:.4f}  "
              f"tail at allocation {w['tail_at_allocation']:.4f} > {ALPHA} >= "
              f"{w['max_tail_at_k_max_plus_1']:.4f} (best at k_max+1)")
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    OUT.with_suffix(".manifest.json").write_text(json.dumps({
        "producer": "papers/lure-broadcast/scripts/bound_worked_example.py",
        "git_commit": git_commit(), "source": str(SRC.relative_to(ROOT))}, indent=1) + "\n")
    print(f"  -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
