"""Sweep stratum granularity and let the CALIBRATION arm choose it.

Why this exists
---------------
Stratum granularity is the one threshold in the target-list overlap test that is not externally fixed, and
it trades off two failure modes in opposite directions:

  too COARSE -- addresses are not exchangeable within a stratum, the popularity confounder leaks, and the
                null manufactures O/E > 1 out of nothing. Measured on synthetic data at 2.23, and on Base
                at 1.281 for contracts that are legitimate broadcasters by construction.
  too FINE   -- a source's count in a stratum approaches the stratum's pool size, the sampler loses
                freedom, and O/E collapses to 1 for the opposite reason: no test is being performed.

So granularity cannot be chosen by taste. It is chosen by the calibration arm: the finest granularity at
which legitimate mass distributors still score O/E ~ 1 AND the null retains freedom. That criterion is
declared here, before the numbers are read, and it is the sensitivity sweep the house style requires for
every threshold.

The lure figure is reported at every granularity so a reader can see it move, but it is only INTERPRETABLE
at granularities where calibration passes. Where it does not, the ratio-of-ratios is the only quotable
quantity and it rests on the leak acting equally on both broadcast families.

Inputs
------
papers/lure-broadcast/results/separation_inputs_<chain>.json.gz  (raw covariates, so no re-scan needed)

Outputs
-------
papers/lure-broadcast/results/strata_sweep_<chain>.json (+ .manifest.json)

Failure modes
-------------
- A granularity where any source's count exceeds its stratum pool RAISES inside the null model rather than
  silently reporting a forced ratio; that granularity is recorded as unusable.
- An empty family at any granularity is reported, not skipped.

Complexity: O(granularities x universe). The closed-form expectation means no permutations are needed for
the point estimates, so the whole sweep is seconds once the inputs are loaded.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analysis"))

from src import nullmodel as nm  # noqa: E402

OUT_ROOT = ROOT / "papers" / "lure-broadcast" / "results"

#: Calibration must land inside this band for a granularity to be usable. Declared before the sweep runs.
CALIB_BAND = (0.80, 1.25)

#: (label, activity-bucket edges, number of era bins). Coarse to fine.
GRANULARITIES = [
    ("act4-era1", [1, 3, 10, 100], 1),
    ("act4-era10", [1, 3, 10, 100], 10),
    ("act8-era10", [1, 2, 3, 5, 10, 25, 100, 500], 10),
    ("act8-era25", [1, 2, 3, 5, 10, 25, 100, 500], 25),
    ("act12-era25", [1, 2, 3, 4, 5, 7, 10, 16, 25, 60, 150, 500], 25),
    ("act12-era50", [1, 2, 3, 4, 5, 7, 10, 16, 25, 60, 150, 500], 50),
    ("act16-era50", [1, 2, 3, 4, 5, 6, 8, 10, 13, 17, 25, 40, 70, 150, 400, 1000], 50),
]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def bucket(n: int, edges: list[int]) -> int:
    lo = 0
    for i, e in enumerate(edges):
        if n >= e:
            lo = i
    return lo


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chain", default="base")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = OUT_ROOT / f"strata_sweep_{args.chain}.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out.relative_to(ROOT)} exists; pass --force")

    src = OUT_ROOT / f"separation_inputs_{args.chain}.json.gz"
    print(f"  loading {src.name} ...")
    with gzip.open(src, "rt") as fh:
        d = json.load(fh)
    cov = d.get("covariates")
    if not cov:
        raise SystemExit("inputs carry no raw covariates; re-run collect_separation_inputs.py (an early "
                         "version stored only the derived stratum label, which cannot be re-binned)")
    lure = {k: set(v) for k, v in d["family_lure"].items()}
    calib = {k: set(v) for k, v in d["family_calibration"].items()}
    victims = {"poisoning": set(d["poisoning_victims"])}
    print(f"  universe {len(cov):,}  lure {len(lure)} campaigns  calibration {len(calib)} distributors  "
          f"victims {len(victims['poisoning']):,}")

    blocks = sorted(v[1] for v in cov.values())
    rows = []
    for label, edges, n_era in GRANULARITIES:
        cuts = [blocks[int(q * (len(blocks) - 1))] for q in
                ((i + 1) / n_era for i in range(n_era - 1))] if n_era > 1 else []
        strata = {}
        for a, (act, fb, sent) in cov.items():
            e = sum(1 for c in cuts if fb > c)
            strata[a] = f"{sent}|{bucket(act, edges)}|{e}"
        pools = collections.Counter(strata.values())

        rec: dict = {"granularity": label, "n_strata": len(pools),
                     "median_pool": sorted(pools.values())[len(pools) // 2]}
        for name, fam in (("lure", lure), ("calibration", calib)):
            try:
                cA = nm.per_stratum_counts(fam, strata)
                cB = nm.per_stratum_counts(victims, strata)
                E = nm.expected_overlap_closed_form(cA, cB, dict(pools))
                O = nm.cross_overlap(fam, victims)
                rec[name] = {"observed": O, "expected": E, "oe": (O / E) if E else None}
            except ValueError as exc:
                rec[name] = {"error": str(exc)[:120]}
        cal = rec.get("calibration", {}).get("oe")
        lu = rec.get("lure", {}).get("oe")
        rec["calibration_passes"] = bool(cal and CALIB_BAND[0] <= cal <= CALIB_BAND[1])
        rec["theta_ratio_of_ratios"] = (lu / cal) if (lu and cal) else None
        rows.append(rec)
        print(f"  {label:14s} strata {len(pools):6,}  median pool {rec['median_pool']:>8,}  "
              f"lure O/E {(f'{lu:6.3f}' if lu else '   n/a')}  "
              f"calib O/E {(f'{cal:6.3f}' if cal else '   n/a')}  "
              f"{'PASS' if rec['calibration_passes'] else 'leak'}  "
              f"theta {th:.3f}" if (th := rec['theta_ratio_of_ratios']) else "theta n/a")

    usable = [r for r in rows if r["calibration_passes"]]
    chosen = usable[-1] if usable else None
    doc = {
        "chain": args.chain,
        "calibration_band": list(CALIB_BAND),
        "criterion": ("the FINEST granularity at which calibration lands inside the band and the null "
                      "retains freedom; declared before the sweep was run"),
        "sweep": rows,
        "chosen": chosen,
        "verdict": ("no granularity passes calibration; only the ratio-of-ratios is quotable and the leak "
                    "must be disclosed" if not chosen else
                    f"granularity {chosen['granularity']} chosen; lure O/E is interpretable there"),
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    (OUT_ROOT / f"strata_sweep_{args.chain}.manifest.json").write_text(json.dumps({
        "producer": "papers/lure-broadcast/scripts/sweep_strata.py",
        "git_commit": git_commit(), "input": src.name,
    }, indent=1) + "\n")
    print(f"\n  {doc['verdict']}")
    print(f"  -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
