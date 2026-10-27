"""Interval and p-value for Theta, the lure-vs-legitimate-broadcast overlap ratio.

Why this exists
---------------
The point estimate Theta = 1.06 is exact but meaningless without an interval: 1.06 cannot be distinguished
from 1.00 by eye, and that distinction IS the claim ("no target sharing beyond shared attractiveness").

What the uncertainty is, and what it is not
-------------------------------------------
A permutation of the NULL model does not give an interval on Theta. Theta is a ratio of two observed-over-
expected quantities computed from the same fixed data, so the null distribution of the overlap answers a
different question ("is this overlap extreme?") than the one that matters here ("do lure campaigns differ
from legitimate broadcasters?").

Two inferences are run instead, and they answer complementary questions:

  BOOTSTRAP OVER SOURCES -- resample campaigns and distributors with replacement and recompute Theta. This
      reflects between-source variability, i.e. how much Theta would move if a different set of campaigns
      and distributors had been observed. It gives the interval.

  LABEL PERMUTATION -- pool all sources, randomly reassign which are "lure" and which are "calibration"
      keeping the group sizes fixed, and recompute Theta. Under the hypothesis that the two families are
      exchangeable with respect to their victim overlap, Theta should centre on 1. This gives the p-value,
      and it is assumption-light: it tests the lure/calibration distinction directly rather than assuming
      a parametric form.

Why this is affordable
----------------------
Both the observed overlap and its null expectation are ADDITIVE over sources. For source a with
per-stratum counts k_{a,s}, and victim counts K_{V,s} in stratum s of pool size N_s:

    O(family) = sum over a of |L_a & V|            =: sum of hits_a
    E(family) = sum over a of sum_s k_{a,s} K_{V,s} / N_s  =: sum of e_a

so each source reduces to the pair (hits_a, e_a), computed once. Every bootstrap or permutation replicate
is then a sum over a resampled index set -- O(n_sources), not O(list size). 797 sources make 10,000
replicates instant.

Inputs
------
papers/lure-broadcast/results/separation_inputs_<chain>.json.gz
The stratum granularity is taken from the sweep's finest setting; Theta was shown stable to 2% across a
200x range, so the choice is not load-bearing, but it is recorded.

Outputs
-------
papers/lure-broadcast/results/theta_<chain>.json (+ .manifest.json)

Failure modes
-------------
- A family with fewer than 2 sources RAISES: a bootstrap over one source is not an interval.
- Any source with e_a == 0 is reported and excluded from both arms, never silently kept (it would make the
  ratio undefined for any replicate containing it).
- The p-value is reported at its resolution bound rather than as 0.

Complexity: O(universe) once, then O(replicates x n_sources).
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analysis"))

from src import stats  # noqa: E402

OUT_ROOT = ROOT / "papers" / "lure-broadcast" / "results"
#: Finest granularity from the sweep; Theta varied by 2% across a 200x range so this is not load-bearing.
ACT_EDGES = [1, 2, 3, 4, 5, 6, 8, 10, 13, 17, 25, 40, 70, 150, 400, 1000]
N_ERA = 50


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
    ap.add_argument("--replicates", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = OUT_ROOT / f"theta_{args.chain}.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out.relative_to(ROOT)} exists; pass --force")

    with gzip.open(OUT_ROOT / f"separation_inputs_{args.chain}.json.gz", "rt") as fh:
        d = json.load(fh)
    cov = d["covariates"]
    victims = set(d["poisoning_victims"])

    blocks = sorted(v[1] for v in cov.values())
    cuts = [blocks[int(q * (len(blocks) - 1))] for q in ((i + 1) / N_ERA for i in range(N_ERA - 1))]
    strata = {}
    for a, (act, fb, sent) in cov.items():
        strata[a] = f"{sent}|{bucket(act, ACT_EDGES)}|{sum(1 for c in cuts if fb > c)}"
    pools = collections.Counter(strata.values())
    kv: dict[str, int] = collections.Counter(strata[a] for a in victims if a in strata)

    # Per-stratum expected-overlap weight: K_{V,s} / N_s.
    w = {s: kv.get(s, 0) / n for s, n in pools.items()}

    def reduce_family(fam: dict[str, list[str]]) -> list[tuple[str, int, float]]:
        """(name, hits, expected) per source -- the sufficient pair for every replicate."""
        red = []
        for name, lst in fam.items():
            hits = sum(1 for a in lst if a in victims)
            e = 0.0
            for a in lst:
                s = strata.get(a)
                if s is not None:
                    e += w[s]
            red.append((name, hits, e))
        return red

    lure = reduce_family(d["family_lure"])
    calib = reduce_family(d["family_calibration"])
    dropped = [n for n, _, e in lure + calib if e <= 0]
    lure = [r for r in lure if r[2] > 0]
    calib = [r for r in calib if r[2] > 0]
    if len(lure) < 2 or len(calib) < 2:
        raise SystemExit("each family needs >= 2 usable sources for a bootstrap")

    def theta(idx_l, idx_c, L, C) -> float | None:
        oh = sum(L[i][1] for i in idx_l); oe = sum(L[i][2] for i in idx_l)
        ch = sum(C[i][1] for i in idx_c); ce = sum(C[i][2] for i in idx_c)
        if oe <= 0 or ce <= 0 or ch <= 0:
            return None
        return (oh / oe) / (ch / ce)

    obs = theta(range(len(lure)), range(len(calib)), lure, calib)
    oe_l = sum(r[1] for r in lure) / sum(r[2] for r in lure)
    oe_c = sum(r[1] for r in calib) / sum(r[2] for r in calib)

    rng = random.Random(args.seed)
    # --- bootstrap over sources -> interval
    boots = []
    nl, nc = len(lure), len(calib)
    for _ in range(args.replicates):
        il = [rng.randrange(nl) for _ in range(nl)]
        ic = [rng.randrange(nc) for _ in range(nc)]
        t = theta(il, ic, lure, calib)
        if t is not None:
            boots.append(t)
    boots.sort()
    lo = boots[int(0.025 * (len(boots) - 1))]
    hi = boots[int(0.975 * (len(boots) - 1))]

    # --- label permutation -> p-value
    pooled = lure + calib
    perms = []
    for _ in range(args.replicates):
        idx = list(range(len(pooled)))
        rng.shuffle(idx)
        il, ic = idx[:nl], idx[nl:]
        t = theta(il, ic, pooled, pooled)
        if t is not None:
            perms.append(t)
    at_least = sum(1 for t in perms if t >= obs)
    p = (at_least + 1) / (len(perms) + 1)

    ls_l = sorted(len(v) for v in d["family_lure"].values())
    ls_c = sorted(len(v) for v in d["family_calibration"].values())
    doc = {
        "chain": args.chain,
        "granularity": {"act_edges": ACT_EDGES, "n_era": N_ERA, "n_strata": len(pools),
                        "note": "Theta varied by 2% across a 200x granularity range; see strata_sweep"},
        "oe_lure": oe_l, "oe_calibration": oe_c,
        "theta": obs,
        "bootstrap_over_sources": {
            "replicates": len(boots), "ci95": [lo, hi],
            "excludes_one": not (lo <= 1.0 <= hi),
            "interpretation": ("between-source variability: how much Theta would move if a different set "
                               "of campaigns and distributors had been observed"),
        },
        "label_permutation": {
            "replicates": len(perms), "p_value": p,
            "p_at_resolution_bound": at_least == 0,
            "null_median": perms[len(perms) // 2] if perms else None,
            "interpretation": ("pool all sources and reassign the lure/calibration labels at fixed group "
                               "sizes; tests the distinction directly"),
        },
        "sources": {"lure": len(lure), "calibration": len(calib), "dropped_zero_expected": dropped},
        "list_size_distribution": {
            "lure": {"min": ls_l[0], "median": ls_l[len(ls_l) // 2], "max": ls_l[-1]},
            "calibration": {"min": ls_c[0], "median": ls_c[len(ls_c) // 2], "max": ls_c[-1]},
            "caveat": ("label permutation assumes the two families are exchangeable under the null. They "
                       "differ in list size; O/E normalises for list size and stratum profile, but the "
                       "distributions are reported so a reader can judge the assumption."),
        },
        "wilson_on_hits": {
            "lure": list(stats.wilson(sum(r[1] for r in lure),
                                      sum(len(v) for v in d["family_lure"].values()))),
            "calibration": list(stats.wilson(sum(r[1] for r in calib),
                                             sum(len(v) for v in d["family_calibration"].values()))),
        },
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    (OUT_ROOT / f"theta_{args.chain}.manifest.json").write_text(json.dumps({
        "producer": "papers/lure-broadcast/scripts/theta_inference.py",
        "git_commit": git_commit(), "seed": args.seed, "replicates": args.replicates,
    }, indent=1) + "\n")

    print(f"  strata {len(pools):,}   sources: lure {len(lure)}, calibration {len(calib)}"
          + (f", dropped {len(dropped)} with zero expectation" if dropped else ""))
    print(f"  O/E lure        {oe_l:.4f}")
    print(f"  O/E calibration {oe_c:.4f}")
    print(f"  THETA           {obs:.4f}   bootstrap 95% CI [{lo:.4f}, {hi:.4f}]"
          f"   {'EXCLUDES 1.0' if not (lo <= 1.0 <= hi) else 'includes 1.0'}")
    print(f"  label-permutation p = {p:.4g}"
          f"{' (resolution bound)' if at_least == 0 else ''}, null median "
          f"{doc['label_permutation']['null_median']:.4f}")
    print(f"  list sizes: lure median {ls_l[len(ls_l)//2]:,}, calibration median {ls_c[len(ls_c)//2]:,}")
    print(f"  -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
